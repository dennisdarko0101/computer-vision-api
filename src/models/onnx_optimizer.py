"""ONNX export, optimization, and inference.

Heavy backends (torch.onnx, onnxruntime, onnx) are imported lazily inside methods so
importing this module is cheap and works even when those packages are absent.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Map a friendly optimization level to an onnxruntime GraphOptimizationLevel.
_GRAPH_OPT_LEVELS = {
    "disabled": "ORT_DISABLE_ALL",
    "basic": "ORT_ENABLE_BASIC",
    "extended": "ORT_ENABLE_EXTENDED",
    "all": "ORT_ENABLE_ALL",
}


@dataclass
class BenchmarkResult:
    """PyTorch vs ONNX latency comparison."""

    pytorch_mean_ms: float
    onnx_mean_ms: float
    speedup: float
    n_runs: int


class ONNXOptimizer:
    """Export PyTorch models to ONNX and run graph optimization."""

    def __init__(self, optimization_level: str = "all") -> None:
        self.optimization_level = optimization_level

    def export_to_onnx(
        self,
        model: Any,
        input_shape: tuple[int, ...],
        output_path: str | Path,
    ) -> str:
        """Export a torch model to ONNX. Returns the output path as a string.

        Input tensor is named ``input`` and output ``output``, with a dynamic batch axis.
        """
        import torch

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model.eval()
        dummy_input = torch.randn(*input_shape)

        export_kwargs: dict[str, Any] = dict(
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            opset_version=13,
        )
        # Prefer the legacy TorchScript exporter when available: it depends only on the
        # ``onnx`` package, avoiding the newer dynamo path's extra ``onnxscript`` requirement.
        try:
            torch.onnx.export(model, dummy_input, str(output_path), dynamo=False, **export_kwargs)
        except TypeError:
            torch.onnx.export(model, dummy_input, str(output_path), **export_kwargs)
        logger.info("onnx_exported", path=str(output_path))
        return str(output_path)

    def optimize_onnx(self, onnx_path: str | Path) -> str:
        """Produce an optimized copy of an ONNX model.

        Uses onnxruntime's offline graph optimization when available; otherwise
        falls back to copying the model so the pipeline still yields an
        ``*_optimized.onnx`` artifact.
        """
        onnx_path = Path(onnx_path)
        optimized_path = onnx_path.with_name(f"{onnx_path.stem}_optimized.onnx")

        try:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            level_name = _GRAPH_OPT_LEVELS.get(self.optimization_level, "ORT_ENABLE_ALL")
            sess_options.graph_optimization_level = getattr(
                ort.GraphOptimizationLevel, level_name
            )
            sess_options.optimized_model_filepath = str(optimized_path)
            # Creating the session writes the optimized model to disk.
            ort.InferenceSession(str(onnx_path), sess_options)
        except Exception as e:  # pragma: no cover - depends on ort build
            logger.warning("onnx_optimize_fallback", error=str(e))
            if not optimized_path.exists():
                shutil.copyfile(onnx_path, optimized_path)

        if not optimized_path.exists():
            shutil.copyfile(onnx_path, optimized_path)

        logger.info("onnx_optimized", path=str(optimized_path))
        return str(optimized_path)


class ONNXPredictor:
    """Run inference with an ONNX model via onnxruntime."""

    def __init__(self, onnx_path: str | Path, device: str = "cpu") -> None:
        import onnxruntime as ort

        self.onnx_path = str(onnx_path)
        self.device = device

        providers = ["CPUExecutionProvider"]
        if device.startswith("cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(self.onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Run a single inference. A 3-D (C,H,W) input is expanded to a batch of 1."""
        arr = np.asarray(inputs, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[np.newaxis, ...]
        outputs = self.session.run([self.output_name], {self.input_name: arr})
        return outputs[0]

    def predict_batch(self, inputs: np.ndarray) -> np.ndarray:
        """Run inference on a pre-batched (N,C,H,W) array."""
        arr = np.asarray(inputs, dtype=np.float32)
        outputs = self.session.run([self.output_name], {self.input_name: arr})
        return outputs[0]


def benchmark_pytorch_vs_onnx(
    torch_model: Any,
    onnx_predictor: ONNXPredictor,
    input_shape: tuple[int, ...],
    n_runs: int = 100,
    warmup_runs: int = 10,
) -> BenchmarkResult:
    """Compare mean inference latency of a torch model against its ONNX export."""
    import torch

    torch_model.eval()
    dummy = torch.randn(*input_shape)
    dummy_np = dummy.numpy().astype(np.float32)

    # Warmup
    for _ in range(warmup_runs):
        with torch.no_grad():
            torch_model(dummy)
        onnx_predictor.predict_batch(dummy_np)

    # PyTorch timing
    pt_times: list[float] = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            torch_model(dummy)
        pt_times.append((time.perf_counter() - start) * 1000)

    # ONNX timing
    onnx_times: list[float] = []
    for _ in range(n_runs):
        start = time.perf_counter()
        onnx_predictor.predict_batch(dummy_np)
        onnx_times.append((time.perf_counter() - start) * 1000)

    pytorch_mean = float(np.mean(pt_times))
    onnx_mean = float(np.mean(onnx_times))
    speedup = pytorch_mean / onnx_mean if onnx_mean > 0 else 0.0

    return BenchmarkResult(
        pytorch_mean_ms=round(pytorch_mean, 4),
        onnx_mean_ms=round(onnx_mean, 4),
        speedup=round(speedup, 4),
        n_runs=n_runs,
    )
