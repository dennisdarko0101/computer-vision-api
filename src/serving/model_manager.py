"""Model lifecycle management with LRU eviction and multi-backend support."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.config.settings import Settings, get_settings
from src.models.classifier import ImageClassifier
from src.models.detector import ObjectDetector
from src.models.embedder import ImageEmbedder
from src.models.onnx_optimizer import ONNXOptimizer, ONNXPredictor
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelInfo:
    """Metadata about a loaded model."""
    name: str
    task: str  # "classify", "detect", "embed"
    backend: str  # "pytorch", "onnx"
    device: str
    loaded_at: float = 0.0
    last_used: float = 0.0
    inference_count: int = 0


class ModelManager:
    """Manage loading, caching, and serving of CV models with LRU eviction."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._models: OrderedDict[str, Any] = OrderedDict()
        self._model_info: dict[str, ModelInfo] = {}
        self._max_models = self.settings.MAX_LOADED_MODELS
        self._onnx_optimizer = ONNXOptimizer(self.settings.ONNX_OPTIMIZATION_LEVEL)
        self._onnx_predictors: dict[str, ONNXPredictor] = {}

    def _make_key(self, name: str, task: str, backend: str = "pytorch") -> str:
        return f"{task}:{name}:{backend}"

    def load_model(
        self,
        name: str,
        task: str = "classify",
        backend: str = "pytorch",
        device: str | None = None,
    ) -> Any:
        """Load a model into memory. Evicts LRU if at capacity."""
        key = self._make_key(name, task, backend)
        device = device or self.settings.DEVICE

        if key in self._models:
            # Move to end (most recently used)
            self._models.move_to_end(key)
            self._model_info[key].last_used = time.time()
            logger.info("model_cache_hit", key=key)
            return self._models[key]

        # Evict if at capacity
        while len(self._models) >= self._max_models:
            evicted_key, _ = self._models.popitem(last=False)
            del self._model_info[evicted_key]
            if evicted_key in self._onnx_predictors:
                del self._onnx_predictors[evicted_key]
            logger.info("model_evicted", key=evicted_key)

        logger.info("loading_model", name=name, task=task, backend=backend, device=device)

        if backend == "onnx":
            model = self._load_onnx_model(name, task, device)
        else:
            model = self._load_pytorch_model(name, task, device)

        self._models[key] = model
        now = time.time()
        self._model_info[key] = ModelInfo(
            name=name,
            task=task,
            backend=backend,
            device=device,
            loaded_at=now,
            last_used=now,
        )

        # Warmup
        if self.settings.MODEL_WARMUP:
            self._warmup(model, task)

        return model

    def _load_pytorch_model(self, name: str, task: str, device: str) -> Any:
        if task == "classify":
            return ImageClassifier(model_name=name, device=device)
        elif task == "detect":
            return ObjectDetector(model_name=name, device=device)
        elif task == "embed":
            return ImageEmbedder(model_name=name, device=device)
        else:
            raise ValueError(f"Unknown task: {task}")

    def _load_onnx_model(self, name: str, task: str, device: str) -> ONNXPredictor:
        """Load an ONNX model. Export from PyTorch if ONNX file doesn't exist."""
        model_dir = self.settings.MODEL_DIR / "onnx"
        onnx_path = model_dir / f"{name}_{task}.onnx"

        if not onnx_path.exists():
            logger.info("exporting_to_onnx", name=name, task=task)
            pytorch_model = self._load_pytorch_model(name, task, device)
            torch_model = pytorch_model.get_torch_model() if hasattr(pytorch_model, "get_torch_model") else pytorch_model.model

            input_shape = (1, 3, 224, 224)
            self._onnx_optimizer.export_to_onnx(torch_model, input_shape, onnx_path)

        predictor = ONNXPredictor(onnx_path, device=device)
        key = self._make_key(name, task, "onnx")
        self._onnx_predictors[key] = predictor
        return predictor

    def _warmup(self, model: Any, task: str) -> None:
        """Run a warmup inference."""
        dummy = Image.new("RGB", (224, 224), color="red")
        try:
            if task == "classify" and hasattr(model, "predict"):
                model.predict(dummy)
            elif task == "detect" and hasattr(model, "detect"):
                model.detect(dummy)
            elif task == "embed" and hasattr(model, "embed"):
                model.embed(dummy)
            elif hasattr(model, "predict"):
                dummy_np = np.random.randn(1, 3, 224, 224).astype(np.float32)
                model.predict(dummy_np)
        except Exception as e:
            logger.warning("warmup_failed", error=str(e))

    def unload_model(self, name: str, task: str = "classify", backend: str = "pytorch") -> None:
        """Remove a model from memory."""
        key = self._make_key(name, task, backend)
        if key in self._models:
            del self._models[key]
            del self._model_info[key]
            if key in self._onnx_predictors:
                del self._onnx_predictors[key]
            logger.info("model_unloaded", key=key)

    def get_model(self, name: str, task: str = "classify", backend: str = "pytorch") -> Any:
        """Get a loaded model, loading it if not already in memory."""
        key = self._make_key(name, task, backend)
        if key not in self._models:
            return self.load_model(name, task, backend)

        self._models.move_to_end(key)
        self._model_info[key].last_used = time.time()
        self._model_info[key].inference_count += 1
        return self._models[key]

    def list_models(self) -> list[ModelInfo]:
        """List all currently loaded models."""
        return list(self._model_info.values())

    def is_loaded(self, name: str, task: str = "classify", backend: str = "pytorch") -> bool:
        """Check if a model is loaded."""
        return self._make_key(name, task, backend) in self._models

    @property
    def loaded_count(self) -> int:
        return len(self._models)

    def clear(self) -> None:
        """Unload all models."""
        self._models.clear()
        self._model_info.clear()
        self._onnx_predictors.clear()
        logger.info("all_models_cleared")
