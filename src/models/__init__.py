"""Model wrappers and computer-vision primitives.

Exposes the detection geometry helpers (BBox, compute_iou, non_max_suppression) and
the result dataclasses eagerly because they are dependency-free. The torchvision-backed
model classes are imported lazily via ``__getattr__`` so that simply importing this
package never pulls in torchvision or downloads weights.
"""

from __future__ import annotations

from src.models.detector import (
    BBox,
    DetectionResult,
    compute_iou,
    non_max_suppression,
)
from src.models.embedder import SimilarityResult

__all__ = [
    "BBox",
    "DetectionResult",
    "compute_iou",
    "non_max_suppression",
    "SimilarityResult",
    "ImageClassifier",
    "ClassificationResult",
    "Prediction",
    "ObjectDetector",
    "ImageEmbedder",
    "SimilaritySearch",
    "ONNXOptimizer",
    "ONNXPredictor",
    "benchmark_pytorch_vs_onnx",
]

_LAZY = {
    "ImageClassifier": "src.models.classifier",
    "ClassificationResult": "src.models.classifier",
    "Prediction": "src.models.classifier",
    "ObjectDetector": "src.models.detector",
    "ImageEmbedder": "src.models.embedder",
    "SimilaritySearch": "src.models.embedder",
    "ONNXOptimizer": "src.models.onnx_optimizer",
    "ONNXPredictor": "src.models.onnx_optimizer",
    "benchmark_pytorch_vs_onnx": "src.models.onnx_optimizer",
}


def __getattr__(name: str):
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, name)
