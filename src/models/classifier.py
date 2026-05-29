"""Image classification: torchvision-backed classifier wrapper.

torchvision is imported lazily inside the methods that build/run the model, so
importing this module is cheap and never triggers a weights download.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from PIL import Image

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Mean/std for ImageNet-pretrained torchvision models.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Models the wrapper knows how to build via torchvision.
_SUPPORTED_MODELS = ["resnet50", "efficientnet_b0", "mobilenet_v3_small"]


@dataclass
class Prediction:
    """A single classification prediction."""

    label: str
    confidence: float
    class_index: int


@dataclass
class ClassificationResult:
    """Result of classifying a single image."""

    top_predictions: list[Prediction]
    latency_ms: float
    model_name: str


def softmax_topk(logits, k: int) -> tuple[list[float], list[int]]:
    """Return (probabilities, indices) of the top-k classes from a 1-D logit tensor/array.

    Uses a numerically stable softmax. Works on torch tensors or numpy arrays.
    """
    import numpy as np

    arr = logits.detach().cpu().numpy() if hasattr(logits, "detach") else np.asarray(logits)
    arr = arr.astype("float64").ravel()
    shifted = arr - arr.max()
    exp = np.exp(shifted)
    probs = exp / exp.sum()

    k = min(k, probs.shape[0])
    top_idx = np.argsort(probs)[::-1][:k]
    return [float(probs[i]) for i in top_idx], [int(i) for i in top_idx]


class ImageClassifier:
    """Wrapper around a torchvision classification model.

    The underlying model is built lazily on first use. An already-constructed
    torch model can be injected (e.g. in tests) to avoid any download.
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        device: str = "cpu",
        top_k: int = 5,
        labels: dict[int, str] | None = None,
        model: Any = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.top_k = top_k
        self.labels = labels or {}
        self._model = model
        self._transform = None

    @staticmethod
    def available_models() -> list[str]:
        return list(_SUPPORTED_MODELS)

    def _build_transform(self):
        if self._transform is not None:
            return self._transform
        from torchvision import transforms

        self._transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        return self._transform

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        import torch
        from torchvision import models as tv_models

        if self.model_name == "resnet50":
            model = tv_models.resnet50(weights="DEFAULT")
        elif self.model_name == "efficientnet_b0":
            model = tv_models.efficientnet_b0(weights="DEFAULT")
        elif self.model_name == "mobilenet_v3_small":
            model = tv_models.mobilenet_v3_small(weights="DEFAULT")
        else:
            model = tv_models.resnet50(weights="DEFAULT")
        model.eval()
        self._model = model.to(torch.device(self.device))
        return self._model

    @property
    def model(self) -> Any:
        return self._ensure_model()

    def get_torch_model(self) -> Any:
        return self._ensure_model()

    def _preprocess(self, image: Image.Image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self._build_transform()(image)

    def predict(self, image: Image.Image, top_k: int | None = None) -> ClassificationResult:
        import torch

        k = top_k or self.top_k
        model = self._ensure_model()
        start = time.perf_counter()
        tensor = self._preprocess(image).unsqueeze(0).to(torch.device(self.device))
        with torch.no_grad():
            logits = model(tensor)[0]
        probs, indices = softmax_topk(logits, k)
        predictions = [
            Prediction(
                label=self.labels.get(idx, f"class_{idx}"),
                confidence=round(prob, 6),
                class_index=idx,
            )
            for prob, idx in zip(probs, indices)
        ]
        latency_ms = (time.perf_counter() - start) * 1000
        return ClassificationResult(
            top_predictions=predictions,
            latency_ms=round(latency_ms, 2),
            model_name=self.model_name,
        )

    def predict_batch(
        self, images: list[Image.Image], top_k: int | None = None
    ) -> list[ClassificationResult]:
        return [self.predict(img, top_k=top_k) for img in images]
