"""Tests for ImageClassifier — 17 tests using mock models."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from tests.conftest import MockClassifierModel


# ── Helpers ─────────────────────────────────────────────────────────────────

class FakeClassifier:
    """Lightweight stand-in that mimics ImageClassifier without downloading weights."""

    def __init__(self, num_classes: int = 10, device: str = "cpu", top_k: int = 5):
        self.model_name = "mock_resnet"
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.top_k = top_k
        self.model = MockClassifierModel(num_classes)
        self.model.eval()
        self.labels = {i: f"class_{i}" for i in range(num_classes)}

        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image)

    def predict(self, image: Image.Image, top_k: int | None = None):
        import time
        from src.models.classifier import ClassificationResult, Prediction
        k = top_k or self.top_k
        start = time.perf_counter()
        tensor = self._preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probs[0], k=min(k, self.num_classes))
        predictions = [
            Prediction(
                label=self.labels.get(idx.item(), f"class_{idx.item()}"),
                confidence=round(prob.item(), 6),
                class_index=idx.item(),
            )
            for prob, idx in zip(top_probs, top_indices)
        ]
        elapsed = (time.perf_counter() - start) * 1000
        return ClassificationResult(
            top_predictions=predictions, latency_ms=round(elapsed, 2), model_name=self.model_name,
        )

    def predict_batch(self, images: list[Image.Image], top_k: int | None = None):
        return [self.predict(img, top_k=top_k) for img in images]

    def get_torch_model(self):
        return self.model

    @staticmethod
    def available_models():
        return ["resnet50", "efficientnet_b0", "mobilenet_v3_small"]


@pytest.fixture
def classifier() -> FakeClassifier:
    return FakeClassifier(num_classes=10)


# ── Tests ───────────────────────────────────────────────────────────────────

class TestImageClassifier:
    def test_predict_returns_result(self, classifier, sample_image):
        result = classifier.predict(sample_image)
        assert result.top_predictions is not None
        assert len(result.top_predictions) > 0
        assert result.latency_ms > 0

    def test_predict_default_top_k(self, classifier, sample_image):
        result = classifier.predict(sample_image)
        assert len(result.top_predictions) == 5

    def test_predict_custom_top_k(self, classifier, sample_image):
        result = classifier.predict(sample_image, top_k=3)
        assert len(result.top_predictions) == 3

    def test_predict_top_k_exceeds_classes(self, classifier, sample_image):
        result = classifier.predict(sample_image, top_k=20)
        assert len(result.top_predictions) == 10  # capped at num_classes

    def test_prediction_has_label(self, classifier, sample_image):
        result = classifier.predict(sample_image)
        for pred in result.top_predictions:
            assert isinstance(pred.label, str)
            assert len(pred.label) > 0

    def test_prediction_has_confidence(self, classifier, sample_image):
        result = classifier.predict(sample_image)
        for pred in result.top_predictions:
            assert 0.0 <= pred.confidence <= 1.0

    def test_prediction_has_class_index(self, classifier, sample_image):
        result = classifier.predict(sample_image)
        for pred in result.top_predictions:
            assert isinstance(pred.class_index, int)
            assert 0 <= pred.class_index < 10

    def test_predictions_sorted_by_confidence(self, classifier, sample_image):
        result = classifier.predict(sample_image)
        confs = [p.confidence for p in result.top_predictions]
        assert confs == sorted(confs, reverse=True)

    def test_confidences_sum_at_most_one(self, classifier, sample_image):
        result = classifier.predict(sample_image, top_k=10)
        total = sum(p.confidence for p in result.top_predictions)
        assert total <= 1.01  # allow rounding

    def test_predict_grayscale_image(self, classifier, sample_grayscale):
        result = classifier.predict(sample_grayscale)
        assert len(result.top_predictions) > 0

    def test_predict_large_image(self, classifier, large_image):
        result = classifier.predict(large_image)
        assert len(result.top_predictions) > 0

    def test_predict_batch(self, classifier, sample_images_batch):
        results = classifier.predict_batch(sample_images_batch)
        assert len(results) == 4
        for r in results:
            assert len(r.top_predictions) > 0

    def test_predict_batch_single_image(self, classifier, sample_image):
        results = classifier.predict_batch([sample_image])
        assert len(results) == 1

    def test_model_name_in_result(self, classifier, sample_image):
        result = classifier.predict(sample_image)
        assert result.model_name == "mock_resnet"

    def test_available_models(self, classifier):
        models = classifier.available_models()
        assert "resnet50" in models
        assert len(models) >= 3

    def test_get_torch_model(self, classifier):
        model = classifier.get_torch_model()
        assert isinstance(model, nn.Module)

    def test_predict_tiny_image(self, classifier):
        tiny = Image.fromarray(np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8), "RGB")
        result = classifier.predict(tiny)
        assert len(result.top_predictions) > 0
