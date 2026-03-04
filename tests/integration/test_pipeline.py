"""Integration tests for end-to-end pipeline flows — 10 tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.processing.preprocessor import ImagePreprocessor
from src.processing.postprocessor import PostProcessor
from src.processing.batch_processor import BatchProcessor
from src.models.detector import BBox, non_max_suppression
from tests.conftest import MockClassifierModel, MockEmbedderModel


# ── Helpers ──────────────────────────────────────────────────────────────────

@dataclass
class MockPrediction:
    label: str
    confidence: float
    class_index: int


@dataclass
class MockClassificationResult:
    top_predictions: list[MockPrediction] = field(default_factory=list)
    latency_ms: float = 1.0
    model_name: str = "mock"


class IntegrationClassifier:
    """Classifier using mock model for pipeline integration tests."""

    def __init__(self):
        self.model = MockClassifierModel(num_classes=10)
        self.model.eval()
        self.preprocessor = ImagePreprocessor(target_size=(224, 224))
        self.labels = {i: f"class_{i}" for i in range(10)}

    def predict(self, image: Image.Image) -> MockClassificationResult:
        tensor = self.preprocessor.full_pipeline(image).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
        top_probs, top_idx = torch.topk(probs[0], k=5)
        preds = [
            MockPrediction(
                label=self.labels[idx.item()],
                confidence=round(prob.item(), 4),
                class_index=idx.item(),
            )
            for prob, idx in zip(top_probs, top_idx)
        ]
        return MockClassificationResult(top_predictions=preds)


class IntegrationEmbedder:
    """Embedder using mock model."""

    def __init__(self):
        self.model = MockEmbedderModel(embedding_dim=128)
        self.model.eval()
        self.preprocessor = ImagePreprocessor(target_size=(224, 224))

    def embed(self, image: Image.Image) -> np.ndarray:
        tensor = self.preprocessor.full_pipeline(image).unsqueeze(0)
        with torch.no_grad():
            out = self.model(tensor)
        vec = out.numpy().flatten()
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


@pytest.fixture
def classifier() -> IntegrationClassifier:
    return IntegrationClassifier()


@pytest.fixture
def embedder() -> IntegrationEmbedder:
    return IntegrationEmbedder()


@pytest.fixture
def postprocessor() -> PostProcessor:
    return PostProcessor(labels={i: f"class_{i}" for i in range(10)})


@pytest.fixture
def batch_processor() -> BatchProcessor:
    return BatchProcessor(max_workers=2)


# ── Tests ────────────────────────────────────────────────────────────────────

class TestUploadProcessReturn:
    """Test: upload bytes → preprocess → classify → return results."""

    def test_bytes_to_classification(self, classifier):
        raw = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        image = Image.fromarray(raw, "RGB")
        result = classifier.predict(image)
        assert len(result.top_predictions) == 5
        assert all(p.confidence > 0 for p in result.top_predictions)

    def test_grayscale_to_classification(self, classifier):
        gray = Image.fromarray(np.random.randint(0, 255, (80, 80), dtype=np.uint8), "L")
        result = classifier.predict(gray)
        assert len(result.top_predictions) > 0

    def test_large_image_pipeline(self, classifier):
        large = Image.fromarray(np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8), "RGB")
        result = classifier.predict(large)
        assert len(result.top_predictions) > 0


class TestPreprocessToPostprocess:
    """Test: preprocess → model → postprocess pipeline."""

    def test_classification_decode(self, postprocessor):
        logits = torch.randn(10)
        preds = postprocessor.decode_classifications(logits, top_k=3)
        assert len(preds) == 3
        confs = [p.confidence for p in preds]
        assert confs == sorted(confs, reverse=True)

    def test_detection_decode_and_nms(self, postprocessor):
        boxes = np.array([[10, 10, 50, 50], [12, 12, 52, 52], [100, 100, 200, 200]])
        scores = np.array([0.9, 0.85, 0.7])
        class_ids = np.array([0, 0, 1])
        detections = postprocessor.decode_detections(boxes, scores, class_ids, conf_threshold=0.5)
        # NMS should merge overlapping class_0 boxes
        class0 = [d for d in detections if d.class_id == 0]
        assert len(class0) <= 1


class TestBatchPipeline:
    """Test: batch directory → classify each → aggregate."""

    def test_batch_directory_pipeline(self, classifier, batch_processor, image_dir):
        result = batch_processor.process_directory(
            image_dir,
            lambda img: classifier.predict(img),
        )
        assert result.total == 5
        assert result.successful == 5
        for item in result.items:
            assert item.result is not None
            assert len(item.result.top_predictions) > 0


class TestEmbeddingPipeline:
    """Test: image → embed → check vector properties."""

    def test_embedding_pipeline(self, embedder):
        image = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8), "RGB")
        vec = embedder.embed(image)
        assert vec.shape == (128,)
        assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-4)

    def test_batch_embeddings_distinct(self, embedder):
        img1 = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), "RGB")
        img2 = Image.fromarray(np.full((64, 64, 3), 255, dtype=np.uint8), "RGB")
        v1 = embedder.embed(img1)
        v2 = embedder.embed(img2)
        cosine_sim = np.dot(v1, v2)
        assert cosine_sim < 0.999  # different images should have different embeddings


class TestValidationPipeline:
    """Test: validate → reject or process."""

    def test_valid_image_proceeds(self, classifier):
        preprocessor = ImagePreprocessor()
        raw = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        image = Image.fromarray(raw, "RGB")
        import io
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        validation = preprocessor.validate_image(raw_bytes=buf.getvalue())
        assert validation.valid
        result = classifier.predict(image)
        assert len(result.top_predictions) > 0

    def test_invalid_image_rejected(self):
        preprocessor = ImagePreprocessor()
        validation = preprocessor.validate_image(raw_bytes=b"garbage")
        assert not validation.valid
