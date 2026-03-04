"""Tests for PostProcessor — 14 tests."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image

from src.models.detector import BBox
from src.processing.postprocessor import PostProcessor, Prediction


@pytest.fixture
def postprocessor() -> PostProcessor:
    labels = {0: "cat", 1: "dog", 2: "car", 3: "person", 4: "bird"}
    return PostProcessor(labels=labels)


@pytest.fixture
def sample_logits() -> torch.Tensor:
    """5-class logits."""
    return torch.tensor([2.0, 0.5, -1.0, 3.0, 0.1])


@pytest.fixture
def sample_detection_arrays():
    """Raw detection output arrays."""
    boxes = np.array([
        [10, 10, 50, 50],
        [12, 12, 52, 52],
        [100, 100, 200, 200],
    ], dtype=np.float32)
    scores = np.array([0.9, 0.85, 0.7], dtype=np.float32)
    class_ids = np.array([0, 0, 2], dtype=np.int32)
    return boxes, scores, class_ids


class TestDecodeClassifications:
    def test_returns_predictions(self, postprocessor, sample_logits):
        preds = postprocessor.decode_classifications(sample_logits, top_k=3)
        assert len(preds) == 3

    def test_sorted_by_confidence(self, postprocessor, sample_logits):
        preds = postprocessor.decode_classifications(sample_logits, top_k=5)
        confs = [p.confidence for p in preds]
        assert confs == sorted(confs, reverse=True)

    def test_prediction_has_label(self, postprocessor, sample_logits):
        preds = postprocessor.decode_classifications(sample_logits, top_k=1)
        assert preds[0].label == "person"  # index 3 has highest logit

    def test_handles_numpy_input(self, postprocessor):
        logits = np.array([1.0, 2.0, 0.5, -1.0, 3.0])
        preds = postprocessor.decode_classifications(logits, top_k=2)
        assert len(preds) == 2

    def test_handles_batch_logits(self, postprocessor):
        logits = torch.tensor([[2.0, 0.5, -1.0, 3.0, 0.1]])
        preds = postprocessor.decode_classifications(logits, top_k=2)
        assert len(preds) == 2

    def test_confidence_sums_to_one(self, postprocessor, sample_logits):
        preds = postprocessor.decode_classifications(sample_logits, top_k=5)
        total = sum(p.confidence for p in preds)
        assert total == pytest.approx(1.0, abs=0.01)


class TestDecodeDetections:
    def test_returns_boxes(self, postprocessor, sample_detection_arrays):
        boxes, scores, class_ids = sample_detection_arrays
        result = postprocessor.decode_detections(boxes, scores, class_ids, conf_threshold=0.5)
        assert len(result) > 0

    def test_confidence_filtering(self, postprocessor, sample_detection_arrays):
        boxes, scores, class_ids = sample_detection_arrays
        result = postprocessor.decode_detections(boxes, scores, class_ids, conf_threshold=0.95)
        # Only the 0.9 box might pass, but NMS could remove it
        assert all(d.confidence >= 0.95 for d in result) or len(result) == 0

    def test_nms_applied(self, postprocessor, sample_detection_arrays):
        boxes, scores, class_ids = sample_detection_arrays
        result = postprocessor.decode_detections(
            boxes, scores, class_ids, conf_threshold=0.5, nms_threshold=0.3
        )
        # Two overlapping cat boxes should be reduced
        cat_boxes = [d for d in result if d.label == "cat"]
        assert len(cat_boxes) <= 1

    def test_labels_mapped(self, postprocessor, sample_detection_arrays):
        boxes, scores, class_ids = sample_detection_arrays
        result = postprocessor.decode_detections(boxes, scores, class_ids, conf_threshold=0.5)
        labels = {d.label for d in result}
        assert labels.issubset({"cat", "car"})


class TestDrawBoxes:
    def test_draw_returns_image(self, postprocessor, sample_image):
        boxes = [BBox(x1=5, y1=5, x2=30, y2=30, label="cat", confidence=0.9)]
        result = postprocessor.draw_boxes(sample_image, boxes)
        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size

    def test_draw_empty_boxes(self, postprocessor, sample_image):
        result = postprocessor.draw_boxes(sample_image, [])
        assert isinstance(result, Image.Image)


class TestFormatResults:
    def test_format_json(self, postprocessor):
        data = [{"label": "cat", "confidence": 0.9}]
        result = postprocessor.format_results(data, "json")
        parsed = json.loads(result)
        assert parsed[0]["label"] == "cat"

    def test_format_csv(self, postprocessor):
        data = [{"label": "cat", "confidence": 0.9}]
        result = postprocessor.format_results(data, "csv")
        assert "label" in result
        assert "cat" in result

    def test_format_invalid(self, postprocessor):
        with pytest.raises(ValueError, match="Unsupported format"):
            postprocessor.format_results([], "xml")


class TestSoftmax:
    def test_softmax_sums_to_one(self):
        logits = np.array([1.0, 2.0, 3.0])
        probs = PostProcessor.softmax(logits)
        assert probs.sum() == pytest.approx(1.0, abs=1e-5)

    def test_softmax_all_positive(self):
        logits = np.array([-1.0, 0.0, 1.0])
        probs = PostProcessor.softmax(logits)
        assert (probs > 0).all()
