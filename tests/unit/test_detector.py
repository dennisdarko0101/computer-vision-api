"""Tests for ObjectDetector — 17 tests using mock detector."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.models.detector import (
    BBox,
    DetectionResult,
    ObjectDetector,
    compute_iou,
    non_max_suppression,
)


@pytest.fixture
def detector() -> ObjectDetector:
    """Mock detector — no weight downloads."""
    return ObjectDetector(model_name="mock", confidence_threshold=0.5)


@pytest.fixture
def sample_boxes() -> list[BBox]:
    """Overlapping boxes for NMS tests."""
    return [
        BBox(x1=10, y1=10, x2=50, y2=50, label="cat", confidence=0.9, class_id=17),
        BBox(x1=12, y1=12, x2=52, y2=52, label="cat", confidence=0.8, class_id=17),
        BBox(x1=100, y1=100, x2=150, y2=150, label="dog", confidence=0.85, class_id=18),
        BBox(x1=200, y1=200, x2=250, y2=250, label="car", confidence=0.6, class_id=3),
    ]


class TestBBox:
    def test_width(self):
        box = BBox(x1=10, y1=20, x2=50, y2=60, label="cat", confidence=0.9)
        assert box.width == 40

    def test_height(self):
        box = BBox(x1=10, y1=20, x2=50, y2=60, label="cat", confidence=0.9)
        assert box.height == 40

    def test_area(self):
        box = BBox(x1=0, y1=0, x2=10, y2=10, label="cat", confidence=0.9)
        assert box.area == 100

    def test_to_dict(self):
        box = BBox(x1=10.123, y1=20.456, x2=50.789, y2=60.012, label="cat", confidence=0.9123)
        d = box.to_dict()
        assert "x1" in d and "label" in d
        assert d["confidence"] == 0.9123


class TestComputeIoU:
    def test_identical_boxes(self):
        box = BBox(x1=0, y1=0, x2=10, y2=10, label="a", confidence=0.9)
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = BBox(x1=0, y1=0, x2=10, y2=10, label="a", confidence=0.9)
        b = BBox(x1=20, y1=20, x2=30, y2=30, label="b", confidence=0.9)
        assert compute_iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = BBox(x1=0, y1=0, x2=10, y2=10, label="a", confidence=0.9)
        b = BBox(x1=5, y1=5, x2=15, y2=15, label="b", confidence=0.9)
        iou = compute_iou(a, b)
        assert 0.0 < iou < 1.0
        # Intersection = 5*5 = 25, Union = 100 + 100 - 25 = 175
        assert iou == pytest.approx(25 / 175, rel=1e-3)

    def test_zero_area_box(self):
        a = BBox(x1=0, y1=0, x2=0, y2=0, label="a", confidence=0.9)
        b = BBox(x1=0, y1=0, x2=10, y2=10, label="b", confidence=0.9)
        assert compute_iou(a, b) == 0.0


class TestNMS:
    def test_nms_removes_overlapping(self, sample_boxes):
        kept = non_max_suppression(sample_boxes, iou_threshold=0.3)
        # The two overlapping cat boxes should be reduced to one
        cat_boxes = [b for b in kept if b.label == "cat"]
        assert len(cat_boxes) == 1
        assert cat_boxes[0].confidence == 0.9  # highest confidence kept

    def test_nms_keeps_non_overlapping(self, sample_boxes):
        kept = non_max_suppression(sample_boxes, iou_threshold=0.3)
        labels = {b.label for b in kept}
        assert "dog" in labels
        assert "car" in labels

    def test_nms_empty_input(self):
        assert non_max_suppression([]) == []

    def test_nms_single_box(self):
        box = BBox(x1=0, y1=0, x2=10, y2=10, label="a", confidence=0.9)
        result = non_max_suppression([box])
        assert len(result) == 1


class TestObjectDetector:
    def test_detect_returns_result(self, detector, sample_image):
        result = detector.detect(sample_image)
        assert isinstance(result, DetectionResult)
        assert result.count >= 0

    def test_detect_has_detections(self, detector, sample_image):
        result = detector.detect(sample_image)
        assert isinstance(result.detections, list)

    def test_detect_bbox_format(self, detector, sample_image):
        result = detector.detect(sample_image)
        for det in result.detections:
            assert det.x1 < det.x2
            assert det.y1 < det.y2
            assert 0.0 <= det.confidence <= 1.0
            assert isinstance(det.label, str)

    def test_detect_latency_recorded(self, detector, sample_image):
        result = detector.detect(sample_image)
        assert result.latency_ms > 0

    def test_detect_model_name(self, detector, sample_image):
        result = detector.detect(sample_image)
        assert result.model_name == "mock"

    def test_detect_image_size(self, detector, sample_image):
        result = detector.detect(sample_image)
        assert result.image_size == sample_image.size

    def test_detect_batch(self, detector, sample_images_batch):
        results = detector.detect_batch(sample_images_batch)
        assert len(results) == 4
        for r in results:
            assert isinstance(r, DetectionResult)

    def test_detect_grayscale(self, detector, sample_grayscale):
        result = detector.detect(sample_grayscale)
        assert isinstance(result, DetectionResult)

    def test_confidence_threshold_applied(self, detector, sample_image):
        result = detector.detect(sample_image)
        for det in result.detections:
            assert det.confidence >= detector.confidence_threshold
