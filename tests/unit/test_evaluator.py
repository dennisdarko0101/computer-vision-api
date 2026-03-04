"""Tests for CVEvaluator — 14 tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from PIL import Image

from src.evaluation.evaluator import (
    CVEvaluator,
    ClassificationMetrics,
    ComparisonReport,
    DetectionMetrics,
    LatencyReport,
)
from src.models.detector import BBox


# ── Mock model helpers ──────────────────────────────────────────────────────

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


@dataclass
class MockDetectionResult:
    detections: list[BBox] = field(default_factory=list)
    count: int = 0
    latency_ms: float = 1.0
    model_name: str = "mock"
    image_size: tuple[int, int] = (64, 64)


class MockClassifierForEval:
    """Mock classifier that returns predictable results."""

    def __init__(self, correct_class: int = 0):
        self.correct_class = correct_class

    def predict(self, image: Image.Image) -> MockClassificationResult:
        return MockClassificationResult(
            top_predictions=[
                MockPrediction(label=f"class_{self.correct_class}", confidence=0.8, class_index=self.correct_class),
                MockPrediction(label=f"class_{(self.correct_class + 1) % 5}", confidence=0.1, class_index=(self.correct_class + 1) % 5),
                MockPrediction(label=f"class_{(self.correct_class + 2) % 5}", confidence=0.05, class_index=(self.correct_class + 2) % 5),
                MockPrediction(label=f"class_{(self.correct_class + 3) % 5}", confidence=0.03, class_index=(self.correct_class + 3) % 5),
                MockPrediction(label=f"class_{(self.correct_class + 4) % 5}", confidence=0.02, class_index=(self.correct_class + 4) % 5),
            ]
        )


class MockDetectorForEval:
    """Mock detector returning fixed detections."""

    def __init__(self, detections: list[BBox] | None = None):
        self._detections = detections or []

    def detect(self, image: Image.Image) -> MockDetectionResult:
        return MockDetectionResult(
            detections=self._detections,
            count=len(self._detections),
        )


@pytest.fixture
def evaluator() -> CVEvaluator:
    return CVEvaluator()


@pytest.fixture
def classification_test_data(sample_image) -> list[tuple[Image.Image, int]]:
    """5 images, all class 0."""
    return [(sample_image, 0) for _ in range(5)]


@pytest.fixture
def detection_test_data(sample_image) -> list[tuple[Image.Image, list[BBox]]]:
    gt = [
        BBox(x1=10, y1=10, x2=50, y2=50, label="cat", confidence=1.0, class_id=0),
        BBox(x1=100, y1=100, x2=150, y2=150, label="dog", confidence=1.0, class_id=1),
    ]
    return [(sample_image, gt) for _ in range(3)]


class TestClassifierEvaluation:
    def test_evaluate_returns_metrics(self, evaluator, classification_test_data):
        model = MockClassifierForEval(correct_class=0)
        metrics = evaluator.evaluate_classifier(model, classification_test_data)
        assert isinstance(metrics, ClassificationMetrics)

    def test_accuracy_perfect(self, evaluator, classification_test_data):
        model = MockClassifierForEval(correct_class=0)
        metrics = evaluator.evaluate_classifier(model, classification_test_data)
        assert metrics.accuracy == 1.0

    def test_accuracy_wrong(self, evaluator, classification_test_data):
        model = MockClassifierForEval(correct_class=3)  # wrong class
        metrics = evaluator.evaluate_classifier(model, classification_test_data)
        assert metrics.accuracy == 0.0

    def test_top5_accuracy(self, evaluator, classification_test_data):
        model = MockClassifierForEval(correct_class=0)
        metrics = evaluator.evaluate_classifier(model, classification_test_data)
        assert metrics.top5_accuracy == 1.0

    def test_num_samples(self, evaluator, classification_test_data):
        model = MockClassifierForEval(correct_class=0)
        metrics = evaluator.evaluate_classifier(model, classification_test_data)
        assert metrics.num_samples == 5

    def test_per_class_metrics(self, evaluator, classification_test_data):
        model = MockClassifierForEval(correct_class=0)
        metrics = evaluator.evaluate_classifier(model, classification_test_data)
        assert len(metrics.per_class_precision) > 0

    def test_confusion_matrix(self, evaluator, classification_test_data):
        model = MockClassifierForEval(correct_class=0)
        metrics = evaluator.evaluate_classifier(model, classification_test_data)
        assert metrics.confusion_matrix is not None

    def test_empty_dataset(self, evaluator):
        model = MockClassifierForEval()
        metrics = evaluator.evaluate_classifier(model, [])
        assert metrics.num_samples == 0


class TestDetectorEvaluation:
    def test_evaluate_returns_metrics(self, evaluator, detection_test_data):
        detections = [
            BBox(x1=10, y1=10, x2=50, y2=50, label="cat", confidence=0.9, class_id=0),
        ]
        model = MockDetectorForEval(detections)
        metrics = evaluator.evaluate_detector(model, detection_test_data)
        assert isinstance(metrics, DetectionMetrics)

    def test_num_images(self, evaluator, detection_test_data):
        model = MockDetectorForEval([])
        metrics = evaluator.evaluate_detector(model, detection_test_data)
        assert metrics.num_images == 3

    def test_map_range(self, evaluator, detection_test_data):
        detections = [
            BBox(x1=10, y1=10, x2=50, y2=50, label="cat", confidence=0.9),
        ]
        model = MockDetectorForEval(detections)
        metrics = evaluator.evaluate_detector(model, detection_test_data)
        assert 0.0 <= metrics.mAP <= 1.0

    def test_empty_dataset(self, evaluator):
        model = MockDetectorForEval()
        metrics = evaluator.evaluate_detector(model, [])
        assert metrics.num_images == 0


class TestLatencyBenchmark:
    def test_benchmark_returns_report(self, evaluator, sample_image):
        model = MockClassifierForEval()
        report = evaluator.benchmark_latency(model, sample_image, n_runs=10, warmup=2)
        assert isinstance(report, LatencyReport)
        assert report.n_runs == 10
        assert report.mean_ms > 0

    def test_benchmark_percentiles(self, evaluator, sample_image):
        model = MockClassifierForEval()
        report = evaluator.benchmark_latency(model, sample_image, n_runs=20, warmup=2)
        assert report.p50_ms <= report.p95_ms <= report.p99_ms


class TestModelComparison:
    def test_compare_models(self, evaluator, classification_test_data, sample_image):
        models = {
            "good": MockClassifierForEval(correct_class=0),
            "bad": MockClassifierForEval(correct_class=3),
        }
        report = evaluator.compare_models(models, classification_test_data, sample_image)
        assert isinstance(report, ComparisonReport)
        assert report.best_accuracy == "good"
