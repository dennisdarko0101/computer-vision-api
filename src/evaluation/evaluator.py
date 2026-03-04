"""Model evaluation: classification metrics, detection mAP, latency benchmarks."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from src.models.detector import BBox, compute_iou
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ClassificationMetrics:
    """Aggregate classification evaluation metrics."""
    accuracy: float = 0.0
    top5_accuracy: float = 0.0
    per_class_precision: dict[str, float] = field(default_factory=dict)
    per_class_recall: dict[str, float] = field(default_factory=dict)
    per_class_f1: dict[str, float] = field(default_factory=dict)
    confusion_matrix: np.ndarray | None = None
    num_samples: int = 0


@dataclass
class DetectionMetrics:
    """Aggregate detection evaluation metrics."""
    mAP: float = 0.0
    mAP_50: float = 0.0
    mAP_75: float = 0.0
    per_class_AP: dict[str, float] = field(default_factory=dict)
    num_images: int = 0
    num_ground_truth: int = 0
    num_predictions: int = 0


@dataclass
class LatencyReport:
    """Inference latency statistics."""
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    std_ms: float = 0.0
    n_runs: int = 0
    throughput_fps: float = 0.0


@dataclass
class ComparisonReport:
    """Comparison of multiple models."""
    model_metrics: dict[str, ClassificationMetrics | DetectionMetrics] = field(default_factory=dict)
    model_latencies: dict[str, LatencyReport] = field(default_factory=dict)
    best_accuracy: str = ""
    best_latency: str = ""


class CVEvaluator:
    """Evaluate computer vision models on classification and detection tasks."""

    def evaluate_classifier(
        self,
        model: object,
        test_data: list[tuple[Image.Image, int]],
        label_map: dict[int, str] | None = None,
    ) -> ClassificationMetrics:
        """Evaluate a classifier on labeled test data.

        Args:
            model: Object with predict(image) method returning ClassificationResult.
            test_data: List of (image, ground_truth_class_index) tuples.
            label_map: Optional mapping from class index to label string.
        """
        if not test_data:
            return ClassificationMetrics()

        predictions: list[int] = []
        top5_correct = 0
        true_labels: list[int] = []

        for image, true_class in test_data:
            result = model.predict(image)  # type: ignore[union-attr]
            pred_class = result.top_predictions[0].class_index if result.top_predictions else -1
            predictions.append(pred_class)
            true_labels.append(true_class)

            top5_indices = [p.class_index for p in result.top_predictions[:5]]
            if true_class in top5_indices:
                top5_correct += 1

        predictions_arr = np.array(predictions)
        true_arr = np.array(true_labels)

        accuracy = float(np.mean(predictions_arr == true_arr))
        top5_accuracy = top5_correct / len(test_data)

        # Per-class metrics
        classes = sorted(set(true_labels) | set(predictions))
        n_classes = max(classes) + 1 if classes else 0
        confusion = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(true_labels, predictions):
            if 0 <= t < n_classes and 0 <= p < n_classes:
                confusion[t][p] += 1

        per_class_precision: dict[str, float] = {}
        per_class_recall: dict[str, float] = {}
        per_class_f1: dict[str, float] = {}

        for c in classes:
            if c < 0 or c >= n_classes:
                continue
            label = label_map.get(c, f"class_{c}") if label_map else f"class_{c}"
            tp = confusion[c][c]
            fp = confusion[:, c].sum() - tp
            fn = confusion[c, :].sum() - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            per_class_precision[label] = round(precision, 4)
            per_class_recall[label] = round(recall, 4)
            per_class_f1[label] = round(f1, 4)

        return ClassificationMetrics(
            accuracy=round(accuracy, 4),
            top5_accuracy=round(top5_accuracy, 4),
            per_class_precision=per_class_precision,
            per_class_recall=per_class_recall,
            per_class_f1=per_class_f1,
            confusion_matrix=confusion,
            num_samples=len(test_data),
        )

    def evaluate_detector(
        self,
        model: object,
        test_data: list[tuple[Image.Image, list[BBox]]],
        iou_thresholds: list[float] | None = None,
    ) -> DetectionMetrics:
        """Evaluate detector using mean Average Precision.

        Args:
            model: Object with detect(image) method returning DetectionResult.
            test_data: List of (image, ground_truth_boxes) tuples.
            iou_thresholds: IoU thresholds for AP calculation.
        """
        if not test_data:
            return DetectionMetrics()

        thresholds = iou_thresholds or [0.5 + 0.05 * i for i in range(10)]
        all_predictions: dict[str, list[dict]] = defaultdict(list)
        all_ground_truths: dict[str, int] = defaultdict(int)
        total_gt = 0
        total_pred = 0

        for img_idx, (image, gt_boxes) in enumerate(test_data):
            result = model.detect(image)  # type: ignore[union-attr]
            total_pred += len(result.detections)
            total_gt += len(gt_boxes)

            for gt in gt_boxes:
                all_ground_truths[gt.label] += 1

            gt_matched = [False] * len(gt_boxes)

            sorted_dets = sorted(result.detections, key=lambda d: d.confidence, reverse=True)

            for det in sorted_dets:
                best_iou = 0.0
                best_gt_idx = -1
                for gt_idx, gt in enumerate(gt_boxes):
                    if gt.label != det.label or gt_matched[gt_idx]:
                        continue
                    iou = compute_iou(det, gt)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                all_predictions[det.label].append({
                    "confidence": det.confidence,
                    "iou": best_iou,
                    "matched": best_gt_idx >= 0,
                })
                if best_gt_idx >= 0:
                    gt_matched[best_gt_idx] = True

        # Compute AP per class at each threshold
        per_class_aps: dict[str, list[float]] = defaultdict(list)

        for threshold in thresholds:
            for class_label in all_ground_truths:
                preds = sorted(
                    all_predictions.get(class_label, []),
                    key=lambda x: x["confidence"],
                    reverse=True,
                )
                n_gt = all_ground_truths[class_label]
                if n_gt == 0:
                    continue

                tp = 0
                fp = 0
                precisions: list[float] = []
                recalls: list[float] = []

                for pred in preds:
                    if pred["matched"] and pred["iou"] >= threshold:
                        tp += 1
                    else:
                        fp += 1
                    precisions.append(tp / (tp + fp))
                    recalls.append(tp / n_gt)

                # Compute AP using 11-point interpolation
                ap = self._compute_ap(precisions, recalls)
                per_class_aps[class_label].append(ap)

        # Aggregate
        per_class_AP_final: dict[str, float] = {}
        for class_label, aps in per_class_aps.items():
            per_class_AP_final[class_label] = round(float(np.mean(aps)), 4) if aps else 0.0

        mAP = float(np.mean(list(per_class_AP_final.values()))) if per_class_AP_final else 0.0

        # mAP@50
        ap50_values: list[float] = []
        for class_label in all_ground_truths:
            aps = per_class_aps.get(class_label, [])
            if aps:
                ap50_values.append(aps[0])
        mAP_50 = float(np.mean(ap50_values)) if ap50_values else 0.0

        # mAP@75
        ap75_values: list[float] = []
        for class_label in all_ground_truths:
            aps = per_class_aps.get(class_label, [])
            if len(aps) > 5:
                ap75_values.append(aps[5])
        mAP_75 = float(np.mean(ap75_values)) if ap75_values else 0.0

        return DetectionMetrics(
            mAP=round(mAP, 4),
            mAP_50=round(mAP_50, 4),
            mAP_75=round(mAP_75, 4),
            per_class_AP=per_class_AP_final,
            num_images=len(test_data),
            num_ground_truth=total_gt,
            num_predictions=total_pred,
        )

    def _compute_ap(self, precisions: list[float], recalls: list[float]) -> float:
        """Compute Average Precision using 11-point interpolation."""
        if not precisions or not recalls:
            return 0.0

        ap = 0.0
        for t in np.linspace(0, 1, 11):
            precisions_above = [p for p, r in zip(precisions, recalls) if r >= t]
            if precisions_above:
                ap += max(precisions_above)
        return ap / 11.0

    def benchmark_latency(
        self,
        model: object,
        sample_image: Image.Image,
        n_runs: int = 100,
        warmup: int = 10,
        task: str = "classify",
    ) -> LatencyReport:
        """Benchmark model inference latency."""
        infer_fn = getattr(model, "predict", None) or getattr(model, "detect", None)
        if task == "classify":
            infer_fn = getattr(model, "predict")
        elif task == "detect":
            infer_fn = getattr(model, "detect")
        elif task == "embed":
            infer_fn = getattr(model, "embed")

        # Warmup
        for _ in range(warmup):
            infer_fn(sample_image)

        # Benchmark
        times: list[float] = []
        for _ in range(n_runs):
            start = time.perf_counter()
            infer_fn(sample_image)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        arr = np.array(times)
        mean_ms = float(np.mean(arr))

        return LatencyReport(
            mean_ms=round(mean_ms, 3),
            median_ms=round(float(np.median(arr)), 3),
            p50_ms=round(float(np.percentile(arr, 50)), 3),
            p95_ms=round(float(np.percentile(arr, 95)), 3),
            p99_ms=round(float(np.percentile(arr, 99)), 3),
            min_ms=round(float(np.min(arr)), 3),
            max_ms=round(float(np.max(arr)), 3),
            std_ms=round(float(np.std(arr)), 3),
            n_runs=n_runs,
            throughput_fps=round(1000.0 / mean_ms, 1) if mean_ms > 0 else 0.0,
        )

    def compare_models(
        self,
        models: dict[str, object],
        test_data: list[tuple[Image.Image, int]],
        sample_image: Image.Image | None = None,
        label_map: dict[int, str] | None = None,
    ) -> ComparisonReport:
        """Compare multiple models on accuracy and latency."""
        report = ComparisonReport()

        for name, model in models.items():
            metrics = self.evaluate_classifier(model, test_data, label_map)
            report.model_metrics[name] = metrics

            if sample_image is not None:
                latency = self.benchmark_latency(model, sample_image, n_runs=50)
                report.model_latencies[name] = latency

        # Find bests
        if report.model_metrics:
            report.best_accuracy = max(
                report.model_metrics,
                key=lambda k: report.model_metrics[k].accuracy
                if isinstance(report.model_metrics[k], ClassificationMetrics)
                else 0.0,
            )
        if report.model_latencies:
            report.best_latency = min(
                report.model_latencies,
                key=lambda k: report.model_latencies[k].mean_ms,
            )

        return report
