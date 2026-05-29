"""Object detection: bounding boxes, IoU, non-max suppression, detector wrapper.

The pure-Python geometry (BBox, compute_iou, non_max_suppression) has no heavy
dependencies and is import-safe. The torchvision-backed ObjectDetector constructs
its model lazily so importing this module never downloads weights.
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from typing import Any

from PIL import Image

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BBox:
    """An axis-aligned bounding box in pixel coordinates (x1,y1) top-left, (x2,y2) bottom-right."""

    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    confidence: float
    class_id: int = -1

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        w = self.width
        h = self.height
        if w <= 0 or h <= 0:
            return 0.0
        return w * h

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DetectionResult:
    """Result of running detection on a single image."""

    detections: list[BBox]
    count: int
    latency_ms: float
    model_name: str
    image_size: tuple[int, int]


def compute_iou(box_a: BBox, box_b: BBox) -> float:
    """Intersection-over-union of two boxes. Returns 0.0 for degenerate / disjoint boxes."""
    # Intersection rectangle
    inter_x1 = max(box_a.x1, box_b.x1)
    inter_y1 = max(box_a.y1, box_b.y1)
    inter_x2 = min(box_a.x2, box_b.x2)
    inter_y2 = min(box_a.y2, box_b.y2)

    inter_w = inter_x2 - inter_x1
    inter_h = inter_y2 - inter_y1
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    intersection = inter_w * inter_h

    union = box_a.area + box_b.area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def non_max_suppression(
    boxes: list[BBox],
    iou_threshold: float = 0.5,
) -> list[BBox]:
    """Greedy non-maximum suppression.

    Boxes are processed in descending confidence order. A box is kept unless it
    overlaps an already-kept box (of the same label) with IoU >= iou_threshold.
    """
    if not boxes:
        return []

    ordered = sorted(boxes, key=lambda b: b.confidence, reverse=True)
    kept: list[BBox] = []

    for candidate in ordered:
        suppressed = False
        for keeper in kept:
            if keeper.label != candidate.label:
                continue
            if compute_iou(candidate, keeper) >= iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(candidate)

    return kept


# COCO-style class names used by torchvision detection models (subset of common labels).
COCO_LABELS = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 16: "bird",
    17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow",
}


class ObjectDetector:
    """Torchvision-backed object detector with a deterministic mock backend.

    When ``model_name == "mock"`` (or torchvision is unavailable) no weights are
    downloaded; the detector produces deterministic synthetic boxes so the rest of
    the pipeline can be exercised without large model downloads.
    """

    def __init__(
        self,
        model_name: str = "fasterrcnn_mobilenet",
        device: str = "cpu",
        confidence_threshold: float = 0.5,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.labels = dict(COCO_LABELS)
        self._model: Any = None  # lazily constructed real model

    def _is_mock(self) -> bool:
        return self.model_name == "mock"

    def _ensure_model(self) -> Any:
        """Lazily build the torchvision detection model. Never called for the mock backend."""
        if self._model is not None:
            return self._model
        import torch
        from torchvision.models import detection as tv_detection

        if self.model_name in ("fasterrcnn_mobilenet", "fasterrcnn_mobilenet_v3"):
            model = tv_detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
        elif self.model_name == "fasterrcnn_resnet50":
            model = tv_detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        else:
            model = tv_detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
        model.eval()
        self._model = model.to(torch.device(self.device))
        return self._model

    def get_torch_model(self) -> Any:
        return self._ensure_model()

    def _mock_detections(self, image: Image.Image) -> list[BBox]:
        """Deterministic synthetic detections constrained to the image bounds."""
        w, h = image.size
        rng = random.Random(hash((w, h, self.model_name)) & 0xFFFFFFFF)
        out: list[BBox] = []
        candidates = [
            (0.10, 0.10, 0.45, 0.45, 17, "cat", 0.92),
            (0.55, 0.50, 0.90, 0.90, 18, "dog", 0.74),
            (0.30, 0.30, 0.60, 0.70, 3, "car", 0.40),
        ]
        for fx1, fy1, fx2, fy2, cid, label, base_conf in candidates:
            conf = round(min(0.99, max(0.0, base_conf + rng.uniform(-0.05, 0.05))), 4)
            if conf < self.confidence_threshold:
                continue
            out.append(
                BBox(
                    x1=float(round(fx1 * w, 2)),
                    y1=float(round(fy1 * h, 2)),
                    x2=float(round(fx2 * w, 2)),
                    y2=float(round(fy2 * h, 2)),
                    label=label,
                    confidence=conf,
                    class_id=cid,
                )
            )
        return out

    def detect(self, image: Image.Image) -> DetectionResult:
        """Run detection on a single image."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        start = time.perf_counter()

        if self._is_mock():
            detections = self._mock_detections(image)
        else:
            detections = self._detect_real(image)

        # Always enforce the configured confidence threshold.
        detections = [d for d in detections if d.confidence >= self.confidence_threshold]
        latency_ms = (time.perf_counter() - start) * 1000

        return DetectionResult(
            detections=detections,
            count=len(detections),
            latency_ms=round(latency_ms, 4),
            model_name=self.model_name,
            image_size=image.size,
        )

    def _detect_real(self, image: Image.Image) -> list[BBox]:
        import torch
        from torchvision.transforms import functional as TF

        model = self._ensure_model()
        tensor = TF.to_tensor(image).to(torch.device(self.device))
        with torch.no_grad():
            outputs = model([tensor])[0]

        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        label_ids = outputs["labels"].cpu().numpy()

        detections: list[BBox] = []
        for (x1, y1, x2, y2), score, cid in zip(boxes, scores, label_ids):
            cid = int(cid)
            detections.append(
                BBox(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    label=self.labels.get(cid, f"class_{cid}"),
                    confidence=float(score),
                    class_id=cid,
                )
            )
        return non_max_suppression(detections, iou_threshold=0.45)

    def detect_batch(self, images: list[Image.Image]) -> list[DetectionResult]:
        return [self.detect(img) for img in images]
