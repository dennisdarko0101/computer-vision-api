"""Post-processing for model outputs: decoding, NMS, visualization."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from src.models.detector import BBox, non_max_suppression
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Prediction:
    """Decoded classification prediction."""
    label: str
    confidence: float
    class_index: int


# Color palette for drawing boxes
BOX_COLORS = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
    "#FF8000", "#8000FF", "#0080FF", "#FF0080", "#80FF00", "#00FF80",
    "#FF4040", "#40FF40", "#4040FF", "#FFFF40", "#FF40FF", "#40FFFF",
]


class PostProcessor:
    """Decode and format model outputs."""

    def __init__(self, labels: dict[int, str] | None = None) -> None:
        self.labels = labels or {}

    def decode_classifications(
        self,
        logits: torch.Tensor | np.ndarray,
        top_k: int = 5,
    ) -> list[Prediction]:
        """Decode raw classification logits into sorted predictions."""
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)

        if logits.ndim == 2:
            logits = logits[0]

        probs = F.softmax(logits.float(), dim=0)
        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(probs)))

        return [
            Prediction(
                label=self.labels.get(idx.item(), f"class_{idx.item()}"),
                confidence=round(prob.item(), 6),
                class_index=idx.item(),
            )
            for prob, idx in zip(top_probs, top_indices)
        ]

    def decode_detections(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.45,
    ) -> list[BBox]:
        """Decode raw detection outputs with confidence filtering and NMS."""
        detections: list[BBox] = []

        for i in range(len(scores)):
            if scores[i] >= conf_threshold:
                detections.append(BBox(
                    x1=float(boxes[i][0]),
                    y1=float(boxes[i][1]),
                    x2=float(boxes[i][2]),
                    y2=float(boxes[i][3]),
                    label=self.labels.get(int(class_ids[i]), f"class_{int(class_ids[i])}"),
                    confidence=float(scores[i]),
                    class_id=int(class_ids[i]),
                ))

        return non_max_suppression(detections, nms_threshold)

    def draw_boxes(
        self,
        image: Image.Image,
        detections: list[BBox],
        line_width: int = 3,
        font_size: int = 16,
        show_confidence: bool = True,
    ) -> Image.Image:
        """Draw bounding boxes and labels on an image."""
        annotated = image.copy()
        if annotated.mode != "RGB":
            annotated = annotated.convert("RGB")

        draw = ImageDraw.Draw(annotated)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

        for i, det in enumerate(detections):
            color = BOX_COLORS[i % len(BOX_COLORS)]

            # Draw box
            draw.rectangle(
                [det.x1, det.y1, det.x2, det.y2],
                outline=color,
                width=line_width,
            )

            # Build label text
            label = det.label
            if show_confidence:
                label = f"{label} {det.confidence:.2f}"

            # Draw label background
            bbox = draw.textbbox((det.x1, det.y1), label, font=font)
            text_h = bbox[3] - bbox[1]
            text_w = bbox[2] - bbox[0]
            draw.rectangle(
                [det.x1, det.y1 - text_h - 4, det.x1 + text_w + 4, det.y1],
                fill=color,
            )
            draw.text((det.x1 + 2, det.y1 - text_h - 2), label, fill="white", font=font)

        return annotated

    def format_results(
        self,
        results: list[dict],
        output_format: str = "json",
    ) -> str:
        """Format results as JSON or CSV string."""
        if output_format == "json":
            return json.dumps(results, indent=2, default=str)
        elif output_format == "csv":
            if not results:
                return ""
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {output_format}. Use 'json' or 'csv'.")

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp / np.sum(exp, axis=-1, keepdims=True)
