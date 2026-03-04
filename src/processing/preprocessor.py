"""Image preprocessing pipeline for computer vision models."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Standard ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_DIMENSION = 4096


@dataclass
class ValidationResult:
    """Image validation outcome."""
    valid: bool
    errors: list[str]
    width: int = 0
    height: int = 0
    format: str = ""
    file_size: int = 0


class ImagePreprocessor:
    """Configurable image preprocessing for various CV models."""

    def __init__(
        self,
        target_size: tuple[int, int] = (224, 224),
        mean: list[float] | None = None,
        std: list[float] | None = None,
        max_image_size: int = MAX_IMAGE_BYTES,
        max_dimension: int = MAX_DIMENSION,
        supported_formats: set[str] | None = None,
    ) -> None:
        self.target_size = target_size
        self.mean = mean or IMAGENET_MEAN
        self.std = std or IMAGENET_STD
        self.max_image_size = max_image_size
        self.max_dimension = max_dimension
        self.supported_formats = supported_formats or SUPPORTED_FORMATS

    def resize(
        self,
        image: Image.Image,
        target_size: tuple[int, int] | None = None,
        keep_aspect: bool = True,
    ) -> Image.Image:
        """Resize image, optionally preserving aspect ratio with center crop."""
        size = target_size or self.target_size
        if keep_aspect:
            # Resize shortest edge then center crop
            w, h = image.size
            scale = max(size[0] / w, size[1] / h)
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.BILINEAR)
            # Center crop
            left = (new_w - size[0]) // 2
            top = (new_h - size[1]) // 2
            image = image.crop((left, top, left + size[0], top + size[1]))
        else:
            image = image.resize(size, Image.BILINEAR)
        return image

    def normalize(
        self,
        image: Image.Image,
        mean: list[float] | None = None,
        std: list[float] | None = None,
    ) -> torch.Tensor:
        """Convert image to normalized tensor."""
        m = mean or self.mean
        s = std or self.std
        tensor = transforms.ToTensor()(image)
        return transforms.Normalize(mean=m, std=s)(tensor)

    def augment(
        self,
        image: Image.Image,
        augmentations: list[str] | None = None,
    ) -> Image.Image:
        """Apply data augmentation transforms for training."""
        augs = augmentations or ["horizontal_flip", "color_jitter", "random_crop"]
        augmented = image.copy()

        aug_map: dict[str, transforms.transforms.Transform] = {
            "horizontal_flip": transforms.RandomHorizontalFlip(p=1.0),
            "vertical_flip": transforms.RandomVerticalFlip(p=1.0),
            "color_jitter": transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            "random_rotation": transforms.RandomRotation(degrees=15),
            "random_crop": transforms.RandomResizedCrop(
                size=self.target_size, scale=(0.8, 1.0)
            ),
            "grayscale": transforms.RandomGrayscale(p=1.0),
        }

        for aug_name in augs:
            if aug_name in aug_map:
                augmented = aug_map[aug_name](augmented)

        return augmented

    def to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to torch Tensor (0-1 range, CHW)."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        return transforms.ToTensor()(image)

    def from_bytes(self, raw_bytes: bytes) -> Image.Image:
        """Load a PIL Image from raw bytes."""
        return Image.open(io.BytesIO(raw_bytes)).convert("RGB")

    def validate_image(
        self,
        image: Image.Image | None = None,
        raw_bytes: bytes | None = None,
    ) -> ValidationResult:
        """Validate image format, size, and integrity."""
        errors: list[str] = []

        if raw_bytes is not None:
            if len(raw_bytes) > self.max_image_size:
                errors.append(
                    f"File size {len(raw_bytes)} exceeds maximum {self.max_image_size} bytes"
                )
            try:
                image = Image.open(io.BytesIO(raw_bytes))
                image.verify()
                # Re-open after verify (verify closes the image)
                image = Image.open(io.BytesIO(raw_bytes))
            except Exception as e:
                return ValidationResult(
                    valid=False,
                    errors=[f"Corrupt or unreadable image: {e}"],
                    file_size=len(raw_bytes) if raw_bytes else 0,
                )

        if image is None:
            return ValidationResult(valid=False, errors=["No image provided"])

        fmt = image.format or "UNKNOWN"
        w, h = image.size

        if fmt != "UNKNOWN" and fmt not in self.supported_formats:
            errors.append(f"Unsupported format: {fmt}. Supported: {self.supported_formats}")

        if w > self.max_dimension or h > self.max_dimension:
            errors.append(
                f"Image dimensions {w}x{h} exceed maximum {self.max_dimension}px"
            )

        if w == 0 or h == 0:
            errors.append("Image has zero dimensions")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            width=w,
            height=h,
            format=fmt,
            file_size=len(raw_bytes) if raw_bytes else 0,
        )

    def full_pipeline(self, image: Image.Image) -> torch.Tensor:
        """Full preprocessing: resize + normalize to tensor ready for model input."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = self.resize(image)
        return self.normalize(image)

    def full_pipeline_batch(self, images: list[Image.Image]) -> torch.Tensor:
        """Preprocess a batch of images."""
        return torch.stack([self.full_pipeline(img) for img in images])
