"""Shared test fixtures — mock models, sample images, no large downloads needed."""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image


# ── Sample images ───────────────────────────────────────────────────────────

@pytest.fixture
def sample_image() -> Image.Image:
    """A small random RGB image."""
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


@pytest.fixture
def sample_image_224() -> Image.Image:
    """224x224 random image matching model input size."""
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


@pytest.fixture
def sample_image_bytes() -> bytes:
    """JPEG bytes of a sample image."""
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8), "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_grayscale() -> Image.Image:
    """Grayscale image for mode conversion tests."""
    arr = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    return Image.fromarray(arr, "L")


@pytest.fixture
def sample_images_batch() -> list[Image.Image]:
    """Batch of 4 random images."""
    return [
        Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8), "RGB")
        for _ in range(4)
    ]


@pytest.fixture
def large_image() -> Image.Image:
    """Large image for resize tests."""
    arr = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


# ── Mock models ─────────────────────────────────────────────────────────────

class MockClassifierModel(nn.Module):
    """Tiny classifier outputting random logits for 10 classes."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 224 * 224, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return self.fc(x)


class MockEmbedderModel(nn.Module):
    """Tiny embedder outputting 128-dim vectors."""

    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 224 * 224, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return self.fc(x)


@pytest.fixture
def mock_classifier_model() -> MockClassifierModel:
    model = MockClassifierModel(num_classes=10)
    model.eval()
    return model


@pytest.fixture
def mock_embedder_model() -> MockEmbedderModel:
    model = MockEmbedderModel(embedding_dim=128)
    model.eval()
    return model


@pytest.fixture
def mock_labels() -> dict[int, str]:
    """10 mock class labels."""
    return {i: f"class_{i}" for i in range(10)}


# ── Temp directory fixture ──────────────────────────────────────────────────

@pytest.fixture
def image_dir(tmp_path) -> str:
    """Directory with a few sample images."""
    for i in range(5):
        img = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8), "RGB"
        )
        img.save(tmp_path / f"image_{i}.jpg")
    # Also add one non-image file
    (tmp_path / "readme.txt").write_text("not an image")
    return str(tmp_path)
