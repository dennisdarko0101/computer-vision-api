"""Tests for ImagePreprocessor — 14 tests."""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from PIL import Image

from src.processing.preprocessor import ImagePreprocessor, ValidationResult


@pytest.fixture
def preprocessor() -> ImagePreprocessor:
    return ImagePreprocessor(target_size=(224, 224))


class TestResize:
    def test_resize_to_target(self, preprocessor, sample_image):
        resized = preprocessor.resize(sample_image)
        assert resized.size == (224, 224)

    def test_resize_custom_target(self, preprocessor, sample_image):
        resized = preprocessor.resize(sample_image, target_size=(128, 128))
        assert resized.size == (128, 128)

    def test_resize_no_aspect(self, preprocessor, large_image):
        resized = preprocessor.resize(large_image, keep_aspect=False)
        assert resized.size == (224, 224)

    def test_resize_keeps_rgb(self, preprocessor, sample_image):
        resized = preprocessor.resize(sample_image)
        assert resized.mode == "RGB"


class TestNormalize:
    def test_normalize_returns_tensor(self, preprocessor, sample_image):
        tensor = preprocessor.normalize(sample_image)
        assert isinstance(tensor, torch.Tensor)

    def test_normalize_shape(self, preprocessor, sample_image):
        tensor = preprocessor.normalize(sample_image)
        assert tensor.shape[0] == 3  # channels

    def test_normalize_custom_stats(self, preprocessor, sample_image):
        tensor = preprocessor.normalize(sample_image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        assert isinstance(tensor, torch.Tensor)


class TestAugment:
    def test_augment_returns_image(self, preprocessor, sample_image_224):
        augmented = preprocessor.augment(sample_image_224, ["horizontal_flip"])
        assert isinstance(augmented, Image.Image)

    def test_augment_grayscale(self, preprocessor, sample_image_224):
        augmented = preprocessor.augment(sample_image_224, ["grayscale"])
        assert isinstance(augmented, Image.Image)


class TestConversion:
    def test_to_tensor(self, preprocessor, sample_image):
        tensor = preprocessor.to_tensor(sample_image)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[0] == 3

    def test_from_bytes(self, preprocessor, sample_image_bytes):
        image = preprocessor.from_bytes(sample_image_bytes)
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"


class TestValidation:
    def test_validate_valid_image(self, preprocessor, sample_image_bytes):
        result = preprocessor.validate_image(raw_bytes=sample_image_bytes)
        assert result.valid is True
        assert result.errors == []

    def test_validate_corrupt_bytes(self, preprocessor):
        result = preprocessor.validate_image(raw_bytes=b"not_an_image")
        assert result.valid is False
        assert len(result.errors) > 0

    def test_validate_oversized(self, preprocessor):
        # Create preprocessor with very small size limit
        p = ImagePreprocessor(max_image_size=100)
        large_bytes = b"\x00" * 200
        result = p.validate_image(raw_bytes=large_bytes)
        assert result.valid is False

    def test_validate_no_image(self, preprocessor):
        result = preprocessor.validate_image()
        assert result.valid is False


class TestPipeline:
    def test_full_pipeline(self, preprocessor, sample_image):
        tensor = preprocessor.full_pipeline(sample_image)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)

    def test_full_pipeline_batch(self, preprocessor, sample_images_batch):
        batch_tensor = preprocessor.full_pipeline_batch(sample_images_batch)
        assert batch_tensor.shape == (4, 3, 224, 224)
