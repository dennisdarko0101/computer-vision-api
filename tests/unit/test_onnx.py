"""Tests for ONNX export, optimization, and prediction — 12 tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.onnx_optimizer import ONNXOptimizer, ONNXPredictor, benchmark_pytorch_vs_onnx
from tests.conftest import MockClassifierModel


@pytest.fixture
def optimizer() -> ONNXOptimizer:
    return ONNXOptimizer(optimization_level="basic")


@pytest.fixture
def mock_model() -> MockClassifierModel:
    model = MockClassifierModel(num_classes=10)
    model.eval()
    return model


@pytest.fixture
def onnx_path(tmp_path, optimizer, mock_model) -> str:
    """Export mock model to ONNX and return path."""
    path = tmp_path / "test_model.onnx"
    optimizer.export_to_onnx(mock_model, (1, 3, 224, 224), path)
    return str(path)


class TestONNXOptimizer:
    def test_export_creates_file(self, tmp_path, optimizer, mock_model):
        path = tmp_path / "model.onnx"
        result = optimizer.export_to_onnx(mock_model, (1, 3, 224, 224), path)
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    def test_export_returns_path_string(self, tmp_path, optimizer, mock_model):
        path = tmp_path / "model.onnx"
        result = optimizer.export_to_onnx(mock_model, (1, 3, 224, 224), path)
        assert isinstance(result, str)

    def test_export_creates_parent_dirs(self, tmp_path, optimizer, mock_model):
        path = tmp_path / "subdir" / "deep" / "model.onnx"
        result = optimizer.export_to_onnx(mock_model, (1, 3, 224, 224), path)
        assert Path(result).exists()

    def test_optimize_creates_optimized_file(self, onnx_path, optimizer):
        optimized = optimizer.optimize_onnx(onnx_path)
        assert Path(optimized).exists()
        assert "optimized" in optimized

    def test_optimize_returns_path(self, onnx_path, optimizer):
        result = optimizer.optimize_onnx(onnx_path)
        assert isinstance(result, str)


class TestONNXPredictor:
    def test_load_model(self, onnx_path):
        predictor = ONNXPredictor(onnx_path)
        assert predictor.session is not None

    def test_predict_single(self, onnx_path):
        predictor = ONNXPredictor(onnx_path)
        inp = np.random.randn(1, 3, 224, 224).astype(np.float32)
        output = predictor.predict(inp)
        assert output.shape[0] == 1
        assert output.shape[1] == 10

    def test_predict_expands_dims(self, onnx_path):
        predictor = ONNXPredictor(onnx_path)
        inp = np.random.randn(3, 224, 224).astype(np.float32)
        output = predictor.predict(inp)
        assert output.shape == (1, 10)

    def test_predict_batch(self, onnx_path):
        predictor = ONNXPredictor(onnx_path)
        inp = np.random.randn(4, 3, 224, 224).astype(np.float32)
        output = predictor.predict_batch(inp)
        assert output.shape == (4, 10)

    def test_input_name(self, onnx_path):
        predictor = ONNXPredictor(onnx_path)
        assert predictor.input_name == "input"

    def test_output_name(self, onnx_path):
        predictor = ONNXPredictor(onnx_path)
        assert predictor.output_name == "output"


class TestBenchmark:
    def test_benchmark_runs(self, mock_model, onnx_path):
        predictor = ONNXPredictor(onnx_path)
        result = benchmark_pytorch_vs_onnx(
            mock_model, predictor, (1, 3, 224, 224), n_runs=5, warmup_runs=2,
        )
        assert result.n_runs == 5
        assert result.pytorch_mean_ms > 0
        assert result.onnx_mean_ms > 0
        assert result.speedup > 0
