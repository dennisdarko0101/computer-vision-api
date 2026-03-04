"""Tests for BatchProcessor — 12 tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.processing.batch_processor import BatchProcessor, BatchResult


@pytest.fixture
def batch_processor() -> BatchProcessor:
    return BatchProcessor(max_workers=2, timeout=10.0)


def dummy_process(image: Image.Image) -> dict:
    """Simple processing function for tests."""
    return {"width": image.size[0], "height": image.size[1], "mode": image.mode}


def failing_process(image: Image.Image) -> dict:
    """Process function that always fails."""
    raise ValueError("Intentional failure")


class TestProcessDirectory:
    def test_processes_all_images(self, batch_processor, image_dir):
        result = batch_processor.process_directory(image_dir, dummy_process)
        assert result.total == 5  # 5 jpg files, ignores .txt
        assert result.successful == 5
        assert result.failed == 0

    def test_records_total_time(self, batch_processor, image_dir):
        result = batch_processor.process_directory(image_dir, dummy_process)
        assert result.total_time_ms > 0

    def test_per_item_results(self, batch_processor, image_dir):
        result = batch_processor.process_directory(image_dir, dummy_process)
        for item in result.items:
            assert item.success is True
            assert item.result is not None
            assert item.processing_time_ms > 0

    def test_handles_errors_per_item(self, batch_processor, image_dir):
        result = batch_processor.process_directory(image_dir, failing_process)
        assert result.total == 5
        assert result.failed == 5
        assert result.successful == 0
        for item in result.items:
            assert item.error is not None

    def test_directory_not_found(self, batch_processor):
        with pytest.raises(FileNotFoundError):
            batch_processor.process_directory("/nonexistent/dir", dummy_process)

    def test_custom_extensions(self, batch_processor, image_dir):
        # Only look for .png files (none exist)
        result = batch_processor.process_directory(
            image_dir, dummy_process, extensions={".png"}
        )
        assert result.total == 0


class TestProcessFiles:
    def test_processes_file_list(self, batch_processor, image_dir):
        files = list(Path(image_dir).glob("*.jpg"))
        result = batch_processor.process_files(files, dummy_process)
        assert result.total == len(files)
        assert result.successful == len(files)

    def test_handles_missing_file(self, batch_processor, tmp_path):
        files = [tmp_path / "nonexistent.jpg"]
        result = batch_processor.process_files(files, dummy_process)
        assert result.total == 1
        assert result.failed == 1

    def test_mixed_success_failure(self, batch_processor, image_dir, tmp_path):
        files = list(Path(image_dir).glob("*.jpg"))[:2] + [tmp_path / "missing.jpg"]
        result = batch_processor.process_files(files, dummy_process)
        assert result.successful == 2
        assert result.failed == 1


class TestBatchResult:
    def test_to_dict(self):
        result = BatchResult()
        from src.processing.batch_processor import BatchItemResult
        result.add(BatchItemResult(source="a.jpg", success=True, result={"ok": True}))
        result.add(BatchItemResult(source="b.jpg", success=False, error="fail"))
        d = result.to_dict()
        assert d["total"] == 2
        assert d["successful"] == 1
        assert d["failed"] == 1
        assert len(d["items"]) == 2


class TestExportResults:
    def test_export_json(self, batch_processor, image_dir, tmp_path):
        result = batch_processor.process_directory(image_dir, dummy_process)
        path = batch_processor.export_results(result, tmp_path / "out.json", fmt="json")
        assert Path(path).exists()
        import json
        with open(path) as f:
            data = json.load(f)
        assert data["total"] == 5

    def test_export_csv(self, batch_processor, image_dir, tmp_path):
        result = batch_processor.process_directory(image_dir, dummy_process)
        path = batch_processor.export_results(result, tmp_path / "out.csv", fmt="csv")
        assert Path(path).exists()
        content = Path(path).read_text()
        assert "source" in content
