"""Batch image processing with async support and error handling."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx
from PIL import Image

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BatchItemResult:
    """Result for a single item in a batch."""
    source: str
    success: bool
    result: Any = None
    error: str | None = None
    processing_time_ms: float = 0.0


@dataclass
class BatchResult:
    """Aggregated batch processing results."""
    items: list[BatchItemResult] = field(default_factory=list)
    total: int = 0
    successful: int = 0
    failed: int = 0
    total_time_ms: float = 0.0

    def add(self, item: BatchItemResult) -> None:
        self.items.append(item)
        self.total += 1
        if item.success:
            self.successful += 1
        else:
            self.failed += 1

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "successful": self.successful,
            "failed": self.failed,
            "total_time_ms": round(self.total_time_ms, 2),
            "items": [
                {
                    "source": item.source,
                    "success": item.success,
                    "result": item.result,
                    "error": item.error,
                    "processing_time_ms": round(item.processing_time_ms, 2),
                }
                for item in self.items
            ],
        }


class BatchProcessor:
    """Process batches of images from directories, URLs, or file lists."""

    def __init__(
        self,
        max_workers: int = 4,
        timeout: float = 30.0,
        max_image_size: int = 10 * 1024 * 1024,
    ) -> None:
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_image_size = max_image_size

    def process_directory(
        self,
        dir_path: str | Path,
        process_fn: Callable[[Image.Image], Any],
        extensions: set[str] | None = None,
    ) -> BatchResult:
        """Process all images in a directory."""
        dir_path = Path(dir_path)
        exts = extensions or {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

        if not dir_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        image_files = [
            f for f in sorted(dir_path.iterdir())
            if f.suffix.lower() in exts and f.is_file()
        ]

        batch_start = time.perf_counter()
        batch_result = BatchResult()

        for file_path in image_files:
            item_start = time.perf_counter()
            try:
                image = Image.open(file_path).convert("RGB")
                result = process_fn(image)
                elapsed = (time.perf_counter() - item_start) * 1000
                batch_result.add(BatchItemResult(
                    source=str(file_path),
                    success=True,
                    result=result,
                    processing_time_ms=elapsed,
                ))
            except Exception as e:
                elapsed = (time.perf_counter() - item_start) * 1000
                logger.warning("batch_item_failed", source=str(file_path), error=str(e))
                batch_result.add(BatchItemResult(
                    source=str(file_path),
                    success=False,
                    error=str(e),
                    processing_time_ms=elapsed,
                ))

        batch_result.total_time_ms = (time.perf_counter() - batch_start) * 1000
        return batch_result

    async def process_urls(
        self,
        urls: list[str],
        process_fn: Callable[[Image.Image], Any],
    ) -> BatchResult:
        """Download and process images from URLs concurrently."""
        semaphore = asyncio.Semaphore(self.max_workers)
        batch_start = time.perf_counter()
        batch_result = BatchResult()

        async def process_single(url: str) -> BatchItemResult:
            async with semaphore:
                item_start = time.perf_counter()
                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        resp = await client.get(url)
                        resp.raise_for_status()

                        if len(resp.content) > self.max_image_size:
                            raise ValueError(
                                f"Image too large: {len(resp.content)} bytes"
                            )

                        image = Image.open(httpx._content.ByteStream(resp.content)).convert("RGB")
                except Exception:
                    # Fallback: use io.BytesIO
                    import io
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        resp = await client.get(url)
                        resp.raise_for_status()
                        image = Image.open(io.BytesIO(resp.content)).convert("RGB")

                result = process_fn(image)
                elapsed = (time.perf_counter() - item_start) * 1000

                return BatchItemResult(
                    source=url,
                    success=True,
                    result=result,
                    processing_time_ms=elapsed,
                )

        tasks = []
        for url in urls:
            tasks.append(process_single(url))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning("batch_url_failed", url=urls[i], error=str(result))
                batch_result.add(BatchItemResult(
                    source=urls[i],
                    success=False,
                    error=str(result),
                ))
            else:
                batch_result.add(result)

        batch_result.total_time_ms = (time.perf_counter() - batch_start) * 1000
        return batch_result

    def process_files(
        self,
        file_paths: list[str | Path],
        process_fn: Callable[[Image.Image], Any],
    ) -> BatchResult:
        """Process a list of image file paths."""
        batch_start = time.perf_counter()
        batch_result = BatchResult()

        for file_path in file_paths:
            item_start = time.perf_counter()
            try:
                image = Image.open(file_path).convert("RGB")
                result = process_fn(image)
                elapsed = (time.perf_counter() - item_start) * 1000
                batch_result.add(BatchItemResult(
                    source=str(file_path),
                    success=True,
                    result=result,
                    processing_time_ms=elapsed,
                ))
            except Exception as e:
                elapsed = (time.perf_counter() - item_start) * 1000
                batch_result.add(BatchItemResult(
                    source=str(file_path),
                    success=False,
                    error=str(e),
                    processing_time_ms=elapsed,
                ))

        batch_result.total_time_ms = (time.perf_counter() - batch_start) * 1000
        return batch_result

    @staticmethod
    def export_results(batch_result: BatchResult, output_path: str | Path, fmt: str = "json") -> str:
        """Export batch results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "json":
            with open(output_path, "w") as f:
                json.dump(batch_result.to_dict(), f, indent=2, default=str)
        elif fmt == "csv":
            import csv
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["source", "success", "error", "processing_time_ms"])
                for item in batch_result.items:
                    writer.writerow([
                        item.source, item.success, item.error or "", item.processing_time_ms
                    ])
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        return str(output_path)
