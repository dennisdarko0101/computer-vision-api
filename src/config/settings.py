"""Application settings using pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the computer vision service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CV_",
        case_sensitive=False,
    )

    # Paths
    MODEL_DIR: Path = Path("models")
    CHROMA_DIR: Path = Path("data/chroma")
    UPLOAD_DIR: Path = Path("data/uploads")

    # Compute
    DEVICE: str = "cpu"
    BATCH_SIZE: int = 16
    MAX_WORKERS: int = 4

    # Image constraints
    MAX_IMAGE_SIZE: int = 10_485_760  # 10 MB
    MAX_IMAGE_DIMENSION: int = 4096
    SUPPORTED_FORMATS: list[str] = ["JPEG", "PNG", "WEBP", "BMP", "TIFF"]

    # Model defaults
    DEFAULT_CLASSIFIER: str = "resnet50"
    DEFAULT_DETECTOR: str = "fasterrcnn_mobilenet"
    DEFAULT_EMBEDDER: str = "resnet50"
    CONFIDENCE_THRESHOLD: float = 0.5
    NMS_IOU_THRESHOLD: float = 0.45
    TOP_K: int = 5

    # ONNX
    ONNX_ENABLED: bool = False
    ONNX_OPTIMIZATION_LEVEL: str = "all"  # "basic", "extended", "all"

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    CORS_ORIGINS: list[str] = ["*"]

    # Serving
    MAX_LOADED_MODELS: int = 5
    MODEL_WARMUP: bool = True

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
