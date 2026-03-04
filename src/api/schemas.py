"""Pydantic request/response schemas for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Classification ──────────────────────────────────────────────────────────

class PredictionSchema(BaseModel):
    label: str
    confidence: float
    class_index: int


class ClassificationResponse(BaseModel):
    predictions: list[PredictionSchema]
    model: str
    latency_ms: float


class BatchClassificationResponse(BaseModel):
    results: list[ClassificationResponse]
    total: int
    total_latency_ms: float


class URLClassifyRequest(BaseModel):
    url: str = Field(..., description="URL of the image to classify")
    model: str = Field(default="resnet50", description="Model name")
    top_k: int = Field(default=5, ge=1, le=100)


# ── Detection ───────────────────────────────────────────────────────────────

class BBoxSchema(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    confidence: float


class DetectionResponse(BaseModel):
    detections: list[BBoxSchema]
    count: int
    model: str
    latency_ms: float
    image_size: list[int]


class BatchDetectionResponse(BaseModel):
    results: list[DetectionResponse]
    total: int
    total_latency_ms: float


# ── Embedding & Similarity ──────────────────────────────────────────────────

class EmbeddingResponse(BaseModel):
    embedding: list[float]
    dimension: int
    model: str
    latency_ms: float


class IndexRequest(BaseModel):
    image_id: str = Field(..., description="Unique ID for the image")
    metadata: dict = Field(default_factory=dict)


class IndexResponse(BaseModel):
    image_id: str
    indexed: bool
    collection_size: int


class SimilarityResultSchema(BaseModel):
    image_id: str
    score: float
    metadata: dict


class SimilarityResponse(BaseModel):
    results: list[SimilarityResultSchema]
    query_model: str
    k: int


class SimilarityRequest(BaseModel):
    k: int = Field(default=10, ge=1, le=100, description="Number of similar images to return")


# ── Models ──────────────────────────────────────────────────────────────────

class ModelInfoSchema(BaseModel):
    name: str
    task: str
    backend: str
    device: str
    loaded_at: float
    last_used: float
    inference_count: int


class ModelsListResponse(BaseModel):
    models: list[ModelInfoSchema]
    total: int


# ── Benchmark ───────────────────────────────────────────────────────────────

class BenchmarkRequest(BaseModel):
    model: str = Field(default="resnet50")
    task: str = Field(default="classify", pattern="^(classify|detect|embed)$")
    n_runs: int = Field(default=100, ge=10, le=1000)


class BenchmarkResponse(BaseModel):
    model: str
    task: str
    mean_ms: float
    median_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    throughput_fps: float
    n_runs: int


# ── Health ──────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: int
    device: str


# ── Errors ──────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
