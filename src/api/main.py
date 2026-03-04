"""FastAPI application for the computer vision service."""

from __future__ import annotations

import io
import time

import httpx
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from prometheus_client import Counter, Histogram, generate_latest

from src.api.schemas import (
    BatchClassificationResponse,
    BatchDetectionResponse,
    BBoxSchema,
    BenchmarkRequest,
    BenchmarkResponse,
    ClassificationResponse,
    DetectionResponse,
    EmbeddingResponse,
    ErrorResponse,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    ModelInfoSchema,
    ModelsListResponse,
    PredictionSchema,
    SimilarityRequest,
    SimilarityResponse,
    SimilarityResultSchema,
    URLClassifyRequest,
)
from src.config.settings import get_settings
from src.evaluation.evaluator import CVEvaluator
from src.models.embedder import SimilaritySearch
from src.processing.preprocessor import ImagePreprocessor
from src.serving.model_manager import ModelManager
from src.utils.logging import get_logger, setup_logging

# ── Metrics ─────────────────────────────────────────────────────────────────

REQUEST_COUNT = Counter("cv_requests_total", "Total requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram("cv_request_latency_ms", "Request latency in ms", ["endpoint"])

# ── App setup ───────────────────────────────────────────────────────────────

settings = get_settings()
setup_logging(settings.LOG_LEVEL, settings.LOG_FORMAT)
logger = get_logger(__name__)

app = FastAPI(
    title="Computer Vision API",
    description="Production computer vision service: classification, detection, similarity search",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singletons initialized on startup
model_manager: ModelManager | None = None
similarity_search: SimilaritySearch | None = None
preprocessor = ImagePreprocessor()
evaluator = CVEvaluator()


@app.on_event("startup")
async def startup() -> None:
    global model_manager, similarity_search
    model_manager = ModelManager(settings)
    similarity_search = SimilaritySearch(persist_directory=str(settings.CHROMA_DIR))
    logger.info("api_started", host=settings.API_HOST, port=settings.API_PORT)


def _get_manager() -> ModelManager:
    assert model_manager is not None, "ModelManager not initialized"
    return model_manager


def _get_similarity() -> SimilaritySearch:
    assert similarity_search is not None, "SimilaritySearch not initialized"
    return similarity_search


async def _read_upload(file: UploadFile) -> Image.Image:
    """Read an uploaded file into a PIL Image."""
    content = await file.read()
    validation = preprocessor.validate_image(raw_bytes=content)
    if not validation.valid:
        raise ValueError("; ".join(validation.errors))
    return Image.open(io.BytesIO(content)).convert("RGB")


# ── Classification endpoints ────────────────────────────────────────────────

@app.post("/api/v1/classify", response_model=ClassificationResponse)
async def classify_image(
    file: UploadFile = File(...),
    model: str = Form(default="resnet50"),
    top_k: int = Form(default=5),
) -> ClassificationResponse:
    """Classify an uploaded image."""
    start = time.perf_counter()
    try:
        image = await _read_upload(file)
        classifier = _get_manager().get_model(model, task="classify")
        result = classifier.predict(image, top_k=top_k)

        REQUEST_COUNT.labels(endpoint="classify", status="success").inc()
        latency = (time.perf_counter() - start) * 1000
        REQUEST_LATENCY.labels(endpoint="classify").observe(latency)

        return ClassificationResponse(
            predictions=[
                PredictionSchema(
                    label=p.label, confidence=p.confidence, class_index=p.class_index
                )
                for p in result.top_predictions
            ],
            model=model,
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="classify", status="error").inc()
        raise _api_error(e)


@app.post("/api/v1/classify/batch", response_model=BatchClassificationResponse)
async def classify_batch(
    files: list[UploadFile] = File(...),
    model: str = Form(default="resnet50"),
    top_k: int = Form(default=5),
) -> BatchClassificationResponse:
    """Classify multiple uploaded images."""
    start = time.perf_counter()
    images = [await _read_upload(f) for f in files]
    classifier = _get_manager().get_model(model, task="classify")
    results = classifier.predict_batch(images, top_k=top_k)

    total_latency = (time.perf_counter() - start) * 1000
    return BatchClassificationResponse(
        results=[
            ClassificationResponse(
                predictions=[
                    PredictionSchema(
                        label=p.label, confidence=p.confidence, class_index=p.class_index
                    )
                    for p in r.top_predictions
                ],
                model=model,
                latency_ms=r.latency_ms,
            )
            for r in results
        ],
        total=len(results),
        total_latency_ms=round(total_latency, 2),
    )


@app.post("/api/v1/classify/url", response_model=ClassificationResponse)
async def classify_url(request: URLClassifyRequest) -> ClassificationResponse:
    """Classify an image from URL."""
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(request.url)
        resp.raise_for_status()
    image = Image.open(io.BytesIO(resp.content)).convert("RGB")
    classifier = _get_manager().get_model(request.model, task="classify")
    result = classifier.predict(image, top_k=request.top_k)
    latency = (time.perf_counter() - start) * 1000

    return ClassificationResponse(
        predictions=[
            PredictionSchema(label=p.label, confidence=p.confidence, class_index=p.class_index)
            for p in result.top_predictions
        ],
        model=request.model,
        latency_ms=round(latency, 2),
    )


# ── Detection endpoints ─────────────────────────────────────────────────────

@app.post("/api/v1/detect", response_model=DetectionResponse)
async def detect_objects(
    file: UploadFile = File(...),
    model: str = Form(default="fasterrcnn_mobilenet"),
    confidence: float = Form(default=0.5),
) -> DetectionResponse:
    """Run object detection on an uploaded image."""
    start = time.perf_counter()
    image = await _read_upload(file)
    detector = _get_manager().get_model(model, task="detect")
    result = detector.detect(image)
    latency = (time.perf_counter() - start) * 1000

    REQUEST_COUNT.labels(endpoint="detect", status="success").inc()
    REQUEST_LATENCY.labels(endpoint="detect").observe(latency)

    return DetectionResponse(
        detections=[
            BBoxSchema(
                x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
                label=d.label, confidence=d.confidence,
            )
            for d in result.detections
            if d.confidence >= confidence
        ],
        count=result.count,
        model=model,
        latency_ms=round(latency, 2),
        image_size=list(result.image_size),
    )


@app.post("/api/v1/detect/batch", response_model=BatchDetectionResponse)
async def detect_batch(
    files: list[UploadFile] = File(...),
    model: str = Form(default="fasterrcnn_mobilenet"),
    confidence: float = Form(default=0.5),
) -> BatchDetectionResponse:
    """Run detection on multiple images."""
    start = time.perf_counter()
    images = [await _read_upload(f) for f in files]
    detector = _get_manager().get_model(model, task="detect")
    results = detector.detect_batch(images)
    total_latency = (time.perf_counter() - start) * 1000

    return BatchDetectionResponse(
        results=[
            DetectionResponse(
                detections=[
                    BBoxSchema(
                        x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
                        label=d.label, confidence=d.confidence,
                    )
                    for d in r.detections if d.confidence >= confidence
                ],
                count=r.count,
                model=model,
                latency_ms=r.latency_ms,
                image_size=list(r.image_size),
            )
            for r in results
        ],
        total=len(results),
        total_latency_ms=round(total_latency, 2),
    )


# ── Embedding & Similarity endpoints ────────────────────────────────────────

@app.post("/api/v1/embed", response_model=EmbeddingResponse)
async def embed_image(
    file: UploadFile = File(...),
    model: str = Form(default="resnet50"),
) -> EmbeddingResponse:
    """Get the embedding vector of an image."""
    start = time.perf_counter()
    image = await _read_upload(file)
    embedder = _get_manager().get_model(model, task="embed")
    embedding = embedder.embed(image)
    latency = (time.perf_counter() - start) * 1000

    return EmbeddingResponse(
        embedding=embedding.tolist(),
        dimension=len(embedding),
        model=model,
        latency_ms=round(latency, 2),
    )


@app.post("/api/v1/index", response_model=IndexResponse)
async def index_image(
    file: UploadFile = File(...),
    image_id: str = Form(...),
    model: str = Form(default="resnet50"),
    metadata: str = Form(default="{}"),
) -> IndexResponse:
    """Index an image for similarity search."""
    import json
    image = await _read_upload(file)
    embedder = _get_manager().get_model(model, task="embed")
    embedding = embedder.embed(image)
    meta = json.loads(metadata)

    search = _get_similarity()
    search.index_image(image_id, embedding, meta)

    return IndexResponse(
        image_id=image_id,
        indexed=True,
        collection_size=search.count(),
    )


@app.post("/api/v1/similar", response_model=SimilarityResponse)
async def find_similar(
    file: UploadFile = File(...),
    k: int = Form(default=10),
    model: str = Form(default="resnet50"),
) -> SimilarityResponse:
    """Find similar images to an uploaded query image."""
    image = await _read_upload(file)
    embedder = _get_manager().get_model(model, task="embed")
    query_embedding = embedder.embed(image)

    search = _get_similarity()
    results = search.search(query_embedding, k=k)

    return SimilarityResponse(
        results=[
            SimilarityResultSchema(
                image_id=r.image_id, score=r.score, metadata=r.metadata
            )
            for r in results
        ],
        query_model=model,
        k=k,
    )


# ── Models & Benchmark ──────────────────────────────────────────────────────

@app.get("/api/v1/models", response_model=ModelsListResponse)
async def list_models() -> ModelsListResponse:
    """List all currently loaded models."""
    models = _get_manager().list_models()
    return ModelsListResponse(
        models=[
            ModelInfoSchema(
                name=m.name, task=m.task, backend=m.backend, device=m.device,
                loaded_at=m.loaded_at, last_used=m.last_used,
                inference_count=m.inference_count,
            )
            for m in models
        ],
        total=len(models),
    )


@app.post("/api/v1/benchmark", response_model=BenchmarkResponse)
async def run_benchmark(request: BenchmarkRequest) -> BenchmarkResponse:
    """Benchmark model inference latency."""
    dummy_image = Image.new("RGB", (224, 224), color="blue")
    model = _get_manager().get_model(request.model, task=request.task)
    report = evaluator.benchmark_latency(
        model, dummy_image, n_runs=request.n_runs, task=request.task
    )
    return BenchmarkResponse(
        model=request.model,
        task=request.task,
        mean_ms=report.mean_ms,
        median_ms=report.median_ms,
        p50_ms=report.p50_ms,
        p95_ms=report.p95_ms,
        p99_ms=report.p99_ms,
        min_ms=report.min_ms,
        max_ms=report.max_ms,
        throughput_fps=report.throughput_fps,
        n_runs=report.n_runs,
    )


# ── Health & Metrics ─────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=_get_manager().loaded_count,
        device=settings.DEVICE,
    )


@app.get("/metrics")
async def metrics() -> JSONResponse:
    """Prometheus-compatible metrics endpoint."""
    return JSONResponse(
        content=generate_latest().decode("utf-8"),
        media_type="text/plain",
    )


# ── Error handling ──────────────────────────────────────────────────────────

def _api_error(e: Exception) -> Exception:
    from fastapi import HTTPException
    logger.error("api_error", error=str(e))
    raise HTTPException(status_code=400, detail=str(e))


@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError) -> JSONResponse:  # type: ignore[override]
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(error="Bad Request", detail=str(exc)).model_dump(),
    )


@app.exception_handler(Exception)
async def general_error_handler(request, exc: Exception) -> JSONResponse:  # type: ignore[override]
    logger.error("unhandled_error", error=str(exc))
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error="Internal Server Error", detail=str(exc)).model_dump(),
    )
