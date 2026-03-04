# Computer Vision API

Production computer vision service with image classification, object detection, and similarity search. Features ONNX optimization for fast CPU inference, batch processing, and Azure deployment.

Built as part of ML engineering at **Sinewy Technologies**.

```
┌─────────────────────────────────────────────────┐
│              FastAPI Gateway (12 endpoints)       │
├─────────┬──────────┬───────────┬────────────────┤
│Classify │ Detect   │ Embed     │ Similar Search  │
│ResNet50 │FasterRCNN│FeatureEx  │ ChromaDB/HNSW   │
│EffNet   │MobileNet │ L2 Norm   │ Cosine Sim      │
│MobNetV3 │Mock Mode │           │                 │
├─────────┴──────────┴───────────┴────────────────┤
│          Model Manager (LRU · ONNX toggle)       │
├──────────────────────────────────────────────────┤
│     Processing Pipeline (Batch · Async · NMS)    │
├──────────────────────────────────────────────────┤
│     Evaluation (Accuracy · mAP · Latency)        │
└──────────────────────────────────────────────────┘
```

## Features

- **Image Classification** — ResNet50, EfficientNet-B0, MobileNetV3 with ImageNet labels
- **Object Detection** — FasterRCNN-MobileNet with NMS post-processing
- **Image Similarity** — Feature extraction + ChromaDB vector search
- **ONNX Optimization** — Export, optimize, and benchmark PyTorch vs ONNX inference
- **Batch Processing** — Directory, URL, and file-list processing with error isolation
- **Evaluation Suite** — Accuracy, mAP, latency percentiles, model comparison
- **Production Ready** — Docker, Prometheus metrics, structured logging, health checks

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run API server
make run
# → http://localhost:8000/docs

# Run tests
make test
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/classify` | Classify uploaded image |
| POST | `/api/v1/classify/batch` | Classify multiple images |
| POST | `/api/v1/classify/url` | Classify image from URL |
| POST | `/api/v1/detect` | Object detection |
| POST | `/api/v1/detect/batch` | Batch detection |
| POST | `/api/v1/embed` | Get image embedding |
| POST | `/api/v1/index` | Index image for search |
| POST | `/api/v1/similar` | Find similar images |
| GET | `/api/v1/models` | List loaded models |
| POST | `/api/v1/benchmark` | Run latency benchmark |
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |

## Usage Examples

### Classify an image
```bash
curl -X POST http://localhost:8000/api/v1/classify \
  -F "file=@photo.jpg" \
  -F "model=resnet50" \
  -F "top_k=5"
```

### Object detection
```bash
curl -X POST http://localhost:8000/api/v1/detect \
  -F "file=@street.jpg" \
  -F "confidence=0.5"
```

### Similarity search
```bash
# Index images
curl -X POST http://localhost:8000/api/v1/index \
  -F "file=@image1.jpg" \
  -F "image_id=img001" \
  -F 'metadata={"category":"nature"}'

# Find similar
curl -X POST http://localhost:8000/api/v1/similar \
  -F "file=@query.jpg" \
  -F "k=10"
```

## ONNX Optimization

The ONNX pipeline provides significant speedup for CPU-only deployments:

```python
from src.models.onnx_optimizer import ONNXOptimizer, ONNXPredictor, benchmark_pytorch_vs_onnx

optimizer = ONNXOptimizer()
optimizer.export_to_onnx(model, input_shape=(1, 3, 224, 224), output_path="model.onnx")
optimized = optimizer.optimize_onnx("model.onnx")
predictor = ONNXPredictor(optimized)

# Benchmark
result = benchmark_pytorch_vs_onnx(model, predictor, (1, 3, 224, 224))
print(f"Speedup: {result.speedup}x")
```

Typical CPU inference speedups:
| Model | PyTorch (ms) | ONNX (ms) | Speedup |
|-------|-------------|-----------|---------|
| ResNet50 | ~45 | ~20 | ~2.2x |
| EfficientNet-B0 | ~35 | ~15 | ~2.3x |
| MobileNetV3 | ~15 | ~8 | ~1.9x |

## Batch Processing

```python
from src.processing.batch_processor import BatchProcessor

batch = BatchProcessor(max_workers=4)
results = batch.process_directory("./images/", classifier.predict)
print(f"Processed {results.successful}/{results.total} images")
```

## Deployment

```bash
# Docker
docker compose -f docker/docker-compose.yml up -d

# Azure Container Apps
az containerapp create --name cv-api --image ghcr.io/user/cv-api:latest ...
```

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for full deployment guide.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI + Uvicorn |
| Deep Learning | PyTorch + torchvision |
| Inference | ONNX Runtime |
| Vector Search | ChromaDB (HNSW) |
| Image Processing | Pillow + OpenCV |
| Configuration | pydantic-settings |
| Logging | structlog |
| Metrics | prometheus-client |
| HTTP Client | httpx |
| Testing | pytest + pytest-asyncio |
| CI/CD | GitHub Actions → Azure |
| Container | Docker (multi-stage) |

## Project Structure

```
src/
├── config/          # Settings, env var management
├── models/          # Classifier, detector, embedder, ONNX optimizer
├── processing/      # Pre/post-processing, batch pipeline
├── evaluation/      # Metrics, benchmarking, model comparison
├── serving/         # Model manager with LRU eviction
├── api/             # FastAPI endpoints + Pydantic schemas
└── utils/           # Structured logging
tests/
├── unit/            # 100+ unit tests with mock models
└── integration/     # API endpoint + pipeline tests
docker/              # Dockerfile (CPU/GPU), docker-compose
docs/                # Architecture, deployment, handoff
```

## License

MIT
