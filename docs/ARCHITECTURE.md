# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────┐
│                   FastAPI Gateway                     │
│  /classify  /detect  /embed  /similar  /benchmark    │
└──────────┬──────────┬──────────┬────────────────────┘
           │          │          │
    ┌──────▼──┐ ┌─────▼───┐ ┌───▼──────┐
    │Classifier│ │Detector │ │Embedder  │
    │ ResNet50 │ │FasterRCNN│ │FeatureEx │
    │ EffNet   │ │ Mock    │ │          │
    │ MobNet   │ │         │ │          │
    └──────┬──┘ └─────┬───┘ └───┬──────┘
           │          │          │
    ┌──────▼──────────▼──────────▼──────┐
    │         Model Manager              │
    │  LRU cache · ONNX/PyTorch toggle   │
    │  Warmup · Multi-backend serving    │
    └──────────────┬────────────────────┘
                   │
    ┌──────────────▼────────────────────┐
    │       Processing Pipeline          │
    │  Preprocessor → Model → Postproc   │
    │  Batch processing · Async URLs     │
    └──────────────┬────────────────────┘
                   │
    ┌──────────────▼────────────────────┐
    │        Evaluation Suite            │
    │  Accuracy · mAP · Latency bench    │
    │  Model comparison reports          │
    └──────────────┬────────────────────┘
                   │
    ┌──────────────▼────────────────────┐
    │         Vector Store (ChromaDB)    │
    │  Cosine similarity · HNSW index    │
    └───────────────────────────────────┘
```

## Module Responsibilities

| Module | Purpose |
|--------|---------|
| `src/config` | Pydantic settings, env var management |
| `src/models` | Classifier, detector, embedder, ONNX optimizer |
| `src/processing` | Pre/post-processing, batch pipeline |
| `src/evaluation` | Metrics (accuracy, mAP), latency benchmarking |
| `src/serving` | Model lifecycle management with LRU eviction |
| `src/api` | FastAPI endpoints and Pydantic schemas |
| `src/utils` | Structured logging |

## Design Decisions

1. **Model Manager with LRU**: Prevents OOM by capping loaded models and evicting least-recently-used.
2. **ONNX Dual Backend**: PyTorch for development, ONNX for production CPU inference (1.5-3x speedup).
3. **ChromaDB for Similarity**: Lightweight vector DB with HNSW indexing, no external infra needed.
4. **Mock Detector**: Enables full test coverage without downloading large model weights.
5. **Batch Processing**: Per-item error isolation — one corrupt image doesn't fail the batch.
