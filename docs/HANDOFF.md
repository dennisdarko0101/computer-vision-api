# Handoff Document

## Project Summary

**Computer Vision API** — A production-grade FastAPI service for image classification, object detection, and similarity search. Built with ONNX optimization, batch processing, and Azure deployment support. Developed as part of ML engineering work at Sinewy Technologies.

## What's Implemented

### Core Models
- **ImageClassifier**: ResNet50, EfficientNet-B0, MobileNetV3 with ImageNet labels
- **ObjectDetector**: FasterRCNN-MobileNet (torchvision) + mock backend for testing
- **ImageEmbedder**: Feature extraction (ResNet50/EfficientNet/MobileNet) with L2 normalization
- **SimilaritySearch**: ChromaDB-backed cosine similarity with HNSW indexing

### ONNX Pipeline
- PyTorch → ONNX export with dynamic batch axes
- Graph-level optimization (basic/extended/all)
- ONNXPredictor for CPU-optimized inference
- PyTorch vs ONNX latency benchmarking

### Processing Pipeline
- ImagePreprocessor: resize, normalize, augment, validate, from_bytes
- PostProcessor: decode classifications/detections, NMS, draw boxes, format output
- BatchProcessor: directory/URL/file-list processing with per-item error isolation

### Evaluation
- Classification: accuracy, top-5, per-class precision/recall/F1, confusion matrix
- Detection: mAP, mAP@50, mAP@75, per-class AP with 11-point interpolation
- Latency: mean, p50, p95, p99, throughput FPS
- Model comparison reports

### API (12 endpoints)
- Classification: single, batch, URL
- Detection: single, batch
- Embedding: single
- Similarity: index, search
- Operations: model listing, benchmark, health, metrics

### Infrastructure
- Multi-stage Dockerfile (CPU + GPU targets)
- Docker Compose with ChromaDB sidecar
- GitHub Actions CI/CD with Azure Container Apps deployment
- Prometheus metrics integration

## Known Limitations

1. **Detection models**: Real YOLO not included (would need ultralytics dependency); using torchvision FasterRCNN + mock mode instead
2. **ONNX export**: Only works for classifier models currently; detection models have complex output structures
3. **ChromaDB**: In-process mode; for production scale, use external ChromaDB server
4. **GPU**: Tested on CPU only; CUDA paths are configured but not production-validated

## Next Steps

1. Add ultralytics YOLOv8 integration for real-time detection
2. Implement model versioning with MLflow
3. Add A/B testing support in the model manager
4. WebSocket endpoint for real-time video processing
5. Redis caching layer for frequent queries
6. OpenTelemetry tracing integration
