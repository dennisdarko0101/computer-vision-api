# Deployment Guide

## Local Development

```bash
pip install -e ".[dev]"
make run
# API available at http://localhost:8000/docs
```

## Docker

### CPU Build
```bash
docker build -f docker/Dockerfile --target cpu -t cv-api:latest .
docker run -p 8000:8000 cv-api:latest
```

### GPU Build
```bash
docker build -f docker/Dockerfile --target gpu -t cv-api:gpu .
docker run --gpus all -p 8000:8000 cv-api:gpu
```

### Docker Compose (API + ChromaDB)
```bash
docker compose -f docker/docker-compose.yml up -d
```

## Azure Deployment

### Azure Container Apps

1. **Create resource group and environment:**
```bash
az group create --name cv-api-rg --location eastus
az containerapp env create --name cv-api-env --resource-group cv-api-rg --location eastus
```

2. **Deploy from container registry:**
```bash
az containerapp create \
  --name computer-vision-api \
  --resource-group cv-api-rg \
  --environment cv-api-env \
  --image ghcr.io/username/computer-vision-api:latest \
  --target-port 8000 \
  --ingress external \
  --cpu 2 --memory 4Gi \
  --env-vars CV_DEVICE=cpu CV_ONNX_ENABLED=true CV_LOG_LEVEL=INFO
```

3. **Enable auto-scaling:**
```bash
az containerapp update \
  --name computer-vision-api \
  --resource-group cv-api-rg \
  --min-replicas 1 \
  --max-replicas 10 \
  --scale-rule-name http-rule \
  --scale-rule-type http \
  --scale-rule-http-concurrency 50
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CV_DEVICE` | `cpu` | `cpu` or `cuda` |
| `CV_BATCH_SIZE` | `16` | Max batch size |
| `CV_ONNX_ENABLED` | `false` | Use ONNX backend |
| `CV_MAX_LOADED_MODELS` | `5` | LRU cache size |
| `CV_LOG_LEVEL` | `INFO` | Logging level |
| `CV_CONFIDENCE_THRESHOLD` | `0.5` | Detection threshold |

## Production Checklist

- [ ] Set `CV_ONNX_ENABLED=true` for CPU deployments
- [ ] Configure health check probes
- [ ] Set resource limits (memory, CPU)
- [ ] Enable structured JSON logging
- [ ] Configure CORS origins (restrict from `*`)
- [ ] Mount persistent volumes for model cache and ChromaDB
- [ ] Set up monitoring with Prometheus `/metrics` endpoint
