# ── Stage 1: Builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir --timeout=300 --retries=10 --upgrade pip && \
    pip install --no-cache-dir --timeout=300 --retries=10 . --target=/app/deps

# ── Stage 2: CPU Runtime ───────────────────────────────────────────────────
FROM python:3.11-slim AS cpu

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/deps /usr/local/lib/python3.11/site-packages
COPY src/ /app/src/
COPY pyproject.toml /app/

ENV PYTHONPATH=/app
ENV CV_DEVICE=cpu
ENV CV_API_HOST=0.0.0.0
ENV CV_API_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ── Stage 3: GPU Runtime (build with --target gpu) ─────────────────────────
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS gpu

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/deps /usr/local/lib/python3.11/site-packages
COPY src/ /app/src/
COPY pyproject.toml /app/

# Install GPU-specific packages
RUN pip install --no-cache-dir --timeout=300 --retries=10 onnxruntime-gpu

ENV PYTHONPATH=/app
ENV CV_DEVICE=cuda
ENV CV_API_HOST=0.0.0.0
ENV CV_API_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["python3", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
