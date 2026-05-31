# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Where the embedding model is baked at build time and loaded from offline.
ENV EMBED_CACHE_DIR=/models \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the bge-small model into the image so runtime needs NO network.
# A tiny encode materialises the ONNX weights into EMBED_CACHE_DIR.
RUN python -c "from fastembed import TextEmbedding; \
m = TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir='/models'); \
list(m.embed(['warmup']))"

COPY src ./src

# Defence in depth: force offline mode for HF/transformers libs at runtime.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    EMBED_BACKEND=fastembed

EXPOSE 8080

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
