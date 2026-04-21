# Runtime image for the MindWise FastAPI app.
#
# Deliberately slim: torch / transformers / peft / bitsandbytes live only
# in the [finetune] extras and only matter at training time. Inference
# goes through Ollama (separate container), so the runtime doesn't need
# GPU libraries. The one heavy dep that stays is sentence-transformers,
# which pulls torch CPU — required for the bge embedding model we use
# for Chroma. Could be shrunk further by switching to a Chroma-side
# embedding function, but keep it simple for now.

FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# system libs required by opencv (used for face_emotion), mediapipe, lxml
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
# Install only the runtime deps (not [finetune]) — keeps image ~2GB instead of ~8GB.
RUN pip install --upgrade pip && \
    pip install \
        "fastapi>=0.109" "uvicorn[standard]>=0.27" \
        "pydantic>=2.5" "pydantic-settings>=2.1" \
        "python-dotenv>=1.0" "httpx>=0.26" "python-multipart>=0.0.9" \
        "aiosmtplib>=3.0" "openpyxl>=3.1" \
        "ollama>=0.1.7" "tenacity>=8.2" \
        "chromadb>=0.4.22" "langchain-text-splitters>=0.0.1" \
        "tiktoken>=0.5" "langgraph>=0.0.40" \
        "faster-whisper>=1.0" "mediapipe>=0.10" \
        "opencv-python-headless>=4.9" "numpy>=1.26" \
        "sentence-transformers>=2.2" \
        "mcp>=1.0" \
        "prometheus-client>=0.19" "prometheus-fastapi-instrumentator>=7.0" \
        "langfuse>=2.0" \
        "zhconv>=1.4"

COPY app/ ./app/
COPY mcp_server/ ./mcp_server/
COPY data/kb_docs/ ./data/kb_docs/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
