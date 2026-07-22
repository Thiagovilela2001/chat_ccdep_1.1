FROM python:3.11-slim

ARG APP_UID=1000
ARG APP_GID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ghostscript \
        libglib2.0-0 \
        libgl1 \
        poppler-utils \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

RUN groupadd --gid "${APP_GID}" appuser \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home appuser \
    && mkdir -p /cache/huggingface \
    && chown -R appuser:appuser /app /cache/huggingface

COPY --chown=appuser:appuser rag_core/ ./rag_core/
COPY --chown=appuser:appuser rag_principal/ ./rag_principal/
COPY --chown=appuser:appuser rag_agentic/ ./rag_agentic/
COPY --chown=appuser:appuser rag_raptor/ ./rag_raptor/
COPY --chown=appuser:appuser rag_selfrag/ ./rag_selfrag/
COPY --chown=appuser:appuser rag_orchestrator/ ./rag_orchestrator/
COPY --chown=appuser:appuser .agents/ ./rag_principal/.agents/
COPY --chown=appuser:appuser .agents/ ./rag_agentic/.agents/
COPY --chown=appuser:appuser .agents/ ./rag_raptor/.agents/
COPY --chown=appuser:appuser .agents/ ./rag_selfrag/.agents/

USER appuser

WORKDIR /app/rag_principal

EXPOSE 8000

CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8000"]
