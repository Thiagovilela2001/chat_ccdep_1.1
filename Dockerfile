FROM python:3.11-slim

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

COPY rag_core/ ./rag_core/
COPY rag_principal/ ./rag_principal/
COPY rag_agentic/ ./rag_agentic/
COPY rag_raptor/ ./rag_raptor/
COPY rag_selfrag/ ./rag_selfrag/
COPY rag_orchestrator/ ./rag_orchestrator/
COPY meta_rag_ui/ ./meta_rag_ui/
COPY frontend/ ./frontend/
COPY .agents/ ./rag_principal/.agents/
COPY .agents/ ./rag_agentic/.agents/
COPY .agents/ ./rag_raptor/.agents/
COPY .agents/ ./rag_selfrag/.agents/

WORKDIR /app/rag_principal

EXPOSE 8000

CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8000"]
