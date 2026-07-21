"""
api.py — Ponto de entrada único do Meta RAG (FastAPI).

Endpoints:
    POST /query   — analisa a pergunta, escolhe a estratégia e devolve a resposta
                    da engine selecionada + metadados de rota/qualidade.
    GET  /route   — apenas a decisão de roteamento (sem executar engine).
    GET  /health  — estado do orquestrador e disponibilidade dos backends.
"""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

load_dotenv()

from rag_core.api_security import (
    SecurityHeadersMiddleware,
    cors_origins,
    enforce_rate_limit,
    require_api_key,
)
from rag_core.api_models import QueryRequest
from rag_core.runtime import request_timeout_seconds
from rag_core.metrics import MetricsMiddleware, render_prometheus
from .orchestrator import Orchestrator
from .registry import get_client, get_profiles, health_is_ready

RAG_TYPE = "meta"
RAG_LABEL = "Meta RAG (orquestrador)"

_orchestrator: Orchestrator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator
    multi = os.getenv("ORCHESTRATOR_MULTI_ENGINE", "0") in ("1", "true", "True")
    _orchestrator = Orchestrator(
        multi_engine=multi,
        timeout=int(request_timeout_seconds()),
    )
    yield
    _orchestrator = None


app = FastAPI(title="Meta RAG — Orquestrador", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=cors_origins(), allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(MetricsMiddleware, service_name=RAG_TYPE)


@app.get("/health")
async def health():
    async def _one(key):
        h = await get_client(key).health()
        return key, {"up": health_is_ready(h), "detail": h}
    pairs = await asyncio.gather(*(_one(k) for k in get_profiles()))
    backends = dict(pairs)
    return {
        "status": "ok" if any(item["up"] for item in backends.values()) else "degraded",
        "rag_type": RAG_TYPE,
        "rag_label": RAG_LABEL,
        "orchestrator_ready": _orchestrator is not None,
        "backends": backends,
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return render_prometheus()


@app.post("/query")
async def query(
    request: QueryRequest,
    _rl: None = Depends(enforce_rate_limit),
    _auth: None = Depends(require_api_key),
):
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Orquestrador não inicializado.")
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="A pergunta não pode ser vazia.")

    try:
        result = await asyncio.wait_for(
            _orchestrator.answer(question), timeout=request_timeout_seconds()
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Tempo limite da requisição excedido.") from exc
    if result.get("error"):
        raise HTTPException(
            status_code=502,
            detail="Nenhuma engine disponível conseguiu processar a consulta.",
        )
    result.setdefault("rag_type", RAG_TYPE)
    result.setdefault("rag_label", RAG_LABEL)
    return result


@app.post("/route")
async def route_only(
    request: QueryRequest,
    _rl: None = Depends(enforce_rate_limit),
    _auth: None = Depends(require_api_key),
):
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Orquestrador não inicializado.")
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="A pergunta não pode ser vazia.")
    try:
        return await asyncio.wait_for(
            _orchestrator.route_only(question), timeout=request_timeout_seconds()
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Tempo limite da requisição excedido.") from exc
