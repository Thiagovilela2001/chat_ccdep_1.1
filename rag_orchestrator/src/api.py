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
from pydantic import BaseModel

from rag_core.api_security import enforce_rate_limit, require_api_key
from src.orchestrator import Orchestrator
from src.registry import get_client, get_profiles

RAG_TYPE = "meta"
RAG_LABEL = "Meta RAG (orquestrador)"

_orchestrator: Orchestrator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator
    load_dotenv()
    multi = os.getenv("ORCHESTRATOR_MULTI_ENGINE", "0") in ("1", "true", "True")
    _orchestrator = Orchestrator(multi_engine=multi)
    yield
    _orchestrator = None


app = FastAPI(title="Meta RAG — Orquestrador", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


@app.get("/health")
async def health():
    async def _one(key):
        h = await get_client(key).health()
        return key, {"up": h is not None, "detail": h}
    pairs = await asyncio.gather(*(_one(k) for k in get_profiles()))
    return {
        "status": "ok",
        "rag_type": RAG_TYPE,
        "rag_label": RAG_LABEL,
        "orchestrator_ready": _orchestrator is not None,
        "backends": dict(pairs),
    }


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

    result = await _orchestrator.answer(question)
    if result.get("error"):
        raise HTTPException(
            status_code=502,
            detail=f"Engine '{result.get('route', {}).get('engine')}' indisponível: {result['error']}",
        )
    result.setdefault("rag_type", RAG_TYPE)
    result.setdefault("rag_label", RAG_LABEL)
    return result


@app.post("/route")
async def route_only(request: QueryRequest):
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Orquestrador não inicializado.")
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="A pergunta não pode ser vazia.")
    return await _orchestrator.route_only(question)
