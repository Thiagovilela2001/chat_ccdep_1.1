"""
API — FastAPI app do RAG Estatístico SP.

Endpoints:
    POST /query   — recebe pergunta, retorna resposta + fontes + validação numérica
    GET  /health  — verifica se o sistema está pronto

Inicialização (lifespan):
    Detecta mudanças em /data, reindexada se necessário, monta todos os
    retrievers e o AnalysisEngine antes de aceitar requisições.
"""
import os
import sys
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from src.logger import get_logger, setup_logging
from src.query_interpreter import interpret_query
from src.numerical_validator import validate_numbers
from src.startup import initialize

log = get_logger(__name__)

# ── Estado global da aplicação ────────────────────────────────────────────────

_engine = None
_interp_llm = None


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _interp_llm

    load_dotenv()
    setup_logging()

    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _engine, _interp_llm = initialize(base_dir)
    log.info("API pronta para receber requisicoes")

    yield  # API ativa

    _engine = None
    _interp_llm = None
    log.info("API encerrada")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Estatístico SP",
    description="Sistema de perguntas e respostas sobre dados econômicos do Estado de São Paulo.",
    version="2.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str


class SourceInfo(BaseModel):
    file: str
    score: float


class ValidationInfo(BaseModel):
    verified: int
    total: int
    unverified: list[str]


class QueryResponse(BaseModel):
    answer: str
    sources_used: list[str]
    rewritten_query: str
    sources: list[SourceInfo]
    validation: ValidationInfo


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "engine_ready": _engine is not None}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if _engine is None or _interp_llm is None:
        raise HTTPException(status_code=503, detail="Sistema ainda não inicializado.")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="A pergunta não pode ser vazia.")

    t0 = time.monotonic()
    log.info(
        "Requisicao recebida",
        extra={"question": question[:120]},
    )

    try:
        # 1. Query Interpreter — decide fontes e reescreve query
        interp = interpret_query(question, _interp_llm)

        # 2. Analysis Engine — retrievers paralelos + síntese LLM
        answer, source_nodes = await _engine.answer(
            question=question,
            sources=interp["sources"],
            rewritten_query=interp["rewritten_query"],
            is_labor_market=interp.get("is_labor_market", False),
        )

        # 3. Numeric Validator
        checks = validate_numbers(answer, source_nodes)
        unverified = [c.value for c in checks if not c.verified]

        latency_ms = round((time.monotonic() - t0) * 1000)
        log.info(
            "Requisicao concluida",
            extra={
                "question": question[:120],
                "sources": interp["sources"],
                "chunks": len(source_nodes),
                "latency_ms": latency_ms,
                "verified": f"{len(checks) - len(unverified)}/{len(checks)}",
            },
        )
        if unverified:
            log.warning(
                "Numeros nao verificados na resposta",
                extra={"question": question[:120], "unverified": unverified},
            )

    except Exception as exc:
        latency_ms = round((time.monotonic() - t0) * 1000)
        log.error(
            "Erro ao processar requisicao",
            extra={"question": question[:120], "latency_ms": latency_ms},
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return QueryResponse(
        answer=answer,
        sources_used=interp["sources"],
        rewritten_query=interp["rewritten_query"],
        sources=[
            SourceInfo(
                file=n.metadata.get("source_file") or n.metadata.get("file_name", "?"),
                score=round((n.score or 0) / 10.0, 2),
            )
            for n in source_nodes
        ],
        validation=ValidationInfo(
            verified=len(checks) - len(unverified),
            total=len(checks),
            unverified=unverified,
        ),
    )
