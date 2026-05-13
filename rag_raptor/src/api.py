"""
FastAPI app do RAPTOR RAG.

Endpoints:
    POST /query   - recebe pergunta, retorna resposta + fontes + validacao numerica
    GET  /health  - verifica se o sistema esta pronto
"""
import os
import sys
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.logger import get_logger, setup_logging
from src.numerical_validator import validate_numbers
from src.query_interpreter import interpret_query
from src.startup import initialize

RAG_TYPE = "raptor"
RAG_LABEL = "RAPTOR RAG"

log = get_logger(__name__)

_engine = None
_interp_llm = None


def _cors_origins() -> list[str]:
    raw = os.getenv("RAG_CORS_ORIGINS", "*")
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins or ["*"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _interp_llm

    load_dotenv()
    setup_logging()

    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _engine, _interp_llm = initialize(base_dir)
    log.info("%s API pronta para receber requisicoes", RAG_LABEL)

    yield

    _engine = None
    _interp_llm = None
    log.info("%s API encerrada", RAG_LABEL)


app = FastAPI(
    title=RAG_LABEL,
    description="API do RAPTOR RAG para consultas sobre dados economicos de Sao Paulo.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    rag_type: str
    rag_label: str


@app.get("/")
async def root():
    return {"name": RAG_LABEL, "rag_type": RAG_TYPE}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engine_ready": _engine is not None,
        "rag_type": RAG_TYPE,
        "rag_label": RAG_LABEL,
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if _engine is None or _interp_llm is None:
        raise HTTPException(status_code=503, detail="Sistema ainda nao inicializado.")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="A pergunta nao pode ser vazia.")

    t0 = time.monotonic()
    log.info("Requisicao recebida", extra={"question": question[:120], "rag_type": RAG_TYPE})

    try:
        interp = interpret_query(question, _interp_llm)
        answer, source_nodes = await _engine.answer(
            question=question,
            sources=interp["sources"],
            rewritten_query=interp["rewritten_query"],
            is_labor_market=interp.get("is_labor_market", False),
        )

        checks = validate_numbers(answer, source_nodes)
        unverified = [c.value for c in checks if not c.verified]

        latency_ms = round((time.monotonic() - t0) * 1000)
        log.info(
            "Requisicao concluida",
            extra={
                "question": question[:120],
                "rag_type": RAG_TYPE,
                "sources": interp["sources"],
                "chunks": len(source_nodes),
                "latency_ms": latency_ms,
                "verified": f"{len(checks) - len(unverified)}/{len(checks)}",
            },
        )
    except Exception as exc:
        latency_ms = round((time.monotonic() - t0) * 1000)
        log.error(
            "Erro ao processar requisicao",
            extra={"question": question[:120], "rag_type": RAG_TYPE, "latency_ms": latency_ms},
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
        rag_type=RAG_TYPE,
        rag_label=RAG_LABEL,
    )
