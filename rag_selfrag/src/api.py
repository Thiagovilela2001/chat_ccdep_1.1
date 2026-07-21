"""
FastAPI app do Self-RAG.

Endpoints:
    POST /query   - recebe pergunta, retorna resposta + fontes + validacao numerica
    GET  /health  - verifica se o sistema esta pronto
"""
import os
import sys
import time
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
from rag_core.logger import get_logger, setup_logging
from rag_core.api_models import QueryRequest, QueryResponse
from rag_core.query_service import execute_engine_query
from rag_core.metrics import MetricsMiddleware, render_prometheus
from .query_interpreter import interpret_query
from .startup import initialize

RAG_TYPE = "selfrag"
RAG_LABEL = "Self-RAG"

log = get_logger(__name__)

_engine = None
_interp_llm = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _interp_llm

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
    description="API do Self-RAG para consultas sobre dados economicos de Sao Paulo.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(MetricsMiddleware, service_name=RAG_TYPE)


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


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return render_prometheus()


@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    _rl: None = Depends(enforce_rate_limit),
    _auth: None = Depends(require_api_key),
):
    if _engine is None or _interp_llm is None:
        raise HTTPException(status_code=503, detail="Sistema ainda nao inicializado.")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="A pergunta nao pode ser vazia.")

    t0 = time.monotonic()
    log.info("Requisicao recebida", extra={"question": question[:120], "rag_type": RAG_TYPE})

    try:
        response, diagnostics = await execute_engine_query(
            question=question,
            engine=_engine,
            interp_llm=_interp_llm,
            interpreter=interpret_query,
            rag_type=RAG_TYPE,
            rag_label=RAG_LABEL,
        )
        latency_ms = round((time.monotonic() - t0) * 1000)
        log.info(
            "Requisicao concluida",
            extra={
                "question": question[:120],
                "rag_type": RAG_TYPE,
                "sources": diagnostics.sources,
                "chunks": diagnostics.chunks,
                "latency_ms": latency_ms,
                "verified": f"{diagnostics.verified}/{diagnostics.total}",
                "estimated_input_tokens": diagnostics.estimated_input_tokens,
                "estimated_output_tokens": diagnostics.estimated_output_tokens,
                "estimated_cost_usd": diagnostics.estimated_cost_usd,
            },
        )
    except TimeoutError as exc:
        log.warning("Timeout global ao processar requisicao", extra={"question": question[:120]})
        raise HTTPException(status_code=504, detail="Tempo limite da requisição excedido.") from exc
    except Exception as exc:
        latency_ms = round((time.monotonic() - t0) * 1000)
        log.error(
            "Erro ao processar requisicao",
            extra={"question": question[:120], "rag_type": RAG_TYPE, "latency_ms": latency_ms},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Falha interna ao processar a consulta."
        ) from exc

    return response
