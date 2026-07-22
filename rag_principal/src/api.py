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
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

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
from .startup import graph_enabled_by_env, initialize

RAG_TYPE = "principal"
RAG_LABEL = "RAG Principal"

log = get_logger(__name__)

# ── Estado global da aplicação ────────────────────────────────────────────────

_engine = None
_interp_llm = None


def _frontend_dir() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")
    )


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _interp_llm

    setup_logging()

    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _engine, _interp_llm = initialize(base_dir, use_graph=graph_enabled_by_env())
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(MetricsMiddleware, service_name=RAG_TYPE)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    if os.path.isdir(_frontend_dir()):
        return RedirectResponse(url="/app/")
    return RedirectResponse(url="/docs")


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
                "sources": diagnostics.sources,
                "chunks": diagnostics.chunks,
                "latency_ms": latency_ms,
                "verified": f"{diagnostics.verified}/{diagnostics.total}",
                "estimated_input_tokens": diagnostics.estimated_input_tokens,
                "estimated_output_tokens": diagnostics.estimated_output_tokens,
                "estimated_cost_usd": diagnostics.estimated_cost_usd,
            },
        )
        if diagnostics.unverified:
            log.warning(
                "Numeros nao verificados na resposta",
                extra={"question": question[:120], "unverified": diagnostics.unverified},
            )

    except TimeoutError as exc:
        log.warning("Timeout global ao processar requisicao", extra={"question": question[:120]})
        raise HTTPException(status_code=504, detail="Tempo limite da requisição excedido.") from exc
    except Exception as exc:
        latency_ms = round((time.monotonic() - t0) * 1000)
        log.error(
            "Erro ao processar requisicao",
            extra={"question": question[:120], "latency_ms": latency_ms},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Falha interna ao processar a consulta."
        ) from exc

    return response


_frontend_path = _frontend_dir()
if os.path.isdir(_frontend_path):
    app.mount(
        "/app",
        StaticFiles(directory=_frontend_path, html=True),
        name="frontend",
    )
