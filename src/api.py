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
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.query_interpreter import interpret_query
from src.numerical_validator import validate_numbers, format_validation_report
from src.startup import initialize

# ── Estado global da aplicação ────────────────────────────────────────────────

_engine = None
_interp_llm = None


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _interp_llm

    load_dotenv()

    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _engine, _interp_llm = initialize(base_dir)

    yield  # API ativa

    # Cleanup (se necessário no futuro)
    _engine = None
    _interp_llm = None


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
    sources_used: list[str]       # ex: ["text", "tables"]
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
            verified=sum(1 for c in checks if c.verified),
            total=len(checks),
            unverified=[c.value for c in checks if not c.verified],
        ),
    )
