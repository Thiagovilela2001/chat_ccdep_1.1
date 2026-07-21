"""Fluxo HTTP comum das engines, sem acoplar suas estratégias de recuperação."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .api_models import CitationValidationInfo, QueryResponse, SourceInfo, ValidationInfo
from .citation_validator import validate_citations
from .numerical_validator import validate_numbers
from .metrics import record_estimated_usage
from .provenance import relevance_score, source_file, source_page
from .runtime import request_timeout_seconds


@dataclass(frozen=True)
class QueryDiagnostics:
    sources: list[str]
    chunks: int
    verified: int
    total: int
    unverified: list[str]
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float


async def execute_engine_query(
    *,
    question: str,
    engine,
    interp_llm,
    interpreter,
    rag_type: str,
    rag_label: str,
) -> tuple[QueryResponse, QueryDiagnostics]:
    """Interpreta, executa, valida e serializa uma consulta de engine."""
    interp = await asyncio.to_thread(interpreter, question, interp_llm)
    answer, source_nodes = await asyncio.wait_for(
        engine.answer(
            question=question,
            sources=interp["sources"],
            rewritten_query=interp["rewritten_query"],
            is_labor_market=interp.get("is_labor_market", False),
        ),
        timeout=request_timeout_seconds(),
    )
    checks = await asyncio.to_thread(validate_numbers, answer, source_nodes)
    unverified = [check.value for check in checks if not check.verified]
    verified = len(checks) - len(unverified)
    citation_checks = await asyncio.to_thread(validate_citations, answer, source_nodes)
    unverified_citations = [
        check.citation for check in citation_checks if not check.verified
    ]
    usage = record_estimated_usage(
        rag_type,
        question + "\n" + "\n".join(node.get_content() for node in source_nodes),
        answer,
    )

    response = QueryResponse(
        answer=answer,
        sources_used=interp["sources"],
        rewritten_query=interp["rewritten_query"],
        sources=[
            SourceInfo(
                file=source_file(node),
                score=relevance_score(node),
                page=source_page(node),
            )
            for node in source_nodes
        ],
        validation=ValidationInfo(
            verified=verified,
            total=len(checks),
            unverified=unverified,
        ),
        citation_validation=CitationValidationInfo(
            verified=len(citation_checks) - len(unverified_citations),
            total=len(citation_checks),
            unverified=unverified_citations,
        ),
        rag_type=rag_type,
        rag_label=rag_label,
    )
    diagnostics = QueryDiagnostics(
        sources=interp["sources"],
        chunks=len(source_nodes),
        verified=verified,
        total=len(checks),
        unverified=unverified,
        estimated_input_tokens=usage["estimated_input_tokens"],
        estimated_output_tokens=usage["estimated_output_tokens"],
        estimated_cost_usd=usage["estimated_cost_usd"],
    )
    return response, diagnostics
