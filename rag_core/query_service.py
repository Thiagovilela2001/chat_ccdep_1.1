"""Fluxo HTTP comum das engines, sem acoplar suas estratégias de recuperação."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .answer_policy import REFUSAL_TEXT, sanitize_answer
from .api_models import (
    CitationValidationInfo,
    NumericCitationInfo,
    QueryResponse,
    SourceInfo,
    ValidationInfo,
)
from .argument_validator import validate_arguments
from .citation_validator import validate_citations
from .conversation_memory import conversation_memory
from .logger import get_logger
from .numerical_validator import validate_numbers
from .metrics import record_estimated_usage
from .popup_explanations import generate_popup_explanations
from .provenance import relevance_score, source_file, source_page
from .runtime import request_timeout_seconds

_SOURCE_EXCERPT_MAX_CHARS = 4_000
_RECOVERY_SOURCES = ["text", "tables", "timeseries", "graph"]
log = get_logger(__name__)


def _source_excerpt(node) -> str:
    """Serializa o trecho recuperado sem deixar a resposta HTTP crescer sem limite."""
    content = str(node.get_content() or "").strip()
    if len(content) <= _SOURCE_EXCERPT_MAX_CHARS:
        return content
    marker = "\n\n[Trecho truncado]"
    return content[: _SOURCE_EXCERPT_MAX_CHARS - len(marker)].rstrip() + marker


@dataclass(frozen=True)
class QueryDiagnostics:
    sources: list[str]
    chunks: int
    verified: int
    total: int
    unverified: list[str]
    unsupported_arguments: list[str]
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float


@dataclass(frozen=True)
class _EngineAttempt:
    answer: str
    source_nodes: list
    checks: list
    verified: int
    unverified: list[str]
    unsupported_arguments: list[str]
    sources: list[str]
    rewritten_query: str


async def _run_engine_attempt(
    *,
    question: str,
    display_question: str,
    engine,
    sources: list[str],
    rewritten_query: str,
    is_labor_market: bool,
) -> _EngineAttempt:
    answer, source_nodes = await asyncio.wait_for(
        engine.answer(
            question=question,
            sources=sources,
            rewritten_query=rewritten_query,
            is_labor_market=is_labor_market,
        ),
        timeout=request_timeout_seconds(),
    )
    answer = sanitize_answer(answer, question=display_question)
    checks = await asyncio.to_thread(validate_numbers, answer, source_nodes)
    unverified = [check.value for check in checks if not check.verified]
    argument_checks = await asyncio.to_thread(validate_arguments, answer, source_nodes)
    unsupported_arguments = [
        check.sentence for check in argument_checks if not check.verified
    ]
    return _EngineAttempt(
        answer=answer,
        source_nodes=source_nodes,
        checks=checks,
        verified=len(checks) - len(unverified),
        unverified=unverified,
        unsupported_arguments=unsupported_arguments,
        sources=sources,
        rewritten_query=rewritten_query,
    )


async def execute_engine_query(
    *,
    question: str,
    engine,
    interp_llm,
    interpreter,
    rag_type: str,
    rag_label: str,
    conversation_id: str | None = None,
    history: list | None = None,
) -> tuple[QueryResponse, QueryDiagnostics]:
    """Interpreta, executa, valida e serializa uma consulta de engine."""
    conversation_id = conversation_id or conversation_memory.new_id()
    contextual_question, memory_turns = conversation_memory.contextualize(
        conversation_id, question, history or ()
    )
    interp = await asyncio.to_thread(interpreter, contextual_question, interp_llm)
    attempt = await _run_engine_attempt(
        question=contextual_question,
        display_question=question,
        engine=engine,
        sources=interp["sources"],
        rewritten_query=interp["rewritten_query"],
        is_labor_market=interp.get("is_labor_market", False),
    )
    if attempt.unverified or attempt.unsupported_arguments:
        log.warning(
            "Tentando recuperar lacunas de suporte com busca ampliada",
            extra={
                "question": question[:120],
                "unverified": attempt.unverified,
                "unsupported_arguments": attempt.unsupported_arguments[:3],
            },
        )
        recovery = await _run_engine_attempt(
            question=contextual_question,
            display_question=question,
            engine=engine,
            sources=_RECOVERY_SOURCES,
            rewritten_query=interp["rewritten_query"],
            is_labor_market=interp.get("is_labor_market", False),
        )
        if (
            not recovery.unverified
            and not recovery.unsupported_arguments
            and (recovery.checks or attempt.unsupported_arguments)
        ) or (
            recovery.verified > attempt.verified
            and len(recovery.unverified) <= len(attempt.unverified)
            and len(recovery.unsupported_arguments) <= len(attempt.unsupported_arguments)
        ):
            attempt = recovery

    answer = attempt.answer
    source_nodes = attempt.source_nodes
    checks = attempt.checks
    unverified = attempt.unverified
    verified = attempt.verified
    unsupported_arguments = attempt.unsupported_arguments
    answer_blocked_by_guardrail = bool(unverified or unsupported_arguments)
    if answer_blocked_by_guardrail:
        log.warning(
            "Resposta bloqueada por falta de suporte documental",
            extra={
                "question": question[:120],
                "unverified": unverified,
                "unsupported_arguments": unsupported_arguments[:3],
            },
        )
        answer = REFUSAL_TEXT
    citation_checks = await asyncio.to_thread(validate_citations, answer, source_nodes)
    unverified_citations = [
        check.citation for check in citation_checks if not check.verified
    ]
    usage = record_estimated_usage(
        rag_type,
        question + "\n" + "\n".join(node.get_content() for node in source_nodes),
        answer,
    )
    numeric_citations = [] if answer_blocked_by_guardrail else [
        NumericCitationInfo(
            value=check.value,
            start=check.response_start,
            end=check.response_end,
            source_index=check.source_index,
            file=source_file(source_nodes[check.source_index]),
            score=relevance_score(source_nodes[check.source_index]),
            page=source_page(source_nodes[check.source_index]),
            snippet=check.source_snippet or "",
            content_type=str(
                (
                    getattr(source_nodes[check.source_index], "metadata", {})
                    or {}
                ).get("type")
                or "text"
            ),
            claim=check.response_snippet or "",
        )
        for check in checks
        if (
            check.verified
            and not check.derived
            and check.response_start is not None
            and check.response_end is not None
            and check.source_index is not None
        )
    ]
    if numeric_citations:
        try:
            explanations = await generate_popup_explanations(numeric_citations)
        except Exception as exc:
            log.warning("Falha inesperada nas explicacoes de popup; usando fallback: %s", exc)
            explanations = {}
    else:
        explanations = {}
    for index, explanation in explanations.items():
        if 0 <= index < len(numeric_citations):
            numeric_citations[index].explanation = explanation

    response = QueryResponse(
        answer=answer,
        sources_used=attempt.sources,
        rewritten_query=attempt.rewritten_query,
        sources=[
            SourceInfo(
                file=source_file(node),
                score=relevance_score(node),
                page=source_page(node),
                excerpt=_source_excerpt(node),
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
        numeric_citations=numeric_citations,
        rag_type=rag_type,
        rag_label=rag_label,
        conversation_id=conversation_id,
        memory_turns=memory_turns,
    )
    conversation_memory.remember(conversation_id, question, answer)
    diagnostics = QueryDiagnostics(
        sources=attempt.sources,
        chunks=len(source_nodes),
        verified=verified,
        total=len(checks),
        unverified=unverified,
        unsupported_arguments=unsupported_arguments,
        estimated_input_tokens=usage["estimated_input_tokens"],
        estimated_output_tokens=usage["estimated_output_tokens"],
        estimated_cost_usd=usage["estimated_cost_usd"],
    )
    return response, diagnostics
