"""Fluxo HTTP comum das engines, sem acoplar suas estratégias de recuperação."""
from __future__ import annotations

import asyncio
import re
import unicodedata
from dataclasses import dataclass, replace

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
_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]|$)")
_COMPLETE_COVERAGE_RE = re.compile(
    r"\b(?:quais|compare|comparar|evolu[cç][aã]o|dinamismo|melhoraram|"
    r"pioraram|ao longo|entre os per[ií]odos|s[eé]rie hist[oó]rica)\b.*"
    r"\b(?:regi[oõ]es?|setores?|indicadores?|per[ií]odos?|trimestres?|"
    r"meses?|anos?)\b|"
    r"\b(?:regi[oõ]es?|setores?|indicadores?)\b.*"
    r"\b(?:dinamismo|evolu[cç][aã]o|melhoraram|pioraram|per[ií]odos?)\b",
    re.IGNORECASE,
)
_COMPLETE_COVERAGE_INSTRUCTION = """\


COBERTURA TEMPORAL OBRIGATÓRIA:
Analise todo o intervalo disponível nas evidências, do primeiro ao último
período. A resposta deve identificar os períodos completos, as regiões, setores
ou indicadores relevantes e os respectivos valores disponíveis. Quando os
destaques mudarem, organize os resultados por período. Não responda apenas com
uma contagem agregada e não pare no primeiro ou no último recorte encontrado.
"""
_COMPLETE_RETRIEVAL_SUFFIX = (
    " cobertura completa de todos os periodos disponiveis, do periodo inicial "
    "ao final, com resultados por periodo, regiao e indicador"
)
_SUPPORTED_NUMERIC_RETRY_INSTRUCTION = """\


ORIENTAÇÃO OBRIGATÓRIA PARA ESTA NOVA SÍNTESE:
A tentativa anterior continha afirmações ou valores que não puderam ser
integralmente validados. Responda novamente usando apenas fatos explicitamente
sustentados no contexto. A resposta DEVE apresentar resultados numéricos sempre
que houver qualquer valor documental seguro. Copie cada valor exatamente como
aparece nas evidências, com sua unidade, indicador, território e período. Remova
somente os valores sem suporte; nunca elimine um valor validado apenas porque outro
falhou. Preserve também nomes, tendências e comparações sustentadas. Não mencione
validação, recuperação, contexto ou esta orientação.

Valores que já foram validados na tentativa anterior e devem ser reaproveitados
quando forem pertinentes à pergunta: {verified_values}
"""
log = get_logger(__name__)


def _requires_complete_coverage(question: str) -> bool:
    return bool(_COMPLETE_COVERAGE_RE.search(question or ""))


def _coverage_score(attempt: "_EngineAttempt") -> tuple[int, int, int]:
    sentences = len([
        match for match in _SENTENCE_RE.finditer(attempt.answer or "")
        if match.group(0).strip()
    ])
    return sentences, len(attempt.checks), len(attempt.answer or "")


def _looks_incomplete(attempt: "_EngineAttempt") -> bool:
    sentences, numbers, length = _coverage_score(attempt)
    return sentences < 2 or (numbers < 4 and length < 500)


def _source_excerpt(node) -> str:
    """Serializa o trecho recuperado sem deixar a resposta HTTP crescer sem limite."""
    content = str(node.get_content() or "").strip()
    if len(content) <= _SOURCE_EXCERPT_MAX_CHARS:
        return content
    marker = "\n\n[Trecho truncado]"
    return content[: _SOURCE_EXCERPT_MAX_CHARS - len(marker)].rstrip() + marker


def _trim_unsupported_answer(
    answer: str,
    *,
    checks: list,
    unsupported_arguments: list[str],
) -> str:
    """Preserva frases com números, mesmo quando a validação falha."""
    if not answer:
        return ""

    unsupported = {sentence.strip() for sentence in unsupported_arguments}
    kept: list[str] = []
    for match in _SENTENCE_RE.finditer(answer):
        sentence = match.group(0).strip()
        if not sentence:
            continue
        has_number = any(
            check.response_start is not None
            and check.response_end is not None
            and check.response_start < match.end()
            and check.response_end > match.start()
            for check in checks
        )
        if sentence in unsupported and not has_number:
            continue
        kept.append(sentence)

    if not kept:
        return ""
    return sanitize_answer(" ".join(kept))


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
    argument_checks: list
    sources: list[str]
    rewritten_query: str


def _is_fully_supported(attempt: _EngineAttempt) -> bool:
    """Aceita respostas factuais somente quando todo o texto final tem suporte."""
    return bool(
        attempt.answer != REFUSAL_TEXT
        and not attempt.unverified
        and not attempt.unsupported_arguments
    )


_CAPITALIZED_TOKEN_RE = re.compile(r"\b[A-ZÀ-Ý][A-Za-zÀ-ÿ]{2,}\b")
_GENERIC_CAPITALIZED = {
    "alem", "como", "contudo", "estado", "mercado", "regiao", "regioes",
    "nesse", "neste", "portanto", "trabalho",
}


def _fold_token(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value or "")
    return "".join(ch for ch in folded if not unicodedata.combining(ch)).lower()


def _is_usable_qualitative(attempt: _EngineAttempt) -> bool:
    """Aceita paráfrase sem números quando seus nomes constam nas evidências."""
    if (
        attempt.answer == REFUSAL_TEXT
        or attempt.checks
        or not attempt.source_nodes
    ):
        return False

    source_text = _fold_token("\n".join(
        str(node.get_content() or "") for node in attempt.source_nodes
    ))
    named_tokens = {
        _fold_token(token)
        for token in _CAPITALIZED_TOKEN_RE.findall(attempt.answer)
        if _fold_token(token) not in _GENERIC_CAPITALIZED
    }
    if named_tokens:
        return all(re.search(rf"\b{re.escape(token)}\b", source_text) for token in named_tokens)

    return bool(attempt.argument_checks) and all(
        check.verified or check.support_ratio >= 0.5
        for check in attempt.argument_checks
    )


def _number_free_subset(attempt: _EngineAttempt) -> _EngineAttempt | None:
    """Recupera conclusões qualitativas já presentes sem reter nenhum algarismo."""
    sentences = [
        match.group(0).strip()
        for match in _SENTENCE_RE.finditer(attempt.answer or "")
        if match.group(0).strip() and not re.search(r"\d", match.group(0))
    ]
    if not sentences:
        return None
    answer = sanitize_answer(" ".join(sentences))
    if answer == REFUSAL_TEXT:
        return None
    argument_checks = validate_arguments(answer, attempt.source_nodes)
    unsupported_arguments = [
        check.sentence for check in argument_checks if not check.verified
    ]
    return replace(
        attempt,
        answer=answer,
        checks=[],
        verified=0,
        unverified=[],
        unsupported_arguments=unsupported_arguments,
        argument_checks=argument_checks,
    )


def _verified_value_list(attempt: _EngineAttempt) -> list[str]:
    """Mantém ordem e formatação dos valores já comprovados."""
    values = []
    for check in attempt.checks:
        if check.verified and check.value not in values:
            values.append(check.value)
    return values


def _needs_qualitative_retry(attempt: _EngineAttempt) -> bool:
    """Repete somente recusas ou respostas sem qualquer afirmação numérica."""
    return bool(
        attempt.answer == REFUSAL_TEXT
        or (attempt.unsupported_arguments and not attempt.checks)
    )


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
        argument_checks=argument_checks,
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
    complete_coverage = _requires_complete_coverage(question)
    engine_question = contextual_question
    selected_sources = interp["sources"]
    rewritten_query = interp["rewritten_query"]
    if complete_coverage:
        engine_question += _COMPLETE_COVERAGE_INSTRUCTION
        selected_sources = list(_RECOVERY_SOURCES)
        if _COMPLETE_RETRIEVAL_SUFFIX.strip() not in rewritten_query:
            rewritten_query += _COMPLETE_RETRIEVAL_SUFFIX
    attempt = await _run_engine_attempt(
        question=engine_question,
        display_question=question,
        engine=engine,
        sources=selected_sources,
        rewritten_query=rewritten_query,
        is_labor_market=interp.get("is_labor_market", False),
    )
    needs_recovery = bool(
        attempt.unverified
        or attempt.unsupported_arguments
        or (attempt.answer == REFUSAL_TEXT and attempt.source_nodes)
        or (complete_coverage and _looks_incomplete(attempt))
    )
    if needs_recovery:
        log.warning(
            "Tentando recuperar lacunas de suporte com busca ampliada",
            extra={
                "question": question[:120],
                "unverified": attempt.unverified,
                "unsupported_arguments": attempt.unsupported_arguments[:3],
            },
        )
        recovery_question = engine_question
        if complete_coverage and _looks_incomplete(attempt):
            recovery_question += (
                "\nA resposta anterior ficou incompleta. Amplie a análise com "
                "todos os períodos e resultados numéricos recuperados."
            )
        recovery = await _run_engine_attempt(
            question=recovery_question,
            display_question=question,
            engine=engine,
            sources=_RECOVERY_SOURCES,
            rewritten_query=rewritten_query,
            is_labor_market=interp.get("is_labor_market", False),
        )
        recovery_is_preferred = (
            not recovery.unverified
            and not recovery.unsupported_arguments
            and (recovery.checks or attempt.unsupported_arguments)
        ) or (
            recovery.verified > attempt.verified
            and len(recovery.unverified) <= len(attempt.unverified)
            and len(recovery.unsupported_arguments) <= len(attempt.unsupported_arguments)
        )
        if complete_coverage and recovery_is_preferred:
            recovery_is_preferred = _coverage_score(recovery) >= _coverage_score(attempt)
        if recovery_is_preferred:
            attempt = recovery

    # Recusas ou respostas sem afirmações numéricas ainda recebem nova síntese.
    if _needs_qualitative_retry(attempt):
        pre_qualitative_attempt = attempt
        log.warning(
            "Tentando nova resposta com os valores numericos validados",
            extra={"question": question[:120]},
        )
        verified_values = _verified_value_list(pre_qualitative_attempt)
        retry_instruction = _SUPPORTED_NUMERIC_RETRY_INSTRUCTION.format(
            verified_values=", ".join(verified_values) or "nenhum ainda",
        )
        qualitative = await _run_engine_attempt(
            question=engine_question + retry_instruction,
            display_question=question,
            engine=engine,
            sources=_RECOVERY_SOURCES,
            rewritten_query=rewritten_query,
            is_labor_market=interp.get("is_labor_market", False),
        )
        retry_kept_required_numbers = bool(qualitative.checks) or not verified_values
        retry_has_complete_numeric_support = bool(
            qualitative.checks and not qualitative.unverified
        )
        if (
            _is_fully_supported(qualitative)
            or retry_has_complete_numeric_support
        ) and retry_kept_required_numbers:
            attempt = qualitative
        elif not verified_values and _is_usable_qualitative(qualitative):
            log.warning(
                "Preservando sintese qualitativa com nomes documentados",
                extra={"question": question[:120]},
            )
            attempt = replace(qualitative, unsupported_arguments=[])
        elif not verified_values:
            qualitative_subset = _number_free_subset(pre_qualitative_attempt)
            if qualitative_subset and (
                _is_fully_supported(qualitative_subset)
                or _is_usable_qualitative(qualitative_subset)
            ):
                log.warning(
                    "Recuperando conclusoes qualitativas da resposta anterior",
                    extra={"question": question[:120]},
                )
                attempt = replace(qualitative_subset, unsupported_arguments=[])

    answer = attempt.answer
    source_nodes = attempt.source_nodes
    checks = attempt.checks
    unverified = attempt.unverified
    verified = attempt.verified
    unsupported_arguments = attempt.unsupported_arguments
    answer_blocked_by_guardrail = False
    if unsupported_arguments and checks:
        log.warning(
            "Mantendo analise numerica apesar de baixa sobreposicao lexical",
            extra={
                "question": question[:120],
                "unverified": unverified,
                "unsupported_arguments": unsupported_arguments[:3],
            },
        )
    if unsupported_arguments:
        trimmed_answer = _trim_unsupported_answer(
            answer,
            checks=checks,
            unsupported_arguments=unsupported_arguments,
        )
        if trimmed_answer:
            trimmed_checks = await asyncio.to_thread(
                validate_numbers, trimmed_answer, source_nodes
            )
            trimmed_unverified = [
                check.value for check in trimmed_checks if not check.verified
            ]
            trimmed_argument_checks = await asyncio.to_thread(
                validate_arguments, trimmed_answer, source_nodes
            )
            trimmed_unsupported_arguments = [
                check.sentence
                for check in trimmed_argument_checks
                if not check.verified
            ]
            if not trimmed_unsupported_arguments or trimmed_checks:
                answer = trimmed_answer
                checks = trimmed_checks
                unverified = trimmed_unverified
                verified = len(trimmed_checks) - len(trimmed_unverified)
                unsupported_arguments = trimmed_unsupported_arguments
            else:
                answer_blocked_by_guardrail = True
        else:
            answer_blocked_by_guardrail = True
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
