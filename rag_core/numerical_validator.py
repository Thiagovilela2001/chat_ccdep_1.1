"""
Validação numérica de respostas RAG.

Pipeline:
    resposta do LLM → extrai números → compara com chunks de origem → relatório
"""
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Optional

# ── Regex para números em PT-BR ───────────────────────────────────────────────
# Captura: 1.234,56 | 1.234 | 4,2 | 42 | 9 | com % opcional.
# Inteiros de um dígito também são dados: "9 mil" não pode escapar do
# guardrail. Marcadores puramente estruturais ("1. item", "[1]", "p. 3") são
# filtrados separadamente para não serem confundidos com afirmações numéricas.
_NUM_RE = re.compile(
    r"(?<!\w)"
    r"([-−]?"
    r"(?:"
    r"\d{1,3}(?:\.\d{3})+(?:,\d+)?"   # 1.234 ou 1.234,56
    r"|\d+,\d+"                         # 4,2
    r"|\d+"                             # inteiro, inclusive 1 dígito
    r")"
    r"(?:\s*%)?)"
    r"(?!\w)"
)

_NUM_ATOM = (
    r"[-−]?(?:\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+,\d+|\d+)(?:\s*%)?"
)
_EQUATION_RE = re.compile(
    rf"({_NUM_ATOM})\s*([+\-−×x*/÷])\s*({_NUM_ATOM})\s*=\s*({_NUM_ATOM})"
)

# ── Dataclass de resultado ────────────────────────────────────────────────────

@dataclass
class NumberCheck:
    value: str                          # como aparece na resposta
    verified: bool
    source_snippet: Optional[str] = None  # trecho onde foi encontrado
    response_snippet: Optional[str] = None  # contexto na resposta do LLM
    derived: bool = False
    response_start: Optional[int] = None
    response_end: Optional[int] = None
    source_index: Optional[int] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    """Converte número PT-BR para string numérica comparável."""
    s = s.strip().rstrip("%").strip().replace("−", "-")
    if "," in s and "." in s:
        # 1.234,56 → 1234.56
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        # 4,2 → 4.2
        s = s.replace(",", ".")
    return s


def _is_structural_single_digit(text: str, match: re.Match) -> bool:
    """Ignora numeração de lista/citação, mas nunca um valor em prosa."""
    raw = match.group(1).strip().rstrip("%").strip().lstrip("-−")
    if len(raw) != 1 or not raw.isdigit():
        return False

    start, end = match.start(1), match.end(1)
    line_start = text.rfind("\n", 0, start) + 1
    line_prefix = text[line_start:start]
    suffix = text[end:end + 2]
    if not line_prefix.strip() and re.match(r"[.)]\s", suffix):
        return True

    prefix = text[max(0, start - 24):start]
    if re.search(r"(?:p(?:[\u00e1a]gina)?\.?|p\./aba|aba)\s*:?\s*$", prefix, re.IGNORECASE):
        return True
    if text[max(0, start - 1):start] == "[" and text[end:end + 1] == "]":
        return True
    return False


def _number_matches(text: str):
    return (
        match for match in _NUM_RE.finditer(text)
        if not _is_structural_single_digit(text, match)
    )


_CONTEXT_STOPWORDS = {
    "acumulado", "analisado", "apresentou", "chegou", "emprego", "empregos",
    "entre", "estado", "formal", "mercado", "meses", "periodo", "postos",
    "regiao", "regioes", "registrou", "saldo", "saldos", "trabalho", "variacao",
}
_CONTEXT_WORD_RE = re.compile(r"[a-z]{3,}")
_SCALE_UNIT_RE = re.compile(
    r"^\s*(milh(?:ao|oes)|mil|bilh(?:ao|oes)|trilh(?:ao|oes)|p\.?\s*p\.?)\b",
    re.IGNORECASE,
)


def _fold(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in folded if not unicodedata.combining(ch)).lower()


def _unit_signature(text: str, match: re.Match) -> str:
    """Distingue, por exemplo, `11 mil` de `11%` e de `11` sem escala."""
    raw = match.group(1).strip()
    if raw.endswith("%"):
        return "%"
    suffix = _fold(text[match.end(1):match.end(1) + 24])
    unit = _SCALE_UNIT_RE.match(suffix)
    return re.sub(r"\s+", "", unit.group(1)) if unit else ""


def _nearby_context(text: str, start: int, end: int) -> str:
    window_start = max(0, start - 90)
    boundary = max(text.rfind(mark, window_start, start) for mark in ".!?;\n")
    if boundary >= window_start:
        window_start = boundary + 1

    window_end = min(len(text), end + 40)
    endings = [
        position
        for mark in ".!?;\n"
        if (position := text.find(mark, end, window_end)) >= 0
    ]
    if endings:
        window_end = min(endings)
    return text[window_start:window_end]


def _context_terms(text: str) -> set[str]:
    return {
        term for term in _CONTEXT_WORD_RE.findall(_fold(text))
        if term not in _CONTEXT_STOPWORDS
    }


def _context_overlap(response_context: str, source_context: str) -> int:
    return len(_context_terms(response_context) & _context_terms(source_context))


def _source_candidates(raw: str, response_match: re.Match, response_text: str, source_texts):
    """Retorna ocorrências compatíveis, priorizando forma e unidade exatas."""
    response_unit = _unit_signature(response_text, response_match)
    exact = []
    normalized = []
    norm = _normalize(raw)
    for source_index, source_text in enumerate(source_texts):
        for source_match in _number_matches(source_text):
            if _unit_signature(source_text, source_match) != response_unit:
                continue
            candidate = source_match.group(1).strip()
            item = (source_index, source_text, source_match)
            if candidate == raw:
                exact.append(item)
            elif _normalize(candidate) == norm:
                normalized.append(item)
    return exact or normalized


def _normalized_numbers(text: str) -> set[str]:
    return {_normalize(match.group(1)) for match in _number_matches(text)}


def _derived_results(response_text: str, source_texts: list[str]) -> set[str]:
    """Aceita apenas resultados de equações explícitas cujos operandos estão nas fontes."""
    source_numbers = set().union(*(_normalized_numbers(text) for text in source_texts))
    derived: set[str] = set()
    for match in _EQUATION_RE.finditer(response_text):
        left_raw, operator, right_raw, result_raw = match.groups()
        if _normalize(left_raw) not in source_numbers or _normalize(right_raw) not in source_numbers:
            continue
        try:
            left = Decimal(_normalize(left_raw))
            right = Decimal(_normalize(right_raw))
            expected = Decimal(_normalize(result_raw))
            if operator == "+":
                calculated = left + right
            elif operator in {"-", "−"}:
                calculated = left - right
            elif operator in {"×", "x", "*"}:
                calculated = left * right
            elif right != 0:
                calculated = left / right
            else:
                continue
        except (InvalidOperation, ZeroDivisionError):
            continue
        decimals = max(0, -expected.as_tuple().exponent)
        tolerance = Decimal("0.5") * (Decimal(10) ** -decimals)
        if abs(calculated - expected) <= tolerance:
            derived.add(_normalize(result_raw))
    return derived


def _find_snippet(needle: str, haystack: str, ctx: int = 60) -> str:
    idx = haystack.find(needle)
    if idx == -1:
        return ""
    return _find_snippet_at(haystack, idx, idx + len(needle), ctx=ctx)


def _find_snippet_at(haystack: str, start_idx: int, end_idx: int, ctx: int = 60) -> str:
    start = max(0, start_idx - ctx)
    end = min(len(haystack), end_idx + ctx)
    return "..." + haystack[start:end].strip() + "..."


def _source_snippet(
    needle: str,
    source_text: str,
    source_node,
    match_start: Optional[int] = None,
    match_end: Optional[int] = None,
) -> str:
    """Preserva bloco tabular completo; usa contexto narrativo nos demais casos."""
    metadata = getattr(source_node, "metadata", {}) or {}
    if metadata.get("type") == "table":
        for match in re.finditer(r"(?s)(?:^|\n\s*---\s*\n)(.*?)(?=\n\s*---\s*\n|$)", source_text):
            block = match.group(1)
            if match_start is not None:
                if not (match.start(1) <= match_start <= match.end(1)):
                    continue
            elif needle not in block:
                continue
            lines = [
                line.strip()
                for line in block.splitlines()
                if line.strip() and not line.strip().lower().startswith("fonte:")
            ]
            return "\n".join(lines)[:1_000]
    if match_start is not None and match_end is not None:
        return _find_snippet_at(source_text, match_start, match_end, ctx=140)
    return _find_snippet(needle, source_text, ctx=140)


# Validacao principal

def validate_numbers(response_text: str, source_nodes) -> list[NumberCheck]:
    """
    Extrai todos os números da resposta e verifica se existem nos chunks de origem.
    Tenta primeiro match verbatim, depois match normalizado (diferenças de formatação).
    """
    source_texts = [n.get_content() for n in source_nodes]
    derived_results = _derived_results(response_text, source_texts)

    response_matches = list(_number_matches(response_text))
    value_counts = Counter(_normalize(match.group(1)) for match in response_matches)
    used_source_occurrences: set[tuple[int, int, int]] = set()
    results: list[NumberCheck] = []

    for m in response_matches:
        raw = m.group(1).strip()
        norm = _normalize(raw)
        resp_snippet = _find_snippet_at(response_text, m.start(1), m.end(1), ctx=140)
        response_position = {
            "response_start": m.start(1),
            "response_end": m.end(1),
        }

        response_context = _nearby_context(response_text, m.start(1), m.end(1))
        ranked_candidates = []
        for source_index, node_text, cm in _source_candidates(
            raw, m, response_text, source_texts
        ):
            occurrence = (source_index, cm.start(1), cm.end(1))
            if occurrence in used_source_occurrences:
                continue
            source_context = _nearby_context(node_text, cm.start(1), cm.end(1))
            ranked_candidates.append((
                _context_overlap(response_context, source_context),
                source_index,
                node_text,
                cm,
                occurrence,
            ))

        ranked_candidates.sort(key=lambda item: item[0], reverse=True)
        chosen = ranked_candidates[0] if ranked_candidates else None
        if chosen and (value_counts[norm] == 1 or chosen[0] > 0):
            _, source_index, node_text, cm, occurrence = chosen
            used_source_occurrences.add(occurrence)
            results.append(NumberCheck(
                value=raw,
                verified=True,
                source_snippet=_source_snippet(
                    cm.group(1),
                    node_text,
                    source_nodes[source_index],
                    cm.start(1),
                    cm.end(1),
                ),
                response_snippet=resp_snippet,
                source_index=source_index,
                **response_position,
            ))
            continue

        # 2. Match normalizado (trata diferenças de formatação PT-BR vs EN)
        if norm in derived_results:
            results.append(NumberCheck(
                value=raw,
                verified=True,
                response_snippet=resp_snippet,
                derived=True,
                **response_position,
            ))
        else:
            results.append(NumberCheck(
                value=raw,
                verified=False,
                response_snippet=resp_snippet,
                **response_position,
            ))

    return results


def format_validation_report(checks: list[NumberCheck]) -> str:
    """Formata o relatório de validação para exibição no console."""
    if not checks:
        return "  Nenhum número encontrado na resposta."

    verified = [c for c in checks if c.verified]
    unverified = [c for c in checks if not c.verified]

    lines = [f"  Verificados nos documentos: {len(verified)}/{len(checks)}"]

    if unverified:
        lines.append("  ⚠️  Não encontrados nos documentos originais:")
        for c in unverified:
            lines.append(f"    • {c.value}")
            if c.response_snippet:
                lines.append(f"      Contexto na resposta: {c.response_snippet}")
    else:
        lines.append("  ✅ Todos os números foram verificados nos documentos.")

    return "\n".join(lines)
