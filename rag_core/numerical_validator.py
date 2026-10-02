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
    applicable: bool = True
    reason: Optional[str] = None
    period: Optional[str] = None
    indicator: Optional[str] = None
    unit: Optional[str] = None


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


def _parse_decimal(s: str) -> Optional[Decimal]:
    norm = _normalize(s)
    try:
        return Decimal(norm)
    except (InvalidOperation, ValueError):
        return None


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
    """
    Distingue, por exemplo, `11 mil` de `11%` e de `11` sem escala.
    Reconhece também a unidade declarada em cabeçalho ou bloco de tabela precedente.
    """
    raw = match.group(1).strip()
    if raw.endswith("%"):
        return "%"
    suffix = _fold(text[match.end(1):match.end(1) + 24])
    unit = _SCALE_UNIT_RE.match(suffix)
    if unit:
        return re.sub(r"\s+", "", unit.group(1))

    # Reconhece unidade declarada em cabeçalho ou linha precedente de tabela
    preceding = _fold(text[max(0, match.start(1) - 300):match.start(1)])
    if any(p in preceding for p in ["em %", "em porcentagem", "(%)", "taxas (%)", "variacao (%)", "participacao (%)"]):
        return "%"
    if "por mil" in preceding or "por 1.000" in preceding:
        return "pormil"
    if "em mil" in preceding or "em milhares" in preceding:
        return "mil"
    return ""


_NEGATIVE_DIRECTION_WORDS = {
    "recuou", "recuo", "caiu", "queda", "quedas", "retraiu", "retracao", "retração",
    "retrações", "decresceu", "decrescimo", "decréscimo", "decréscimos", "perda",
    "perdas", "perdeu", "negativa", "negativas", "negativo", "negativos", "contraiu",
    "contracao", "contração", "diminuiu", "diminuição", "reduziu", "reducao", "redução",
    "reduções", "deficit", "déficit",
}


def _has_negative_direction(context: str) -> bool:
    terms = _context_terms(context)
    if terms & _NEGATIVE_DIRECTION_WORDS:
        return True
    folded = _fold(context)
    return any(w in folded for w in ["recuo", "queda", "retra", "decres", "redu", "negativ"])


def _is_rate_or_percentage(raw: str, unit: str, dec: Decimal) -> bool:
    if raw.endswith("%") or unit in {"%", "p.p.", "pp"}:
        return True
    if 1900 <= abs(dec) <= 2100:
        return False
    return True


def _effective_signed_value(val: Decimal, context: str, raw: str, unit: str) -> Decimal:
    """Se o contexto expressa queda/retração explícita e o valor é positivo, seu valor assinado é negativo."""
    if val > 0 and _is_rate_or_percentage(raw, unit, val) and _has_negative_direction(context):
        return -val
    return val


_MONTHS = {
    "janeiro": 1, "fevereiro": 2, "marco": 3, "março": 3,
    "abril": 4, "maio": 5, "junho": 6, "julho": 7,
    "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
}

_QUARTERS = {
    "1t": 1, "1o trimestre": 1, "1º trimestre": 1, "primeiro trimestre": 1,
    "2t": 2, "2o trimestre": 2, "2º trimestre": 2, "segundo trimestre": 2,
    "3t": 3, "3o trimestre": 3, "3º trimestre": 3, "terceiro trimestre": 3,
    "4t": 4, "4o trimestre": 4, "4º trimestre": 4, "quarto trimestre": 4,
}


def _extract_period_signatures(context: str) -> dict[str, set]:
    folded = _fold(context)
    months = {m for m in _MONTHS if re.search(rf"\b{m}\b", folded)}
    quarters = {q for q in _QUARTERS if q in folded}
    years = set(re.findall(r"\b20\d\d\b", folded))
    is_accumulated_12m = bool(re.search(r"\b(?:12\s*meses|4\s*trimestres|quatro\s*trimestres)\b", folded))
    return {
        "months": months,
        "quarters": quarters,
        "years": years,
        "accumulated_12m": is_accumulated_12m,
    }


def _has_period_conflict(resp_context: str, src_context: str) -> bool:
    resp_p = _extract_period_signatures(resp_context)
    src_p = _extract_period_signatures(src_context)

    # Conflito de ano: ambos declaram anos e os conjuntos são completamente disjuntos
    if resp_p["years"] and src_p["years"]:
        if not (resp_p["years"] & src_p["years"]):
            return True

    # Conflito trimestre vs mês:
    # Ex: resposta trata de 1º trimestre (jan-mar) e a fonte exclusivamente de abril (mês 4)
    has_resp_q1 = any(q in resp_p["quarters"] for q in ["1t", "primeiro trimestre", "1º trimestre", "1o trimestre"])
    has_src_q1 = any(q in src_p["quarters"] for q in ["1t", "primeiro trimestre", "1º trimestre", "1o trimestre"])

    shares_quarter = bool(resp_p["quarters"] & src_p["quarters"])

    if has_resp_q1 and not shares_quarter and ("abril" in src_p["months"] or "maio" in src_p["months"]):
        return True

    if has_src_q1 and not shares_quarter and ("abril" in resp_p["months"] or "maio" in resp_p["months"]):
        return True

    return False


_REFUSAL_PATTERNS = [
    "não consta", "nao consta",
    "não está disponível", "nao esta disponivel",
    "não é possível determinar", "nao e possivel determinar",
    "não foi possível determinar", "nao foi possivel determinar",
    "não foram encontrados", "nao foram encontrados",
    "não há dados", "nao ha dados",
    "não há informações", "nao ha informacoes",
    "não permitem determinar", "nao permitem determinar",
    "não sustentam", "nao sustentam",
    "não estão sustentadas", "nao estao sustentadas",
    "sem elementos para avaliar", "sem dados para",
]


_GENERIC_METRIC_TERMS = {
    "indicador", "indicadores", "resultado", "resultados",
    "serie", "series", "indice", "indices", "taxa", "taxas",
    "dado", "dados", "valor", "valores", "medida", "medidas",
    "numero", "numeros", "participacao", "crescimento",
    "saldo", "saldos",
}

_CLAUSE_BOUNDARY_RE = re.compile(
    r"(?:[.!?]\s+|\n\s*\n|;\s*|,\s*e\s+|\s+e\s+(?:o|a|os|as|no|na|nos|nas|em)\s+|\s+enquanto\s+|\s+ao passo que\s+|\s+mas\s+)",
    re.IGNORECASE,
)


def _nearby_context(text: str, start: int, end: int) -> str:
    window_start = max(0, start - 140)
    boundaries = [window_start]
    for m in _CLAUSE_BOUNDARY_RE.finditer(text[window_start:start]):
        boundaries.append(window_start + m.end())
    window_start = max(boundaries)

    window_end = min(len(text), end + 100)
    end_boundaries = [window_end]
    for m in _CLAUSE_BOUNDARY_RE.finditer(text[end:window_end]):
        end_boundaries.append(end + m.start())
    window_end = min(end_boundaries)

    return text[window_start:window_end]


def _context_terms(text: str) -> set[str]:
    return {
        term for term in _CONTEXT_WORD_RE.findall(_fold(text))
        if term not in _CONTEXT_STOPWORDS or term in _GENERIC_METRIC_TERMS
    }


def _context_overlap(response_context: str, source_context: str) -> int:
    return len(_context_terms(response_context) & _context_terms(source_context))


def _source_candidates(raw: str, response_match: re.Match, response_text: str, source_texts: list[str]):
    """Retorna ocorrências compatíveis nas fontes, priorizando forma e suporte semântico."""
    response_unit = _unit_signature(response_text, response_match)
    resp_dec = _parse_decimal(raw)
    resp_context = _nearby_context(response_text, response_match.start(1), response_match.end(1))
    resp_effective = _effective_signed_value(resp_dec, resp_context, raw, response_unit) if resp_dec is not None else None

    exact = []
    normalized = []
    for source_index, source_text in enumerate(source_texts):
        for source_match in _number_matches(source_text):
            src_unit = _unit_signature(source_text, source_match)
            if src_unit != response_unit:
                continue

            candidate = source_match.group(1).strip()
            item = (source_index, source_text, source_match)

            src_context = _nearby_context(source_text, source_match.start(1), source_match.end(1))
            if _has_period_conflict(resp_context, src_context):
                continue

            src_dec = _parse_decimal(candidate)
            if src_dec is not None and resp_dec is not None:
                src_effective = _effective_signed_value(src_dec, src_context, candidate, src_unit)
                # Equivalência de valor decimal ou com direção negativa
                if candidate == raw:
                    exact.append(item)
                elif resp_dec == src_dec or (resp_effective is not None and resp_effective == src_effective):
                    normalized.append(item)
            elif candidate == raw:
                exact.append(item)

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

def validate_numbers(
    response_text: str,
    source_nodes,
    question: Optional[str] = None,
) -> list[NumberCheck]:
    """
    Extrai todos os números da resposta e verifica se existem nos chunks de origem.
    Aplica validação semântica:
    - Equivalência decimal (1,60 == 1,6)
    - Unidade herdada de cabeçalho de tabela
    - Linguagem direcional (recuou 1,1% == -1,1%)
    - Conflito de período (trimestre vs mês)
    - Sobreposição contextual mínima (> 0)
    - Números da pergunta repetidos em recusas seguras não são penalizados
    """
    source_texts = [n.get_content() for n in source_nodes]
    derived_results = _derived_results(response_text, source_texts)

    response_matches = list(_number_matches(response_text))
    used_source_occurrences: set[tuple[int, int, int]] = set()
    results: list[NumberCheck] = []

    question_numbers = set()
    if question:
        question_numbers = {
            _normalize(m.group(1)) for m in _number_matches(question)
        }

    is_refusal_response = any(
        kw in _fold(response_text) for kw in _REFUSAL_PATTERNS
    )

    for m in response_matches:
        raw = m.group(1).strip()
        norm = _normalize(raw)
        resp_snippet = _find_snippet_at(response_text, m.start(1), m.end(1), ctx=140)
        response_position = {
            "response_start": m.start(1),
            "response_end": m.end(1),
        }

        response_context = _nearby_context(response_text, m.start(1), m.end(1))
        unit = _unit_signature(response_text, m)
        resp_p = _extract_period_signatures(response_context)
        period_str = ", ".join(sorted(resp_p["years"] | resp_p["quarters"] | resp_p["months"])) or None

        # 0. Números da pergunta repetidos em contexto de recusa segura não são penalizados
        if norm in question_numbers and is_refusal_response:
            snippet_has_refusal = any(kw in _fold(resp_snippet) for kw in _REFUSAL_PATTERNS)
            if snippet_has_refusal or len(response_text.strip()) < 300:
                results.append(NumberCheck(
                    value=raw,
                    verified=False,
                    applicable=False,
                    reason="Numero repetido da pergunta em recusa segura",
                    response_snippet=resp_snippet,
                    unit=unit,
                    period=period_str,
                    **response_position,
                ))
                continue

        ranked_candidates = []
        for source_index, node_text, cm in _source_candidates(
            raw, m, response_text, source_texts
        ):
            occurrence = (source_index, cm.start(1), cm.end(1))
            if occurrence in used_source_occurrences:
                continue
            source_context = _nearby_context(node_text, cm.start(1), cm.end(1))
            overlap = _context_overlap(response_context, source_context)
            has_metric_compat = bool(
                _context_terms(response_context) & _GENERIC_METRIC_TERMS
                and _context_terms(source_context) & _GENERIC_METRIC_TERMS
            )
            if overlap == 0 and not has_metric_compat:
                # Regra Frente B: exige compatibilidade mínima; overlap 0 sem indicador compatível não sustenta
                continue

            is_exact_str = (cm.group(1).strip() == raw)
            score = (overlap * 10) + (3 if has_metric_compat else 0) + (5 if is_exact_str else 0)
            ranked_candidates.append((
                score,
                source_index,
                node_text,
                cm,
                occurrence,
            ))

        ranked_candidates.sort(key=lambda item: item[0], reverse=True)
        chosen = ranked_candidates[0] if ranked_candidates else None
        if chosen and chosen[0] > 0:
            _, source_index, node_text, cm, occurrence = chosen
            used_source_occurrences.add(occurrence)
            results.append(NumberCheck(
                value=raw,
                verified=True,
                applicable=True,
                reason="Verificado com suporte contextual e equivalência semântica",
                source_snippet=_source_snippet(
                    cm.group(1),
                    node_text,
                    source_nodes[source_index],
                    cm.start(1),
                    cm.end(1),
                ),
                response_snippet=resp_snippet,
                source_index=source_index,
                unit=unit or _unit_signature(node_text, cm),
                period=period_str,
                **response_position,
            ))
            continue

        # 2. Match normalizado (trata diferenças de formatação PT-BR vs EN)
        if norm in derived_results:
            results.append(NumberCheck(
                value=raw,
                verified=True,
                applicable=True,
                derived=True,
                reason="Resultado derivado de operacao aritmetica explicita",
                response_snippet=resp_snippet,
                unit=unit,
                period=period_str,
                **response_position,
            ))
        else:
            results.append(NumberCheck(
                value=raw,
                verified=False,
                applicable=True,
                reason="Nao encontrado nas fontes ou sem sobreposicao contextual",
                response_snippet=resp_snippet,
                unit=unit,
                period=period_str,
                **response_position,
            ))

    return results


def format_validation_report(checks: list[NumberCheck]) -> str:
    """Formata o relatório de validação para exibição no console."""
    applicable = [c for c in checks if getattr(c, "applicable", True)]
    if not applicable:
        return "  Nenhum número aplicável para validação na resposta."

    verified = [c for c in applicable if c.verified]
    unverified = [c for c in applicable if not c.verified]

    lines = [f"  Verificados nos documentos: {len(verified)}/{len(applicable)}"]

    if unverified:
        lines.append("  ⚠️  Não encontrados nos documentos originais:")
        for c in unverified:
            lines.append(f"    • {c.value}")
            if c.response_snippet:
                lines.append(f"      Contexto na resposta: {c.response_snippet}")
    else:
        lines.append("  ✅ Todos os números foram verificados nos documentos.")

    return "\n".join(lines)
