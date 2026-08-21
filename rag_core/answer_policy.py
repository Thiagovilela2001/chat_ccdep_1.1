"""Política compartilhada para o texto visível das respostas RAG."""
from __future__ import annotations

import re
import unicodedata

from .demographic_indicators import is_demographic_indicator_query

REFUSAL_TEXT = (
    "Os documentos disponíveis não fornecem evidência suficiente para responder "
    "ao ponto solicitado."
)

_REFUSAL_RE = re.compile(re.escape(REFUSAL_TEXT), re.IGNORECASE)

CALCULATION_PROVENANCE_TEXT = (
    "Proveniência obrigatória do cálculo: este resultado foi obtido exclusivamente "
    "com valores extraídos dos documentos fornecidos e validados contra esses documentos; "
    "não foi criado pela LLM nem obtido de fonte externa."
)
CALCULATION_FAILURE_PROVENANCE_TEXT = (
    "Proveniência obrigatória do cálculo: nenhum resultado numérico foi criado pela "
    "LLM. O cálculo não foi apresentado porque seus valores, sua operação ou seus "
    "documentos de origem não puderam ser validados."
)
CALCULATION_SOURCES_HEADER = "Documentos dos quais os valores foram extraídos:"
CALCULATION_CLARIFICATION_PERIOD_TEXT = (
    "Quais períodos você deseja comparar? Informe dois anos, trimestres ou datas "
    "de referência para que o cálculo seja rastreável."
)
CALCULATION_CLARIFICATION_TERRITORY_TEXT = (
    "Qual território você deseja analisar? Informe o estado, município ou recorte "
    "regional para que o cálculo seja rastreável."
)
CALCULATION_CLARIFICATION_INDICATOR_TEXT = (
    "Qual indicador você deseja calcular ou comparar? Informe também a unidade "
    "quando houver mais de uma definição nos documentos."
)
CALCULATION_CLARIFICATION_AGE_TEXT = (
    "Quais faixas etárias devem ser usadas? Informe os limites das faixas ou autorize "
    "o padrão 0–14, 15–64 e 65+."
)
CALCULATION_CLARIFICATION_GENERIC_TEXT = (
    "O pedido de cálculo admite mais de um recorte. Informe os períodos, o território "
    "e o indicador desejados para que o resultado seja rastreável."
)
CALCULATION_CANCELLED_TEXT = (
    "Operação cancelada por ausência de dados na fonte."
)
_CALCULATION_SOURCE_BLOCK_RE = re.compile(
    rf"{re.escape(CALCULATION_SOURCES_HEADER)}\n(?:- [^\n]+\n?)+",
    re.IGNORECASE,
)
_CALCULATION_AUDIT_BLOCK_RE = re.compile(
    rf"{re.escape(CALCULATION_SOURCES_HEADER)}\n(?:- [^\n]+\n?)+\n"
    rf"{re.escape(CALCULATION_PROVENANCE_TEXT)}",
    re.IGNORECASE,
)

_CURRENT_QUESTION_RE = re.compile(
    r"<PERGUNTA_ATUAL>\s*(.*?)\s*</PERGUNTA_ATUAL>", re.DOTALL | re.IGNORECASE
)
_CALCULATION_REQUEST_PATTERNS = (
    re.compile(r"\bcalcul(?:e|ar|ando|ado|ada)\b", re.IGNORECASE),
    re.compile(r"\bfa[cç]a\s+(?:(?:o|um|os)\s+)?c[aá]lculos?\b", re.IGNORECASE),
    re.compile(
        r"\b(?:some|somar|subtraia|subtrair|multiplique|multiplicar|"
        r"divida|dividir|compute|computar)\b",
        re.IGNORECASE,
    ),
)
_CALCULATION_VALUE_PATTERN = r"[-−]?\d(?:[\d.,]*\d)?(?:\s*%)?"
_EXPLICIT_EQUATION_RE = re.compile(
    rf"{_CALCULATION_VALUE_PATTERN}\s*(?:[+\-−×x*/÷])\s*"
    rf"{_CALCULATION_VALUE_PATTERN}(?:\s*(?:[+\-−×x*/÷])\s*"
    rf"{_CALCULATION_VALUE_PATTERN})*\s*=\s*{_CALCULATION_VALUE_PATTERN}",
    re.IGNORECASE,
)
_CALCULATION_FAILURE_RE = re.compile(
    r"\b(?:n[aã]o foi poss[ií]vel|n[aã]o realizado|n[aã]o p[oô]de ser|"
    r"dados? insuficientes?|valores? incompat[ií]veis?)\b",
    re.IGNORECASE,
)
_MISSING_DATA_SIGNAL_RE = re.compile(
    r"\b(?:ausente|ausência|indispon[ií]vel|nulo|null|n[aã]o consta|"
    r"n[aã]o cont[eé]m|n[aã]o fornece|n[aã]o foi encontrad[oa]|"
    r"n[aã]o est[aá] dispon[ií]vel|sem dados?|falta(?:m|ndo)?)\b",
    re.IGNORECASE,
)
_PERIOD_RE = re.compile(
    r"\b(?:"
    r"(?:(?:[1-4](?:º|°|o)?|primeiro|segundo|terceiro|quarto)\s+trimestre|"
    r"(?:[12](?:º|°|o)?|primeiro|segundo)\s+semestre)\s+(?:de\s+)?"
    r"(?:19|20)\d{2}|"
    r"[1-4]\s*[Tt]\s*(?:19|20)\d{2}|"
    r"(?:janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|"
    r"outubro|novembro|dezembro)\s+de\s+(?:19|20)\d{2}|"
    r"(?:19|20)\d{2}"
    r")\b",
    re.IGNORECASE,
)
_AMBIGUITY_QUALIFIER_PATTERN = (
    r"(?:nao (?:(?:foi|foram|esta|estao) )?"
    r"(?:especific\w*|inform\w*|identific\w*|defin\w*)|"
    r"nao e possivel identificar|sem (?:a )?definicao clara|ambigu\w*|"
    r"precis\w* (?:que )?(?:voce )?informe)"
)

_SOURCE_REQUEST_PATTERNS = (
    re.compile(
        r"\b(?:qual|quais|liste|mostre|informe|indique)\s+"
        r"(?:(?:foi|foram|[eé]|s[aã]o)\s+)?"
        r"(?:a\s+|as\s+|o\s+|os\s+)?"
        r"(?:fontes?|refer[eê]ncias?|documentos?|arquivos?|pdfs?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bde onde (?:veio|vem|saiu|foi retirada|foi obtida)\b", re.IGNORECASE),
    re.compile(
        r"\bonde (?:isso|essa informa[cç][aã]o|esse dado|essa afirma[cç][aã]o) "
        r"(?:consta|aparece|est[aá])\b",
        re.IGNORECASE,
    ),
    re.compile(r"\borigem (?:da|dessa) informa[cç][aã]o\b", re.IGNORECASE),
)

_INTERNAL_MARKER_RE = re.compile(
    r"\b(?:"
    r"chunks?|embeddings?|reranking|re-ranking|"
    r"scores? de similaridade|pontua[cç][aã]o de relev[aâ]ncia|"
    r"contexto enviado ao modelo|contexto recuperado|"
    r"documentos? recuperados?|trechos? recuperados?|"
    r"recupera[cç][aã]o (?:de documentos|sem[aâ]ntica|vetorial)|"
    r"sistema RAG|pipeline RAG"
    r")\b",
    re.IGNORECASE,
)

_PARENTHETICAL_SOURCE_RE = re.compile(
    r"\s*\(\s*(?:\*{0,2})?"
    r"(?:fonte|fontes|documento|pdf|arquivo)"
    r"(?:\*{0,2})?\s*(?::|,).*?\)",
    re.IGNORECASE,
)
_PARENTHETICAL_FILE_RE = re.compile(
    r"\s*\([^()\n]*\.(?:pdf|docx?|xlsx?|xls|csv|tsv|txt)"
    r"[^()\n]*\)",
    re.IGNORECASE,
)
_BRACKET_SOURCE_RE = re.compile(
    r"\s*\[(?:\s*\d+(?:\s*[,;]\s*\d+)*\s*|"
    r"\s*(?:fonte|fontes|documento|pdf|arquivo)\b[^\]]*)\]",
    re.IGNORECASE,
)
_ATTRIBUTION_PREFIX_RE = re.compile(
    r"\b(?:segundo|conforme|de acordo com)\s+"
    r"(?:o|a|os|as)?\s*(?:documento|pdf|arquivo|fonte)s?"
    r"(?:\s+[^,;:]+)?\s*[,;:]\s*",
    re.IGNORECASE,
)
_FILE_NAME_RE = re.compile(
    r"(?<![\w.-])[\wÀ-ÿ()_-]+(?:[ _-][\wÀ-ÿ()_-]+)*"
    r"\.(?:pdf|docx?|xlsx?|xls|csv|tsv|txt)\b",
    re.IGNORECASE,
)
_PAGE_REFERENCE_RE = re.compile(
    r"\b(?:p(?:[áa]gina)?\.?|p\./aba|aba)\s*:?\s*"
    r"\d+(?:\s*[-–—]\s*\d+)?\b",
    re.IGNORECASE,
)
_PAGE_ONLY_SENTENCE_RE = re.compile(
    r"(?:^|(?<=[.!?])\s+)"
    r"[^.!?\n]*(?:aparece|consta|encontra-se|est[aá]|veja|consulte)"
    r"\s+(?:na|no|à|a)?\s*(?:p(?:[áa]gina)?\.?|p\./aba|aba)"
    r"\s*:?\s*\d+(?:\s*[-–—]\s*\d+)?[^.!?\n]*[.!?]?",
    re.IGNORECASE,
)
_METADATA_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\*{0,2})?"
    r"(?:fonte|fontes|documento|documentos|p[áa]gina|p[áa]ginas|aba|abas)"
    r"(?:\*{0,2})?\s*:?.*$",
    re.IGNORECASE,
)
_SOURCE_SECTION_RE = re.compile(
    r"^\s{0,3}#{1,6}\s*(?:fontes?|refer[eê]ncias?|documentos?)\s*:?\s*$",
    re.IGNORECASE,
)
_SOURCE_BULLET_RE = re.compile(
    r"^\s*[-*]\s+.*(?:\.(?:pdf|docx?|xlsx?|xls|csv|tsv|txt)\b|"
    r"\bp(?:[áa]gina)?\.?\s*:?\s*\d+|\bscore\b).*$",
    re.IGNORECASE,
)
_SOURCE_SENTENCE_RE = re.compile(
    r"(?:^|(?<=[.!?])\s+)(?:\*{0,2})?"
    r"(?:fonte|fontes|documento|documentos|refer[eê]ncia|refer[eê]ncias)"
    r"(?:\*{0,2})?\s*:[^\n]*?"
    r"(?:[.!?](?=\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ])|$)",
    re.IGNORECASE,
)


def asks_for_sources(question: str) -> bool:
    """Detecta pedido explícito por fonte, referência ou documento de origem."""
    return any(pattern.search(question or "") for pattern in _SOURCE_REQUEST_PATTERNS)


def _current_question(question: str) -> str:
    match = _CURRENT_QUESTION_RE.search(question or "")
    return match.group(1) if match else question


def asks_for_calculation(question: str) -> bool:
    """Detecta pedido explícito de conta sem herdar assunto antigo da memória."""
    current = _current_question(question)
    return is_demographic_indicator_query(current) or any(
        pattern.search(current or "") for pattern in _CALCULATION_REQUEST_PATTERNS
    )


def has_explicit_calculation(answer: str) -> bool:
    """Exige substituição numérica com operador e resultado explícito."""
    return bool(_EXPLICIT_EQUATION_RE.search(answer or ""))


def _fold_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", (value or "").casefold())
    without_marks = "".join(
        character for character in decomposed
        if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", without_marks).strip()


def _ambiguous_dimension(text: str, terms: str) -> bool:
    return bool(re.search(
        rf"(?:{_AMBIGUITY_QUALIFIER_PATTERN}).{{0,120}}(?:{terms})|"
        rf"(?:{terms}).{{0,120}}(?:{_AMBIGUITY_QUALIFIER_PATTERN})",
        text,
    ))


def calculation_clarification(answer: str) -> str | None:
    """Converte ambiguidade reconhecida em pergunta objetiva, sem calcular."""
    normalized = _fold_text(answer)
    if _ambiguous_dimension(normalized, r"period\w*|anos?|trimestres?|datas?"):
        return CALCULATION_CLARIFICATION_PERIOD_TEXT
    if _ambiguous_dimension(
        normalized,
        r"territori\w*|municipi\w*|regi(?:ao|oes|onal|onais)|"
        r"recorte geografic\w*|localidade\w*",
    ):
        return CALCULATION_CLARIFICATION_TERRITORY_TEXT
    if _ambiguous_dimension(normalized, r"indicador\w*|variave\w*|medida\w*|taxa\w*"):
        return CALCULATION_CLARIFICATION_INDICATOR_TEXT
    if _ambiguous_dimension(normalized, r"faixas? etari\w*|grupos? etari\w*"):
        return CALCULATION_CLARIFICATION_AGE_TEXT
    if re.search(r"\bambigu\w*\b", normalized):
        return CALCULATION_CLARIFICATION_GENERIC_TEXT
    return None


def _requested_periods(question: str) -> list[str]:
    periods: list[str] = []
    for match in _PERIOD_RE.finditer(_current_question(question)):
        period = re.sub(r"\s+", " ", match.group(0)).strip()
        if period.casefold() not in {item.casefold() for item in periods}:
            periods.append(period)
    return periods


def _missing_periods(answer: str, periods: list[str]) -> list[str]:
    """Localiza somente recortes pedidos que o texto marcou como ausentes."""
    normalized = _fold_text(answer)
    missing: list[str] = []
    absence = (
        r"(?:ausent\w*|indisponivel|nulo|null|nao consta|nao contem|nao fornece|"
        r"nao foi encontrad\w*|nao esta disponivel|sem (?:o )?dado|falt\w*)"
    )
    for period in periods:
        folded_period = re.escape(_fold_text(period))
        direct_relation = re.search(
            rf"(?:{folded_period})[^.;,]{{0,45}}(?:{absence})|"
            rf"(?:{absence})[^.;,]{{0,45}}(?:{folded_period})",
            normalized,
        )
        if direct_relation:
            missing.append(period)
    return missing


def _calculation_subject(question: str) -> str:
    normalized = _fold_text(_current_question(question))
    subjects = (
        (r"indices? de envelhecimento", "Índice de envelhecimento"),
        (r"razao de dependencia jovem", "Razão de dependência jovem"),
        (r"razao de dependencia idosa", "Razão de dependência idosa"),
        (r"razao de dependencia total", "Razão de dependência total"),
        (r"razoes? de dependencia", "Razão de dependência"),
    )
    for pattern, label in subjects:
        if re.search(pattern, normalized):
            return label
    return "Dado solicitado"


def _calculation_failure_status(question: str) -> str:
    normalized = _fold_text(_current_question(question))
    if "diferenca absoluta" in normalized:
        return "Não é possível calcular a diferença absoluta."
    if "variacao percentual" in normalized:
        return "Não é possível calcular a variação percentual."
    if re.search(r"\bcompar(?:e|ar|acao)\b", normalized):
        return "Não é possível realizar a comparação solicitada."
    return "Não é possível realizar o cálculo solicitado."


def calculation_missing_data_refusal(answer: str, *, question: str) -> str | None:
    """Padroniza falta de dados sem expor valores ou períodos não pedidos."""
    periods = _requested_periods(question)
    if not periods:
        return None
    if answer.strip() != REFUSAL_TEXT and not _MISSING_DATA_SIGNAL_RE.search(answer):
        return None

    missing = _missing_periods(answer, periods)
    if not missing:
        missing = periods
    found = [period for period in periods if period not in missing]
    subject = _calculation_subject(question)

    found_text = (
        f"{subject} para {', '.join(found)}."
        if found
        else "Nenhum dos períodos solicitados."
    )
    missing_text = f"{subject} para {', '.join(missing)}."
    return (
        f"{_calculation_failure_status(question)}\n\n"
        f"- Dado encontrado: {found_text}\n"
        f"- Dado ausente: {missing_text}\n"
        f"{CALCULATION_CANCELLED_TEXT}"
    )


def enforce_calculation_provenance(
    answer: str,
    *,
    question: str,
    checks: list,
    sources: list[str] | None = None,
) -> str:
    """Anexa proveniência ou bloqueia resultado numérico sem validação documental."""
    explicit_operation = has_explicit_calculation(answer)
    calculation = asks_for_calculation(question) or explicit_operation
    if not calculation:
        return answer
    clarification = calculation_clarification(answer)
    if clarification:
        return clarification
    missing_data_refusal = calculation_missing_data_refusal(answer, question=question)
    if missing_data_refusal:
        return missing_data_refusal
    answer = _CALCULATION_SOURCE_BLOCK_RE.sub("", answer)
    answer = answer.replace(CALCULATION_PROVENANCE_TEXT, "")
    answer = answer.replace(CALCULATION_FAILURE_PROVENANCE_TEXT, "").rstrip()

    clean_sources = []
    for source in sources or []:
        normalized = re.sub(r"[\r\n]+", " ", str(source)).strip()
        if normalized and normalized not in clean_sources:
            clean_sources.append(normalized[:500])

    if (
        explicit_operation
        and checks
        and all(getattr(check, "verified", False) for check in checks)
        and clean_sources
    ):
        source_list = "\n".join(f"- {source}" for source in clean_sources)
        return (
            f"{answer.rstrip()}\n\n{CALCULATION_SOURCES_HEADER}\n{source_list}\n\n"
            f"{CALCULATION_PROVENANCE_TEXT}"
        )

    refused_or_failed = answer.strip() == REFUSAL_TEXT or _CALCULATION_FAILURE_RE.search(answer)
    if refused_or_failed and not checks:
        return f"{answer.rstrip()}\n\n{CALCULATION_FAILURE_PROVENANCE_TEXT}"
    return f"{REFUSAL_TEXT}\n\n{CALCULATION_FAILURE_PROVENANCE_TEXT}"


def _strip_internal_details(text: str) -> str:
    """Remove frases que expõem mecanismos internos sem afetar análise econômica."""
    parts = re.split(r"(?<=[.!?])(\s+)", text)
    kept: list[str] = []
    for index in range(0, len(parts), 2):
        sentence = parts[index]
        separator = parts[index + 1] if index + 1 < len(parts) else ""
        if not _INTERNAL_MARKER_RE.search(sentence):
            kept.extend((sentence, separator))
    return "".join(kept)


def _strip_source_metadata(text: str) -> str:
    text = _SOURCE_SENTENCE_RE.sub("", text)
    lines = text.splitlines()
    kept: list[str] = []
    in_source_section = False

    for line in lines:
        if _SOURCE_SECTION_RE.match(line):
            in_source_section = True
            continue
        if in_source_section:
            if not line.strip() or _SOURCE_BULLET_RE.match(line):
                continue
            in_source_section = False
        if _METADATA_LINE_RE.match(line) or _SOURCE_BULLET_RE.match(line):
            continue
        kept.append(line)

    cleaned = "\n".join(kept)
    cleaned = _PARENTHETICAL_SOURCE_RE.sub("", cleaned)
    cleaned = _PARENTHETICAL_FILE_RE.sub("", cleaned)
    cleaned = _BRACKET_SOURCE_RE.sub("", cleaned)
    cleaned = _ATTRIBUTION_PREFIX_RE.sub("", cleaned)
    cleaned = _FILE_NAME_RE.sub("", cleaned)
    cleaned = _PAGE_ONLY_SENTENCE_RE.sub("", cleaned)
    cleaned = _PAGE_REFERENCE_RE.sub("", cleaned)
    return cleaned


def _protect_calculation_audit(text: str) -> tuple[str, dict[str, str]]:
    protected: dict[str, str] = {}

    def replace(match: re.Match) -> str:
        token = f"CALCULATIONAUDITBLOCK{len(protected)}TOKEN"
        protected[token] = match.group(0)
        return token

    return _CALCULATION_AUDIT_BLOCK_RE.sub(replace, text), protected


def _strip_mixed_refusal(text: str) -> str:
    """Impede recusa global anexada a uma resposta já sustentada."""
    if not _REFUSAL_RE.search(text):
        return text

    without_refusal = _REFUSAL_RE.sub("", text)
    if re.search(r"\w", without_refusal, re.UNICODE):
        return without_refusal
    return REFUSAL_TEXT


def sanitize_answer(answer: str, *, question: str = "") -> str:
    """Aplica política de apresentação ao texto que será exibido no chat."""
    cleaned, protected_audits = _protect_calculation_audit(answer or "")
    cleaned = _strip_internal_details(cleaned)
    if not asks_for_sources(question):
        cleaned = _strip_source_metadata(cleaned)
    cleaned = _strip_mixed_refusal(cleaned)

    cleaned = re.sub(r"\(\s*[,;:-]?\s*\)", "", cleaned)
    cleaned = re.sub(r"[ \t]+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip(" \t\n,;:-")
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    for token, audit in protected_audits.items():
        cleaned = cleaned.replace(token, audit)
    return cleaned or REFUSAL_TEXT
