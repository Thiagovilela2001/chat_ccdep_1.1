"""Cálculo determinístico de indicadores de estrutura etária."""
from __future__ import annotations

import math
import re
import unicodedata
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

import pandas as pd


class DemographicCalculationError(ValueError):
    """Dados ausentes, ambíguos ou incompatíveis com as faixas exigidas."""


DEMOGRAPHIC_EXTRACTION_RULES = """
REGRA ESPECIAL PARA INDICADORES DEMOGRÁFICOS:
- Extraia somente população absoluta, nunca percentuais.
- Retorne obrigatoriamente uma tabela com estas colunas exatas:
  ["idade_inicial", "idade_final", "populacao", "territorio", "periodo", "unidade", "fonte", "pagina"]
- Cada linha deve representar uma faixa etária encontrada na mesma fonte.
- Use limites inclusivos. Exemplo: 0 a 4 anos = idade_inicial 0 e idade_final 4.
- Para a última faixa aberta, use null em idade_final. Exemplo: 80+ = [80, null].
- Preserve as faixas da fonte. Não divida, estime, interpole ou combine faixas.
- Preserve território, período e unidade documentados em todas as linhas.
- Copie `fonte` e `pagina` dos rótulos de proveniência do trecho. Nunca combine
  faixas de documentos diferentes na mesma linha ou altere o nome da fonte.
- Se a fonte não trouxer algum desses campos, não o invente: use null.
""".strip()

_REQUIRED_COLUMNS = {
    "idade_inicial",
    "idade_final",
    "populacao",
    "territorio",
    "periodo",
    "unidade",
    "fonte",
    "pagina",
}
_CURRENT_QUESTION_RE = re.compile(
    r"<PERGUNTA_ATUAL>\s*(.*?)\s*</PERGUNTA_ATUAL>", re.DOTALL | re.IGNORECASE
)


def _normalize(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value).casefold())
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _current_question(question: str) -> str:
    match = _CURRENT_QUESTION_RE.search(question or "")
    return match.group(1) if match else question


def is_demographic_indicator_query(question: str) -> bool:
    normalized = _normalize(_current_question(question))
    return (
        "indice_de_envelhecimento" in normalized
        or "razao_de_dependencia" in normalized
        or "razoes_de_dependencia" in normalized
        or "dependencia_jovem" in normalized
        or "dependencia_idosa" in normalized
    )


def demographic_failure(reason: object) -> str:
    return (
        "[Cálculo Demográfico Determinístico]\n"
        "Status: não realizado.\n"
        f"Motivo: {reason}.\n"
        "[Fim do Cálculo Demográfico Determinístico]"
    )


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _integer_age(value: object, *, open_end: bool = False) -> int | None:
    if _is_missing(value) or (isinstance(value, str) and not value.strip()):
        if open_end:
            return None
        raise DemographicCalculationError("faixa etária sem idade inicial")
    try:
        number = Decimal(str(value).strip().replace(",", "."))
    except InvalidOperation as exc:
        raise DemographicCalculationError(f"idade inválida: {value!r}") from exc
    if number != number.to_integral_value() or number < 0 or number > 150:
        raise DemographicCalculationError(f"idade fora do intervalo válido: {value!r}")
    return int(number)


def _population(value: object) -> Decimal:
    if _is_missing(value):
        raise DemographicCalculationError("população ausente")
    if isinstance(value, bool):
        raise DemographicCalculationError("população inválida")
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise DemographicCalculationError("população inválida")
        number = Decimal(str(value))
    else:
        text = str(value).strip().replace("\xa0", "").replace(" ", "")
        if text.endswith("%"):
            raise DemographicCalculationError("percentual não pode substituir população")
        if "," in text:
            text = text.replace(".", "").replace(",", ".")
        elif re.fullmatch(r"[-+]?\d{1,3}(?:\.\d{3})+", text):
            text = text.replace(".", "")
        try:
            number = Decimal(text)
        except InvalidOperation as exc:
            raise DemographicCalculationError(f"população inválida: {value!r}") from exc
    if not number.is_finite() or number < 0:
        raise DemographicCalculationError("população deve ser não negativa")
    return number


def _label(value: object, field: str) -> str:
    if _is_missing(value) or not str(value).strip():
        raise DemographicCalculationError(f"{field} ausente")
    return str(value).strip()


def _display_label(value: str) -> str:
    try:
        number = Decimal(value)
    except InvalidOperation:
        return value
    if number == number.to_integral_value():
        return str(int(number))
    return value


def _format_pt(value: Decimal, decimals: int | None = None) -> str:
    if decimals is None:
        decimals = 0 if value == value.to_integral_value() else 2
    quantizer = Decimal(1).scaleb(-decimals)
    formatted = f"{value.quantize(quantizer, rounding=ROUND_HALF_UP):,.{decimals}f}"
    return formatted.replace(",", "_").replace(".", ",").replace("_", ".")


def _ratio(numerator: Decimal, denominator: Decimal) -> Decimal:
    if denominator == 0:
        raise DemographicCalculationError("denominador populacional igual a zero")
    return numerator / denominator * Decimal(100)


def _group_components(rows: list[tuple[int, int | None, Decimal]]) -> tuple[Decimal, Decimal, Decimal]:
    ordered = sorted(rows, key=lambda row: row[0])
    if not ordered or ordered[0][0] != 0:
        raise DemographicCalculationError("faixas etárias não começam em 0")

    expected_start = 0
    open_range_seen = False
    young = Decimal(0)
    working_age = Decimal(0)
    elderly = Decimal(0)

    for start, end, population in ordered:
        if open_range_seen or start != expected_start:
            raise DemographicCalculationError("faixas etárias possuem lacuna ou sobreposição")
        if end is not None and end < start:
            raise DemographicCalculationError("faixa etária possui limites invertidos")

        effective_end = end if end is not None else 150
        if 0 <= start and effective_end <= 14:
            young += population
        elif 15 <= start and effective_end <= 64:
            working_age += population
        elif start >= 65:
            elderly += population
        else:
            raise DemographicCalculationError(
                f"faixa {start}{'+' if end is None else f'–{end}'} cruza os limites 0–14, 15–64 ou 65+"
            )

        if end is None:
            open_range_seen = True
        else:
            expected_start = end + 1

    if not open_range_seen:
        raise DemographicCalculationError("faixa idosa final não é aberta")
    if young == 0 or working_age == 0:
        raise DemographicCalculationError("componentes 0–14 ou 15–64 têm população zero")
    return young, working_age, elderly


def calculate_demographic_indicators(question: str, frame: pd.DataFrame | None) -> str:
    """Valida faixas, agrega populações e produz contas auditáveis."""
    if frame is None or frame.empty:
        raise DemographicCalculationError("tabela etária estruturada ausente")

    columns = {_normalize(column): column for column in frame.columns}
    missing = sorted(_REQUIRED_COLUMNS - set(columns))
    if missing:
        raise DemographicCalculationError("colunas ausentes: " + ", ".join(missing))

    groups: dict[tuple[str, str, str, str], list[tuple[int, int | None, Decimal]]] = {}
    pages_by_group: dict[tuple[str, str, str, str], set[str]] = {}
    source_territory_period_units: set[tuple[str, str, str]] = set()
    for _, row in frame.iterrows():
        territory = _label(row[columns["territorio"]], "território")
        period = _display_label(_label(row[columns["periodo"]], "período"))
        unit = _label(row[columns["unidade"]], "unidade")
        source = _label(row[columns["fonte"]], "fonte")
        page = _label(row[columns["pagina"]], "página/aba")
        if "%" in unit or "percent" in _normalize(unit):
            raise DemographicCalculationError("unidade percentual incompatível com população")
        start = _integer_age(row[columns["idade_inicial"]])
        if start is None:  # garantia adicional para tipagem; _integer_age já rejeita
            raise DemographicCalculationError("faixa etária sem idade inicial")
        end = _integer_age(row[columns["idade_final"]], open_end=True)
        population = _population(row[columns["populacao"]])
        key = (source, territory, period, unit)
        groups.setdefault(key, []).append((start, end, population))
        pages_by_group.setdefault(key, set()).add(page)
        source_territory_period_units.add(
            (source.casefold(), territory.casefold(), period.casefold())
        )

    if len(groups) != len(source_territory_period_units):
        raise DemographicCalculationError(
            "mesma fonte, território e período aparecem com unidades diferentes"
        )

    current = _normalize(_current_question(question))
    wants_aging = "envelhecimento" in current
    wants_dependency = "dependencia" in current
    sections = ["[Cálculo Demográfico Determinístico]"]

    for key, rows in sorted(groups.items()):
        source, territory, period, unit = key
        pages = sorted(pages_by_group[key])
        young, working_age, elderly = _group_components(rows)
        sections.extend(
            (
                f"Fonte documental: {source}; página/aba: {', '.join(pages)}",
                f"Território: {territory}",
                f"Período: {period}",
                "Faixas compatíveis: 0–14, 15–64 e 65+",
                f"Componentes ({unit}): P(0–14)={_format_pt(young)}; "
                f"P(15–64)={_format_pt(working_age)}; P(65+)={_format_pt(elderly)}.",
            )
        )
        if wants_aging:
            result = _ratio(elderly, young)
            sections.append(
                "Índice de envelhecimento: "
                f"{_format_pt(elderly)} / {_format_pt(young)} × 100 = {_format_pt(result, 2)}."
            )
        if wants_dependency:
            young_ratio = _ratio(young, working_age)
            elderly_ratio = _ratio(elderly, working_age)
            total = young + elderly
            total_ratio = _ratio(total, working_age)
            sections.extend(
                (
                    "Razão de dependência jovem: "
                    f"{_format_pt(young)} / {_format_pt(working_age)} × 100 = "
                    f"{_format_pt(young_ratio, 2)}.",
                    "Razão de dependência idosa: "
                    f"{_format_pt(elderly)} / {_format_pt(working_age)} × 100 = "
                    f"{_format_pt(elderly_ratio, 2)}.",
                    "Razão de dependência total: "
                    f"({_format_pt(young)} + {_format_pt(elderly)}) / "
                    f"{_format_pt(working_age)} × 100 = {_format_pt(total_ratio, 2)}.",
                )
            )
        sections.append("")

    sections.append(
        "Interpretação: resultados por 100 pessoas de 15–64 anos; "
        "razões etárias não medem dependência econômica observada."
    )
    sections.append("[Fim do Cálculo Demográfico Determinístico]")
    return "\n".join(sections)
