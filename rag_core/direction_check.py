"""
Checagem determinística de direção: "de X para Y" contra o verbo da frase.

O validador numérico confere cada número com as fontes, mas não a palavra que
o interpreta. Esta checagem pega o erro "número certo, leitura errada" mais
comum na avaliação científica: a participação "passou de 48% para 57%" descrita
como "redução". Só sinaliza frases cujas palavras de direção são todas do mesmo
sentido; frases mistas ("caiu X, mas subiu Y") são ambíguas e ficam de fora.

`RAG_DIRECTION_CHECK` (padrão `1`; `0` desliga) controla a revisão em
rag_principal/src/analysis_engine.py.
"""
import os
import re
import unicodedata
from decimal import Decimal, InvalidOperation

_NUM = r"(-?\d{1,3}(?:\.\d{3})+(?:,\d+)?|-?\d+(?:,\d+)?)\s*(%|p\.?\s?p\.?)?(?:\s*(mil|milh(?:ao|oes)|bilh(?:ao|oes)|trilh(?:ao|oes))\b)?"
_TRANSITION = re.compile(rf"\bde\s+(?:r\$\s*)?{_NUM}\s+para\s+(?:r\$\s*)?{_NUM}")
_UP = re.compile(
    r"\b(aument\w*|alta|altas|cresc\w*|eleva(?:cao|ou|ram|r|ndo|da|do)|expan\w*|avanc\w*|"
    r"subi(?:u|ram|r|ndo)|sobe|sobem|ampli\w*|ganh\w*|acrescimo|increment\w*|aceler\w*)\b"
)
_DOWN = re.compile(
    r"\b(redu\w*|reduz\w*|queda|quedas|cai|caiu|cairam|caindo|caem|recu(?:o|ou|aram|ar|ando)|"
    r"retra(?:cao|iu|iram|ir|indo|ida)|diminu\w*|contra(?:cao|iu|iram)|perd\w*|declin\w*|"
    r"decresc\w*|retroc\w*|encolh\w*|desaceler\w*)\b"
)
_SCALE = {"mil": 10**3, "milhao": 10**6, "milhoes": 10**6, "bilhao": 10**9, "bilhoes": 10**9,
          "trilhao": 10**12, "trilhoes": 10**12}
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ])|\n+")


def direction_check_enabled() -> bool:
    return os.getenv("RAG_DIRECTION_CHECK", "1").strip().lower() not in {"", "0", "false", "no", "off"}


def _fold(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in folded if not unicodedata.combining(ch)).lower()


def _value(raw: str) -> Decimal | None:
    try:  # _NUM só aceita ponto como separador de milhar (1.234,5)
        return Decimal(raw.replace(".", "").replace(",", "."))
    except InvalidOperation:
        return None


def _is_year(raw: str, unit: str | None) -> bool:
    return not unit and raw.isdigit() and 1990 <= int(raw) <= 2035


def find_direction_conflicts(text: str) -> list[str]:
    """Frases em que o verbo de direção contradiz uma transição "de X para Y"."""
    conflicts = []
    for sentence in _SENTENCE_SPLIT.split(text or ""):
        folded = _fold(sentence)
        up, down = bool(_UP.search(folded)), bool(_DOWN.search(folded))
        if up == down:  # nenhuma palavra de direção, ou frase mista
            continue
        for m in _TRANSITION.finditer(folded):
            a_raw, a_unit, a_scale, b_raw, b_unit, b_scale = m.groups()
            if _is_year(a_raw, a_unit) or _is_year(b_raw, b_unit):
                continue
            if a_unit and b_unit and a_unit[0] != b_unit[0]:  # % vs p.p.
                continue
            a, b = _value(a_raw), _value(b_raw)
            if a is None or b is None or a < 0 or b < 0:
                continue
            scale = a_scale or b_scale
            a *= _SCALE.get(a_scale or scale or "", 1)
            b *= _SCALE.get(b_scale or scale or "", 1)
            if a == b:
                continue
            if (b > a and down) or (b < a and up):
                conflicts.append(sentence.strip())
                break
    return conflicts


REVISION_INSTRUCTION = """

REVISÃO OBRIGATÓRIA DE DIREÇÃO
Um rascunho anterior desta resposta continha frases em que a palavra de direção
(alta/queda, aumento/redução etc.) contradiz os valores "de X para Y" citados na
mesma frase. Reescreva a resposta completa seguindo todas as regras acima e
corrija a direção dessas frases a partir dos valores; se a frase mistura dois
indicadores, separe-os e descreva cada um no sentido correto.
Frases problemáticas do rascunho:
{conflicts}"""
