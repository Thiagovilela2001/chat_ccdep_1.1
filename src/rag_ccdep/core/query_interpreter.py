"""Interpretação comum às engines que pesquisam todas as fontes internamente."""
from __future__ import annotations

from .logger import get_logger

log = get_logger(__name__)

_LABOR_MARKET_KEYWORDS = {
    "emprego", "desemprego", "desocupação", "ocupação", "trabalho",
    "caged", "rais", "pnad", "pnadc", "ped",
    "salário", "remuneração", "rendimento", "renda do trabalho",
    "informalidade", "informal", "carteira assinada", "clt",
    "taxa de desocupação", "saldo de empregos", "rotatividade",
    "mercado de trabalho", "empregado", "desempregado",
}


def interpret_all_sources(question: str, _llm=None, *, engine_name: str = "engine") -> dict:
    """Retorna o contrato de interpretação para engines com roteamento interno."""
    lowered = question.lower()
    is_labor_market = any(keyword in lowered for keyword in _LABOR_MARKET_KEYWORDS)
    log.info(
        "QueryInterpreter (%s) | is_labor_market=%s | query: %s",
        engine_name, is_labor_market, question[:80],
    )
    return {
        "sources": ["text", "tables", "timeseries"],
        "rewritten_query": question,
        "is_labor_market": is_labor_market,
    }
