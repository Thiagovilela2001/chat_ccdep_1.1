"""
Query Interpreter — versão simplificada para o Self-RAG.

Detecta is_labor_market via keywords. O engine decide internamente
se e como buscar.
"""
from src.logger import get_logger

log = get_logger(__name__)

_LABOR_MARKET_KEYWORDS = {
    "emprego", "desemprego", "desocupação", "ocupação", "trabalho",
    "caged", "rais", "pnad", "pnadc", "ped",
    "salário", "remuneração", "rendimento", "renda do trabalho",
    "informalidade", "informal", "carteira assinada", "clt",
    "taxa de desocupação", "saldo de empregos", "rotatividade",
    "mercado de trabalho", "empregado", "desempregado",
}


def interpret_query(question: str, llm) -> dict:
    q_lower = question.lower()
    is_labor_market = any(kw in q_lower for kw in _LABOR_MARKET_KEYWORDS)

    log.info(
        "QueryInterpreter (selfrag) | is_labor_market=%s | query: %s",
        is_labor_market, question[:80],
    )

    return {
        "sources": ["text", "tables", "timeseries"],
        "rewritten_query": question,
        "is_labor_market": is_labor_market,
    }