"""
Query Interpreter — versão simplificada para o RAPTOR RAG.

No rag_principal, o interpreter roteia para fontes específicas.
Aqui, o engine decide internamente — o interpreter apenas:
  1. Detecta is_labor_market para injetar a skill no prompt do engine.
  2. Retorna sources=["text","tables","timeseries"] (todas — engine filtra).
  3. Retorna a query original como rewritten_query.
"""
from rag_core.logger import get_logger

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
    """
    Retorna dict compatível com evaluate.py:
        sources         : lista de fontes (todas — engine decide)
        rewritten_query : query original
        is_labor_market : bool
    """
    q_lower = question.lower()
    is_labor_market = any(kw in q_lower for kw in _LABOR_MARKET_KEYWORDS)

    log.info(
        "QueryInterpreter (raptor) | is_labor_market=%s | query: %s",
        is_labor_market, question[:80],
    )

    return {
        "sources": ["text", "tables", "timeseries"],
        "rewritten_query": question,
        "is_labor_market": is_labor_market,
    }
