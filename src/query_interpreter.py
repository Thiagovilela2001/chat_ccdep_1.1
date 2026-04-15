"""
Query Interpreter — analisa a pergunta e determina:
  - quais fontes de dados consultar (text / tables / timeseries)
  - versão reescrita da query para melhor recuperação
"""
import json
from src.labor_market_skill import is_labor_market_query

INTERPRET_PROMPT = """\
Você é um roteador de consultas para um sistema RAG de dados econômicos do Estado de São Paulo.

Analise a pergunta e determine:
1. Quais fontes de dados são necessárias para responder
2. Uma versão reescrita da pergunta para maximizar a precisão na recuperação

Fontes disponíveis:
- "text": trechos narrativos dos boletins (análises, comentários, contexto qualitativo)
- "tables": tabelas com dados estáticos ou comparativos (valores pontuais, rankings, comparações entre regiões/setores)
- "timeseries": séries temporais (evolução mensal/trimestral, tendências, crescimento, variação ao longo do tempo)

Regras de seleção:
- Inclua "text" para qualquer pergunta que precise de contexto narrativo ou analítico
- Inclua "tables" se a pergunta busca valores específicos, rankings ou comparações pontuais
- Inclua "timeseries" se a pergunta envolve evolução, tendência, crescimento, variação temporal ou sequência de períodos
- Expanda siglas na reescrita (ex: PIB → Produto Interno Bruto, PNAD, IPCA)
- Seja específico sobre períodos, setores e indicadores na reescrita

Responda SOMENTE com JSON válido (sem markdown, sem texto extra):
{{"sources": ["text"], "rewritten_query": "versão reescrita da pergunta"}}

Pergunta: {question}
"""

_VALID_SOURCES = {"text", "tables", "timeseries"}


def interpret_query(question: str, llm) -> dict:
    """
    Interpreta a query e retorna:
        {"sources": [...], "rewritten_query": "..."}

    Sempre retorna ao menos "text" em sources como fallback seguro.
    """
    raw = llm.complete(INTERPRET_PROMPT.format(question=question)).text.strip()

    # Remove markdown code fences, se presentes
    raw = raw.strip("` \n")
    if raw.startswith("json"):
        raw = raw[4:].strip()

    try:
        result = json.loads(raw)
        sources = [s for s in result.get("sources", []) if s in _VALID_SOURCES]
        if "text" not in sources:
            sources = ["text"] + sources
        return {
            "sources": sources or ["text"],
            "rewritten_query": result.get("rewritten_query", question),
            "is_labor_market": is_labor_market_query(question),
        }
    except (json.JSONDecodeError, AttributeError):
        return {
            "sources": ["text"],
            "rewritten_query": question,
            "is_labor_market": is_labor_market_query(question),
        }
