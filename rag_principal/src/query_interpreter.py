"""
Query Interpreter — analisa a pergunta e determina:
  - quais fontes de dados consultar (text / tables / timeseries)
  - versão reescrita da query para melhor recuperação

São duas decisões de naturezas diferentes: escolher as fontes é um julgamento
semântico fechado; reescrever a consulta é geração de texto. Com a Jev
configurada, a primeira passa a ser feita por ela (`jev_interpreter`), tipada e
com incerteza explícita, e o LLM fica só com a reescrita. Sem chave ou em falha,
tudo segue exatamente como antes.
"""
import json

from rag_core.jev import jev_enabled
from rag_core.labor_market_skill import is_labor_market_query
from rag_core.logger import get_logger

from .jev_interpreter import JevSourceSelector

log = get_logger(__name__)

# Um cliente Jev por processo: `interpret_query` roda a cada consulta e o pool de
# conexões do httpx não faz sentido ser recriado a cada chamada.
_jev_selector: JevSourceSelector | None = None

INTERPRET_PROMPT = """\
Você é um roteador de consultas para um sistema RAG de dados econômicos do Estado de São Paulo.

Analise a pergunta e determine:
1. Quais fontes de dados são necessárias para responder
2. Uma versão reescrita da pergunta para maximizar a precisão na recuperação

Fontes disponíveis:
- "text": trechos narrativos dos boletins (análises, comentários, contexto qualitativo)
- "tables": tabelas com dados estáticos ou comparativos (valores pontuais, rankings, comparações entre regiões/setores)
- "timeseries": séries temporais (evolução mensal/trimestral, tendências, crescimento, variação ao longo do tempo)
- "graph": grafo de conhecimento — relações entre indicadores, setores, regiões e fontes de dados

Regras de seleção:
- Inclua "text" para qualquer pergunta que precise de contexto narrativo ou analítico
- Inclua "tables" se a pergunta busca valores específicos, rankings ou comparações pontuais
- Inclua "timeseries" se a pergunta envolve evolução, tendência, crescimento, variação temporal ou sequência de períodos
- Inclua "graph" se a pergunta envolve relações entre múltiplos indicadores, comparações entre setores/regiões, causalidade ou correlação entre variáveis econômicas
- Expanda siglas na reescrita (ex: PIB → Produto Interno Bruto, PNAD, IPCA)
- Seja específico sobre períodos, setores e indicadores na reescrita

Responda SOMENTE com JSON válido (sem markdown, sem texto extra):
{{"sources": ["text"], "rewritten_query": "versão reescrita da pergunta"}}

Pergunta: {question}
"""

_VALID_SOURCES = {"text", "tables", "timeseries", "graph"}


def _source_selector() -> JevSourceSelector | None:
    """Seletor Jev do processo (None quando a Jev não está configurada)."""
    global _jev_selector
    if not jev_enabled():
        return None
    if _jev_selector is None:
        _jev_selector = JevSourceSelector()
    return _jev_selector


def _jev_sources(question: str, selector) -> dict | None:
    """Fontes decididas pela Jev, ou None para manter a decisão anterior."""
    if selector is None:
        return None
    try:
        return selector.select(question)
    except Exception as exc:  # nenhuma falha da Jev pode derrubar a consulta
        log.warning("QueryInterpreter (principal): seleção por Jev falhou: %s", exc)
        return None


def _with_jev(question: str, sources: list[str], selector) -> list[str]:
    """
    Deixa a Jev decidir as fontes; sem resposta dela, mantém a lista recebida.

    Vale inclusive quando o JSON do LLM veio malformado: a lista de fontes é o
    que decide quais retrievers rodam, e a Jev a entrega tipada.
    """
    decision = _jev_sources(question, selector if selector is not None else _source_selector())
    if decision is None:
        return sources
    return decision["sources"]


def interpret_query(question: str, llm, selector=None) -> dict:
    """
    Interpreta a query e retorna:
        {"sources": [...], "rewritten_query": "..."}

    Sempre retorna ao menos "text" em sources como fallback seguro.

    `selector` injeta um seletor Jev (usado nos testes); sem ele, o seletor do
    processo entra quando a Jev está configurada. A reescrita da consulta segue
    sendo do LLM — a Jev decide apenas as fontes.
    """
    try:
        raw = llm.complete(INTERPRET_PROMPT.format(question=question)).text.strip()

        # Remove markdown code fences, se presentes
        raw = raw.strip("` \n")
        if raw.startswith("json"):
            raw = raw[4:].strip()

        result = json.loads(raw)
        sources = [s for s in result.get("sources", []) if s in _VALID_SOURCES]
        if "text" not in sources:
            sources = ["text"] + sources
        return {
            "sources": _with_jev(question, sources or ["text"], selector),
            "rewritten_query": result.get("rewritten_query", question),
            "is_labor_market": is_labor_market_query(question),
        }
    except (json.JSONDecodeError, AttributeError):
        return {
            "sources": _with_jev(question, ["text"], selector),
            "rewritten_query": question,
            "is_labor_market": is_labor_market_query(question),
        }
