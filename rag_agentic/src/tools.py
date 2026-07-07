"""
tools — wraps dos retrievers como FunctionTools para o agente ReAct.

Cada tool é uma função Python com docstring clara: o LLM usa a docstring
para decidir quando e como chamar cada ferramenta.
"""
import asyncio
from llama_index.core.tools import FunctionTool
from rag_core.logger import get_logger

log = get_logger(__name__)


def make_retriever_tools(
    text_ret,
    tables_ret,
    ts_ret,
) -> tuple[list, list]:
    """
    Cria as três FunctionTools a partir dos retrievers e retorna
    (tools, source_nodes_collector).

    source_nodes_collector é uma lista mutável preenchida durante
    as chamadas das tools — usada pelo AgenticEngine para rastrear fontes.
    """
    source_nodes: list = []

    def search_narrative(query: str) -> str:
        """
        Busca informações em texto narrativo dos boletins de conjuntura.
        Use para perguntas sobre contexto, análises qualitativas, tendências
        descritas em prosa e indicadores sem estrutura tabular.
        """
        log.info("Tool: search_narrative | query: %s", query[:80])
        nodes = text_ret.retrieve(query)
        if not nodes:
            return "Nenhum trecho narrativo relevante encontrado."
        source_nodes.extend(nodes)
        return "\n\n---\n\n".join(n.get_content() for n in nodes)

    def search_tables(query: str) -> str:
        """
        Busca e extrai dados de tabelas estáticas (rankings, comparações por
        categoria, valores pontuais). Use para perguntas sobre um único período
        ou comparações entre setores/regiões sem série histórica.
        """
        log.info("Tool: search_tables | query: %s", query[:80])
        result = tables_ret.retrieve(query)
        if result is None:
            return "Nenhuma tabela relevante encontrada."
        data, nodes = result
        source_nodes.extend(nodes)
        return data or "Sem dados estruturados extraídos."

    def search_timeseries(query: str) -> str:
        """
        Busca e analisa séries temporais (dados mensais, trimestrais, anuais).
        Use para perguntas sobre evolução, variação, tendência ou comparação
        entre períodos de um mesmo indicador ao longo do tempo.
        """
        log.info("Tool: search_timeseries | query: %s", query[:80])
        result = ts_ret.retrieve(query)
        if result is None:
            return "Nenhuma série temporal relevante encontrada."
        data, nodes = result
        source_nodes.extend(nodes)
        return data or "Sem dados de série temporal extraídos."

    tools = [
        FunctionTool.from_defaults(fn=search_narrative),
        FunctionTool.from_defaults(fn=search_tables),
        FunctionTool.from_defaults(fn=search_timeseries),
    ]
    return tools, source_nodes
