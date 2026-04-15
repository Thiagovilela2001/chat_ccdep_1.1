"""
AgenticEngine — agente ReAct que decide iterativamente quais retrievers
chamar para responder a uma pergunta.

Diferença do AnalysisEngine (rag_principal):
  - Roteamento determinístico → decisão iterativa do LLM
  - Chamada única de retrievers → múltiplas chamadas possíveis
  - Síntese em LLM call separado → o próprio agente sintetiza

Fluxo:
    pergunta → ReActAgent (ferramentas: narrative / tables / timeseries)
             → agente chama ferramentas conforme necessário
             → agente decide quando tem contexto suficiente
             → resposta final + source_nodes coletados
"""
import asyncio
from llama_index.core.agent import ReActAgent

from src.tools import make_retriever_tools
from src.logger import get_logger

log = get_logger(__name__)

_SYSTEM_PROMPT = """\
Você é um analista especialista em dados econômicos e estatísticos do Estado de São Paulo.

Regras obrigatórias:
1. Responda SOMENTE com base nas informações retornadas pelas ferramentas.
   Conhecimento externo é proibido.
2. Se a informação não foi encontrada pelas ferramentas, responda exatamente:
   'A informação não consta nos documentos fornecidos.'
3. Toda afirmação factual deve citar a fonte retornada pela ferramenta.
4. Não combine fragmentos de fontes distintas para criar uma afirmação
   que nenhuma fonte expressa diretamente.
5. Se a pergunta pede cálculo e os dois valores estão disponíveis,
   calcule e mostre (ex: 3,4% − 2,8% = 0,6 p.p.).
6. Chame as ferramentas com queries específicas — uma query vaga retorna
   resultados ruins. Se o primeiro resultado for insuficiente, refine a
   query e chame novamente.
{skill_block}
Linguagem clara, direta e profissional."""

_SKILL_BLOCK = """\

[Conhecimento Especializado — Mercado de Trabalho]
{skill_context}
[Fim do Conhecimento Especializado]
"""


class AgenticEngine:
    """
    Engine baseada em agente ReAct com três ferramentas de retrieval.
    Interface idêntica ao AnalysisEngine para compatibilidade com evaluate.py.
    """

    def __init__(
        self,
        text_retriever,
        tables_retriever,
        timeseries_retriever,
        llm,
        labor_market_skill=None,
    ):
        self._text   = text_retriever
        self._tables = tables_retriever
        self._ts     = timeseries_retriever
        self._llm    = llm
        self._labor_skill = labor_market_skill

    async def answer(
        self,
        question: str,
        sources: list[str],        # ignorado — agente decide internamente
        rewritten_query: str,
        is_labor_market: bool = False,
    ) -> tuple[str, list]:
        tools, source_nodes = make_retriever_tools(
            self._text, self._tables, self._ts
        )

        skill_block = ""
        if is_labor_market and self._labor_skill and self._labor_skill.is_loaded():
            skill_block = _SKILL_BLOCK.format(
                skill_context=self._labor_skill.get_context()
            )

        system_prompt = _SYSTEM_PROMPT.format(skill_block=skill_block)

        agent = ReActAgent.from_tools(
            tools,
            llm=self._llm,
            verbose=False,
            system_prompt=system_prompt,
            max_iterations=8,
        )

        log.info("AgenticEngine: iniciando agente | question: %s", question[:80])
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(agent.chat, question),
                timeout=180.0,
            )
            answer_text = response.response
        except asyncio.TimeoutError:
            log.warning("AgenticEngine: timeout após 180s")
            answer_text = "A informação não consta nos documentos fornecidos."

        # Deduplica source_nodes por id de objeto
        seen: set[int] = set()
        unique_nodes: list = []
        for n in source_nodes:
            if id(n) not in seen:
                seen.add(id(n))
                unique_nodes.append(n)

        log.info(
            "AgenticEngine: concluído | %d source nodes coletados",
            len(unique_nodes),
        )
        return answer_text.strip(), unique_nodes
