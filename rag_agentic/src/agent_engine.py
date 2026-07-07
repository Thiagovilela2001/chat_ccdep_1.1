"""
AgenticEngine — loop agentic com OpenAI function calling nativo.

Substitui FunctionAgent do LlamaIndex (bugado no 0.14 com gpt-5) por
loop manual via openai.AsyncOpenAI, que garante tool calls reais.

Fluxo iterativo (max 8 rounds):
    pergunta
    → LLM recebe tools como OpenAI functions
    → LLM decide chamar tool(s) ou responder
    → se tool: executa retriever, adiciona resultado ao histórico
    → repete até resposta final ou max_iterations
"""
import asyncio
import json
import os

from openai import AsyncOpenAI

from rag_core.llm import openai_client_kwargs

from rag_core.logger import get_logger

log = get_logger(__name__)

MAX_ITERATIONS = 8

_SYSTEM_PROMPT = """\
Você é um analista especialista em dados econômicos e estatísticos do Estado de São Paulo.

As três ferramentas (search_narrative, search_tables, search_timeseries) já
foram chamadas automaticamente com a pergunta original — os resultados estão
no histórico acima.

Regras obrigatórias:
1. Use SOMENTE o que as ferramentas retornaram. Conhecimento externo é proibido.
2. Se os resultados pré-carregados forem insuficientes, chame as ferramentas
   novamente com queries mais específicas antes de responder.
3. Se a informação não foi encontrada em nenhuma ferramenta, responda
   exatamente: 'A informação não consta nos documentos fornecidos.'
4. Toda afirmação factual deve citar a fonte retornada pela ferramenta.
5. Não combine fragmentos de fontes distintas para criar uma afirmação
   que nenhuma fonte expressa diretamente.
6. Se a pergunta pede cálculo e os dois valores estão disponíveis,
   calcule e mostre (ex: 3,4% − 2,8% = 0,6 p.p.).
{skill_block}
Linguagem clara, direta e profissional."""

_SKILL_BLOCK = """\

[Conhecimento Especializado — Mercado de Trabalho]
{skill_context}
[Fim do Conhecimento Especializado]
"""

# Definições das tools no formato OpenAI functions
_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_narrative",
            "description": (
                "Busca informações em texto narrativo dos boletins de conjuntura. "
                "Use para contexto, análises qualitativas, tendências descritas em prosa "
                "e indicadores sem estrutura tabular."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query de busca específica e detalhada.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_tables",
            "description": (
                "Busca e extrai dados de tabelas estáticas (rankings, comparações por "
                "categoria, valores pontuais). Use para um único período ou comparações "
                "entre setores/regiões sem série histórica."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query de busca específica e detalhada.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_timeseries",
            "description": (
                "Busca e analisa séries temporais (dados mensais, trimestrais, anuais). "
                "Use para evolução, variação, tendência ou comparação entre períodos "
                "de um mesmo indicador ao longo do tempo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query de busca específica e detalhada.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


class AgenticEngine:
    """
    Engine agentic com loop manual de function calling OpenAI.
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
        self._model  = getattr(llm, "model", "gpt-5-chat-latest")
        self._labor_skill = labor_market_skill
        self._client = AsyncOpenAI(**openai_client_kwargs())

    def _call_tool(self, name: str, query: str, source_nodes: list) -> str:
        """Executa a tool síncrona e coleta source_nodes."""
        try:
            if name == "search_narrative":
                nodes = self._text.retrieve(query)
                if nodes:
                    source_nodes.extend(nodes)
                    return "\n\n---\n\n".join(n.get_content() for n in nodes)
                return "Nenhum trecho narrativo relevante encontrado."

            elif name == "search_tables":
                result = self._tables.retrieve(query)
                if result is None:
                    return "Nenhuma tabela relevante encontrada."
                data, nodes = result
                source_nodes.extend(nodes)
                return data or "Sem dados estruturados extraídos."

            elif name == "search_timeseries":
                result = self._ts.retrieve(query)
                if result is None:
                    return "Nenhuma série temporal relevante encontrada."
                data, nodes = result
                source_nodes.extend(nodes)
                return data or "Sem dados de série temporal extraídos."

            return f"Ferramenta '{name}' desconhecida."
        except Exception as exc:
            log.warning("Tool '%s' falhou: %s", name, exc)
            return f"[Erro na ferramenta {name}: {exc}]"

    async def answer(
        self,
        question: str,
        sources: list[str],        # mantido por contrato de interface com evaluate.py
        rewritten_query: str,      # agente decide internamente quais tools chamar
        is_labor_market: bool = False,
    ) -> tuple[str, list]:

        skill_block = ""
        if is_labor_market and self._labor_skill and self._labor_skill.is_loaded():
            skill_block = _SKILL_BLOCK.format(
                skill_context=self._labor_skill.get_context()
            )

        system_prompt = _SYSTEM_PROMPT.format(skill_block=skill_block)
        source_nodes: list = []
        answer_text = "A informação não consta nos documentos fornecidos."

        log.info("AgenticEngine: iniciando | question: %s", question[:80])

        # ── Fase 1: pre-fetch paralelo das 3 fontes ───────────────────────────
        # Garante cobertura completa (recall parity com rag_principal) antes
        # do loop agentic de refinamento.
        narrative_r, tables_r, ts_r = await asyncio.gather(
            asyncio.to_thread(self._call_tool, "search_narrative", question, source_nodes),
            asyncio.to_thread(self._call_tool, "search_tables",    question, source_nodes),
            asyncio.to_thread(self._call_tool, "search_timeseries", question, source_nodes),
        )
        log.info(
            "AgenticEngine: pre-fetch concluído | %d source nodes", len(source_nodes)
        )

        # Injeta resultados do pre-fetch como mensagens sintéticas de tool
        # (estrutura válida na OpenAI API — o agente vê o contexto completo)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": question},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "pf_narrative", "type": "function",
                     "function": {"name": "search_narrative",
                                  "arguments": json.dumps({"query": question})}},
                    {"id": "pf_tables", "type": "function",
                     "function": {"name": "search_tables",
                                  "arguments": json.dumps({"query": question})}},
                    {"id": "pf_ts", "type": "function",
                     "function": {"name": "search_timeseries",
                                  "arguments": json.dumps({"query": question})}},
                ],
            },
            {"role": "tool", "tool_call_id": "pf_narrative", "content": narrative_r},
            {"role": "tool", "tool_call_id": "pf_tables",    "content": tables_r},
            {"role": "tool", "tool_call_id": "pf_ts",        "content": ts_r},
        ]

        # ── Fase 2: loop agentic de refinamento ───────────────────────────────
        # O agente pode chamar tools adicionais com queries mais específicas
        # se julgar que o contexto pré-carregado é insuficiente.
        try:
            for iteration in range(MAX_ITERATIONS):
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=_TOOL_DEFINITIONS,
                    tool_choice="auto",
                    timeout=60.0,
                )

                msg = response.choices[0].message

                # Resposta final — sem tool calls
                if not msg.tool_calls:
                    answer_text = msg.content or answer_text
                    log.info("AgenticEngine: resposta final na iteração %d", iteration + 1)
                    break

                # Processa tool calls de refinamento
                messages.append(msg.model_dump(exclude_unset=False))
                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    args = json.loads(tc.function.arguments)
                    query = args.get("query", question)

                    log.info(
                        "AgenticEngine [refinamento it=%d]: tool=%s | query=%s",
                        iteration + 1, tool_name, query[:60],
                    )

                    result = await asyncio.to_thread(
                        self._call_tool, tool_name, query, source_nodes
                    )
                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      result,
                    })

        except Exception as exc:
            log.warning("AgenticEngine: erro: %s", exc)

        # Deduplica source_nodes
        seen: set[int] = set()
        unique_nodes = [
            n for n in source_nodes
            if not (id(n) in seen or seen.add(id(n)))  # type: ignore[func-returns-value]
        ]

        log.info(
            "AgenticEngine: concluído | %d source nodes coletados",
            len(unique_nodes),
        )
        return answer_text.strip(), unique_nodes