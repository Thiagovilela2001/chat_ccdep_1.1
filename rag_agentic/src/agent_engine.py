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
from rag_core.runtime import bounded_int, limit_chat_messages, limit_context
from rag_core.provenance import format_source_context, source_labels
from rag_core.metrics import record_reported_usage

from rag_core.logger import get_logger

log = get_logger(__name__)

def _max_iterations() -> int:
    return bounded_int("RAG_AGENTIC_MAX_ITERATIONS", 8, 1, 12)


def _max_tool_calls() -> int:
    return bounded_int("RAG_AGENTIC_MAX_TOOL_CALLS", 12, 3, 32)


def _critic_rounds() -> int:
    """Nº máximo de ciclos pesquisa→crítica→pesquisa (env RAG_AGENTIC_CRITIC_ROUNDS).

    Padrão 1 (gera → critica → 1 revisão): equilibra ganho de cobertura e
    latência. Aumente para 2+ para revisões mais profundas ao custo de tempo.
    """
    try:
        configured = int(os.getenv("RAG_AGENTIC_CRITIC_ROUNDS", "1"))
    except ValueError:
        configured = 1
    return min(max(configured, 0), 3)

_SYSTEM_PROMPT = """\
Você é um pesquisador especializado em análise documental de conjuntura econômica
e estatística do Estado de São Paulo. Sua função não é responder rapidamente. Sua
função é produzir a resposta mais completa e fiel possível utilizando
EXCLUSIVAMENTE as informações recuperadas pelo sistema RAG.

FERRAMENTAS DE INVESTIGAÇÃO
Você dispõe de três ferramentas de busca, que pode chamar quantas vezes precisar,
com queries diferentes, para investigar cada aspecto da pergunta:
- search_narrative — texto narrativo (contexto, análises qualitativas, tendências);
- search_tables — tabelas estáticas (rankings, comparações por categoria, valores pontuais);
- search_timeseries — séries temporais (evolução, variação, tendência entre períodos).
As três já foram chamadas automaticamente com a pergunta original; os resultados
estão no histórico acima. Trate isso como ponto de partida, não como o fim da
investigação — use as buscas adicionais para cobrir os aspectos e sinônimos abaixo.

Antes de responder qualquer pergunta, siga OBRIGATORIAMENTE o processo a seguir.

Etapa 1. Compreensão da pergunta
Determine: qual é o fenômeno principal; quais entidades estão envolvidas; qual
período temporal está implícito ou explícito; quais mecanismos econômicos, sociais
ou institucionais podem estar relacionados. Não responda ainda.

Etapa 2. Planejamento da investigação
Transforme a pergunta em um plano de pesquisa. Identifique todos os possíveis
aspectos que podem conter informação relevante (por exemplo, para um choque
econômico: agricultura, indústria, comércio exterior, inflação, mercado de
trabalho, logística, infraestrutura, energia, PIB, produtividade, investimentos,
cadeias produtivas, exportações, importações). Nunca assuma que todas estarão
presentes — são hipóteses de investigação.

Etapa 3. Expansão semântica
Gere consultas alternativas usando sinônimos, conceitos correlatos e termos
técnicos (ex.: "mudanças climáticas" → crise climática → eventos extremos →
secas → enchentes → La Niña → El Niño → choques de oferta → produção agrícola).
Use essas variações para ampliar a recuperação de documentos.

Etapa 4. Investigação iterativa
Realize buscas independentes para cada aspecto identificado. Após cada recuperação:
identifique novos conceitos relevantes; registre quais tópicos já têm evidência;
identifique quais permanecem sem cobertura. Continue investigando enquanto houver
lacuna relevante. NÃO interrompa a pesquisa após a primeira resposta satisfatória.

Etapa 5. Verificação de cobertura
Antes de redigir, confirme que a investigação cobre o maior número possível de
aspectos relevantes. Pergunte internamente: a resposta contempla todos os
mecanismos encontrados nas fontes? considera diferentes documentos? há informação
complementar em outros períodos? há evidência contraditória? Se algum item for
negativo, faça novas buscas.

Etapa 6. Síntese
Não apresente um resumo por documento. Integre todas as evidências em uma única
narrativa. Agrupe informações semelhantes, explique relações de causa e efeito,
mostre como diferentes documentos se complementam e evite repetir a mesma informação.

Etapa 7. Controle de fidelidade
Toda afirmação deve ter suporte explícito nos documentos recuperados. Nunca
complete lacunas com conhecimento próprio, nunca deduza números inexistentes e
nunca apresente relações causais que as fontes não sustentem. Quando houver
incerteza, deixe-a explícita. Os números devem ser transcritos EXATAMENTE como
constam na fonte (mesmos dígitos e formatação) — nunca arredonde nem converta
unidades. Se a informação não foi encontrada em nenhuma ferramenta, responda
exatamente: 'A informação não consta nos documentos fornecidos.'

Etapa 8. Estrutura da resposta
Organize sempre nesta ordem:
1. Resposta direta à pergunta.
2. Explicação dos principais mecanismos encontrados.
3. Evidências documentais organizadas por tema (não por documento).
4. Relação entre os diferentes documentos.
5. Limitações encontradas nas fontes.

Critério de parada
Só finalize a investigação quando: todos os aspectos relevantes tiverem sido
investigados; não surgirem novos conceitos relevantes das buscas; houver
evidência suficiente para uma síntese consistente. Não pare apenas porque
encontrou uma resposta parcial.
{skill_block}
"""

# Segundo agente: avalia a resposta ANTES da entrega. Se houver lacuna, força o
# retorno à investigação (ciclo pesquisa → crítica → pesquisa).
_CRITIC_SYSTEM_PROMPT = """\
Você é um revisor crítico de análises documentais. Avalie, com rigor, se a
RESPOSTA está completa e fiel ao CONTEXTO RECUPERADO — antes de ela ser entregue.

Avalie a resposta contra o contexto e a pergunta, respondendo internamente:
1. A resposta deixou algum aspecto importante sem cobertura?
2. Há evidências no contexto recuperado que não foram utilizadas?
3. A resposta explica apenas um mecanismo ou todos os mecanismos encontrados?
4. Existe alguma afirmação sem suporte no contexto (fato, número ou relação causal)?
5. Há repetições ou informações redundantes?

Se qualquer item indicar problema relevante, a resposta NÃO está aprovada e deve
voltar à investigação. Seja exigente, mas não invente lacunas: se a resposta já
cobre o que o contexto oferece, aprove.

Responda EXCLUSIVAMENTE em JSON válido, sem nenhum texto fora do JSON:
{
  "approved": true,
  "gaps": [],
  "followup_queries": [],
  "unsupported_claims": [],
  "redundancies": []
}
- "gaps": aspectos relevantes ainda sem cobertura.
- "followup_queries": queries de busca específicas para cobrir cada lacuna.
- "unsupported_claims": afirmações sem suporte a remover ou fundamentar.
- "redundancies": informação repetida a enxugar.
Use approved=false sempre que houver item relevante em gaps, unsupported_claims
ou redundancies.
"""

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
                    return "\n\n---\n\n".join(format_source_context(n) for n in nodes)
                return "Nenhum trecho narrativo relevante encontrado."

            elif name == "search_tables":
                result = self._tables.retrieve(query)
                if result is None:
                    return "Nenhuma tabela relevante encontrada."
                data, nodes = result
                source_nodes.extend(nodes)
                return (f"{source_labels(nodes)}\n{data}"
                        if data else "Sem dados estruturados extraídos.")

            elif name == "search_timeseries":
                result = self._ts.retrieve(query)
                if result is None:
                    return "Nenhuma série temporal relevante encontrada."
                data, nodes = result
                source_nodes.extend(nodes)
                return (f"{source_labels(nodes)}\n{data}"
                        if data else "Sem dados de série temporal extraídos.")

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
        tool_budget = {"used": 3, "limit": _max_tool_calls()}

        # ── Fase 2: geração agentic + ciclo de crítica ───────────────────────
        # Gera a resposta; em seguida um segundo agente (crítico) a avalia. Se
        # apontar lacunas, a resposta volta à investigação (pesquisa → crítica →
        # pesquisa) antes da entrega. Bounded por RAG_AGENTIC_CRITIC_ROUNDS.
        answer_text = await self._agentic_generate(
            messages, source_nodes, answer_text, question, tool_budget
        )
        messages.append({"role": "assistant", "content": answer_text})

        for cr in range(_critic_rounds()):
            verdict = await self._run_critic(question, answer_text, messages)
            if verdict.get("approved", True):
                log.info("AgenticEngine: crítica aprovou (rodada %d)", cr + 1)
                break
            log.info(
                "AgenticEngine: crítica reprovou (rodada %d) — gaps=%s",
                cr + 1, (verdict.get("gaps") or [])[:3],
            )
            messages.append({"role": "user", "content": self._critic_feedback(verdict)})
            answer_text = await self._agentic_generate(
                messages, source_nodes, answer_text, question, tool_budget
            )
            messages.append({"role": "assistant", "content": answer_text})

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

    # ── Geração agentic (loop de refinamento com tools) ───────────────────────
    async def _agentic_generate(
        self, messages: list, source_nodes: list, fallback: str, default_query: str,
        tool_budget: dict | None = None,
    ) -> str:
        """Executa o loop de function calling até a resposta final; retorna o texto."""
        answer_text = fallback
        try:
            budget = tool_budget or {"used": 0, "limit": _max_tool_calls()}
            for iteration in range(_max_iterations()):
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=limit_chat_messages(messages),
                    tools=_TOOL_DEFINITIONS,
                    tool_choice="auto",
                    timeout=60.0,
                )
                record_reported_usage("agentic", response)
                msg = response.choices[0].message

                # Resposta final — sem tool calls
                if not msg.tool_calls:
                    answer_text = msg.content or answer_text
                    log.info("AgenticEngine: resposta na iteração %d", iteration + 1)
                    break

                # Processa tool calls de refinamento
                messages.append(msg.model_dump(exclude_unset=False))
                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except (TypeError, json.JSONDecodeError):
                        args = {}
                    query = args.get("query") or default_query
                    log.info(
                        "AgenticEngine [refinamento it=%d]: tool=%s | query=%s",
                        iteration + 1, tool_name, query[:60],
                    )
                    if budget["used"] >= budget["limit"]:
                        result = "[Orçamento de chamadas de ferramenta esgotado.]"
                    else:
                        budget["used"] += 1
                        result = await asyncio.to_thread(
                            self._call_tool, tool_name, query, source_nodes
                        )
                    messages.append({
                        "role": "tool", "tool_call_id": tc.id, "content": result,
                    })
        except Exception as exc:
            log.warning("AgenticEngine: erro na geração: %s", exc)
        return answer_text

    # ── Crítico (segundo agente) ──────────────────────────────────────────────
    @staticmethod
    def _context_digest(messages: list) -> str:
        """Concatena os resultados de tool coletados até agora (truncado)."""
        parts = [
            m.get("content") or ""
            for m in messages
            if isinstance(m, dict) and m.get("role") == "tool"
        ]
        return limit_context("\n\n---\n\n".join(p for p in parts if p))

    async def _run_critic(self, question: str, answer: str, messages: list) -> dict:
        """Avalia a resposta. Fail-open: em erro/JSON inválido, aprova (entrega)."""
        context = self._context_digest(messages)
        user = (
            f"PERGUNTA:\n{question}\n\n"
            f"CONTEXTO RECUPERADO (resultados das ferramentas):\n{context}\n\n"
            f"RESPOSTA A AVALIAR:\n{answer}\n\n"
            "Avalie e responda apenas em JSON."
        )
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _CRITIC_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                timeout=60.0,
            )
            record_reported_usage("agentic", resp)
            return self._parse_json(resp.choices[0].message.content or "")
        except Exception as exc:
            log.warning("AgenticEngine: crítico falhou (%s) — aprovando", exc)
            return {"approved": True}

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Extrai o primeiro objeto JSON do texto. Fail-open para approved=True."""
        try:
            data = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
            return data if isinstance(data, dict) else {"approved": True}
        except (ValueError, json.JSONDecodeError):
            return {"approved": True}

    @staticmethod
    def _critic_feedback(verdict: dict) -> str:
        """Mensagem que devolve o agente à investigação com as lacunas apontadas."""
        def _blk(titulo: str, itens: list) -> str:
            itens = [str(i) for i in (itens or []) if str(i).strip()]
            return f"{titulo}\n" + "\n".join(f"- {i}" for i in itens) + "\n\n" if itens else ""

        corpo = (
            _blk("Aspectos ainda sem cobertura:", verdict.get("gaps"))
            + _blk("Faça estas buscas antes de reescrever:", verdict.get("followup_queries"))
            + _blk("Afirmações sem suporte (remova ou fundamente nas fontes):",
                   verdict.get("unsupported_claims"))
            + _blk("Redundâncias a enxugar:", verdict.get("redundancies"))
        )
        return (
            "[REVISÃO CRÍTICA] Um avaliador identificou problemas na resposta acima. "
            "Antes de reescrever, INVESTIGUE os pontos abaixo chamando as ferramentas "
            "de busca com queries específicas; depois produza uma resposta revisada, "
            "mais completa e fiel, mantendo a estrutura da Etapa 8.\n\n" + corpo
        )
