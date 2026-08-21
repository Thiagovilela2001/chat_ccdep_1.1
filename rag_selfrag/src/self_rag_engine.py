"""
SelfRAGEngine — Self-Reflective RAG com critério adaptativo de retrieval.

Implementação baseada em Asai et al. (2023), adaptada para LLMs off-the-shelf
(sem fine-tuning nos reflection tokens do paper original). Os "special tokens"
do paper são substituídos por chamadas LLM explícitas ao modelo de crítica.

Fluxo:
  1. RETRIEVE?  LLM decide se busca em documentos é necessária.
  2. RETRIEVE   Busca nas 3 fontes em paralelo (se necessário).
  3. ISREL      LLM filtra passages irrelevantes em batch.
  4. GENERATE   LLM principal gera resposta apenas com passages relevantes.
  5. ISSUP      LLM verifica se a resposta é suportada factualmente.
  6. RETRY      Se support="none", refina a query e re-busca uma vez.

Modelo de crítica (gpt-5-mini): RETRIEVE?, ISREL, ISSUP, refinamento de query.
Modelo principal (gpt-5-chat-latest): GENERATE apenas.
"""
import asyncio
import json
import os
import re

from openai import AsyncOpenAI

from rag_core.answer_policy import REFUSAL_TEXT, sanitize_answer
from rag_core.answer_style import ANALYST_WRITING_GUIDE
from rag_core.domain_skills import build_domain_prompt_block
from rag_core.llm import interp_model, openai_client_kwargs
from rag_core.runtime import bounded_int, limit_context
from rag_core.provenance import format_source_context, source_labels
from rag_core.metrics import record_reported_usage
from rag_core.logger import get_logger

log = get_logger(__name__)


def _max_retries() -> int:
    return bounded_int("RAG_SELFRAG_MAX_RETRIES", 1, 0, 2)

# ── Prompts de critique ───────────────────────────────────────────────────────

_RETRIEVE_PROMPT = """\
Você deve decidir se uma busca em documentos econômicos e estatísticos do \
Estado de São Paulo é necessária para responder a esta pergunta.

Responda APENAS com JSON: {{"retrieve": true}} ou {{"retrieve": false}}

PADRÃO: retrieve=true. Use retrieve=false SOMENTE nos casos abaixo:
- A pergunta pede opinião pessoal ou filosófica sem base em dados (ex: "o que você acha de...")
- A pergunta é sobre culinária, esportes, entretenimento ou outro tema sem relação com economia/estatística

Qualquer pergunta sobre conjuntura, indicadores, PIB, emprego, inflação, setores econômicos,
dados populacionais ou análise de períodos históricos DEVE ter retrieve=true.

Pergunta: {question}"""

_ISREL_PROMPT = """\
Avalie a relevância de cada trecho numerado abaixo para responder à pergunta.
Retorne APENAS um JSON com os índices (0-based) dos trechos RELEVANTES.
Exemplo: {{"relevant": [0, 2, 3]}}
Se todos forem relevantes: {{"relevant": [0, 1, 2, ...]}}
Se nenhum for relevante: {{"relevant": []}}

Pergunta: {question}

Trechos:
{passages}"""

_GENERATE_SYSTEM = """\
Você é um analista de conjuntura econômica e estatística do Estado de São Paulo.
Sua tarefa é redigir uma análise que responda à pergunta do usuário com base
exclusivamente no contexto fornecido.

Use SOMENTE o contexto abaixo para responder. Use exatamente
'""" + REFUSAL_TEXT + """' somente quando nenhum ponto central da pergunta estiver
sustentado. Nunca anexe essa mensagem a uma resposta factual já sustentada.

FIDELIDADE ÀS FONTES (inegociável)
1. Conhecimento externo é proibido.
2. Todo fato e todo número deve ser rastreável ao contexto. Use rótulos de
   origem apenas para verificação interna; nunca os copie para a resposta.
   Organizar, comparar e encadear fatos de
   trechos diferentes em uma mesma narrativa é permitido e esperado; criar
   fato novo, não: nenhuma afirmação causal, estimativa ou conclusão que
   nenhum trecho sustente, direta ou numericamente.
3. Se a pergunta pede cálculo e os valores estão disponíveis, calcule e
   mostre a conta (ex: 3,4% − 2,8% = 0,6 p.p.).
4. Para índice de envelhecimento e razões de dependência, use exclusivamente
   [Cálculo Demográfico Determinístico] retornado pelas tabelas. Não recalcule,
   estime ou combine faixas. Se o cálculo foi bloqueado, explique o motivo.
5. Se período, território, indicador ou faixa estiver ambíguo, não escolha um
   recorte nem calcule. Faça uma única pergunta objetiva de esclarecimento.
6. Se faltar dado de um período pedido para cálculo ou comparação, responda somente
   com uma frase direta, "Dado encontrado" (período disponível, sem valor), "Dado
   ausente" e "Operação cancelada por ausência de dados na fonte." Não cite valores
   nem períodos diferentes dos solicitados.
{skill_block}
""" + ANALYST_WRITING_GUIDE

_ISSUP_PROMPT = """\
A resposta abaixo é suportada pelo contexto fornecido?

Responda APENAS com JSON:
{{"support": "full"}}     → totalmente suportada pelos documentos
{{"support": "partial"}}  → parcialmente suportada (alguns dados confirmados)
{{"support": "none"}}     → não suportada ou é uma resposta de recusa

Contexto (primeiros 3000 chars):
{context}

Resposta gerada:
{answer}"""

_REFINE_PROMPT = """\
A resposta gerada não foi suficientemente suportada pelos documentos recuperados.
Com base na pergunta original e na resposta parcial, escreva uma query de busca \
mais específica para encontrar as informações que faltam.

Responda APENAS com JSON: {{"query": "..."}}

Pergunta original: {question}
Resposta parcial (primeiros 300 chars): {answer}"""

# ── Engine ────────────────────────────────────────────────────────────────────

class SelfRAGEngine:
    """
    Self-RAG: critério adaptativo de retrieval + auto-avaliação de suporte.
    Interface idêntica ao AgenticEngine para compatibilidade com evaluate.py.
    """

    def __init__(
        self,
        text_retriever,
        tables_retriever,
        timeseries_retriever,
        llm,
        domain_skills=None,
        labor_market_skill=None,
    ):
        self._text         = text_retriever
        self._tables       = tables_retriever
        self._ts           = timeseries_retriever
        self._model        = getattr(llm, "model", "gpt-5-chat-latest")
        self._critic_model = interp_model()
        self._domain_skills = domain_skills
        self._labor_skill  = labor_market_skill
        self._client       = AsyncOpenAI(**openai_client_kwargs())

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _json_call(self, prompt: str, model: str | None = None) -> dict:
        """Chamada LLM que retorna JSON. Remove markdown code fences se presentes."""
        try:
            resp = await self._client.chat.completions.create(
                model=model or self._critic_model,
                messages=[{"role": "user", "content": prompt}],
                timeout=30.0,
            )
            record_reported_usage("selfrag", resp)
            text = resp.choices[0].message.content or "{}"
            # Remove possíveis blocos ```json ... ```
            text = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", text, flags=re.DOTALL).strip()
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            log.warning("SelfRAG: critique call falhou: %s", exc)
            return {}

    def _fetch_all(self, query: str, source_nodes: list) -> list[str]:
        """Busca nas 3 fontes e retorna lista de passages (strings)."""
        passages: list[str] = []

        try:
            nodes = self._text.retrieve(query)
            if nodes:
                source_nodes.extend(nodes)
                for n in nodes:
                    passages.append(format_source_context(n))
        except Exception as exc:
            log.warning("SelfRAG: text retriever falhou: %s", exc)

        try:
            result = self._tables.retrieve(query)
            if result:
                data, nodes = result
                source_nodes.extend(nodes)
                if data:
                    passages.append(f"[Tabela]\n{source_labels(nodes)}\n{data}")
        except Exception as exc:
            log.warning("SelfRAG: tables retriever falhou: %s", exc)

        try:
            result = self._ts.retrieve(query)
            if result:
                data, nodes = result
                source_nodes.extend(nodes)
                if data:
                    passages.append(f"[Série Temporal]\n{source_labels(nodes)}\n{data}")
        except Exception as exc:
            log.warning("SelfRAG: timeseries retriever falhou: %s", exc)

        return passages

    # ── Fases Self-RAG ────────────────────────────────────────────────────────

    async def _decide_retrieve(self, question: str) -> bool:
        """RETRIEVE? — decide se busca é necessária para esta pergunta."""
        result = await self._json_call(_RETRIEVE_PROMPT.format(question=question))
        decision = result.get("retrieve", True)
        return decision if isinstance(decision, bool) else True

    async def _filter_relevant(self, question: str, passages: list[str]) -> list[str]:
        """ISREL — filtra passages não-relevantes em batch (uma única chamada LLM)."""
        if not passages:
            return []

        numbered = "\n\n".join(f"[{i}] {p[:500]}" for i, p in enumerate(passages))
        result = await self._json_call(
            _ISREL_PROMPT.format(question=question, passages=numbered)
        )

        indices = result.get("relevant", None)
        if not isinstance(indices, list):
            return passages  # fallback: usa todos

        relevant = [passages[i] for i in indices if isinstance(i, int) and 0 <= i < len(passages)]
        log.info("SelfRAG ISREL: %d/%d passages relevantes", len(relevant), len(passages))
        return relevant if relevant else passages  # fallback: usa todos se vazio

    async def _generate(self, question: str, context: str, skill_block: str) -> str:
        """GENERATE — gera resposta com o modelo principal."""
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": _GENERATE_SYSTEM.format(skill_block=skill_block),
                    },
                    {
                        "role": "user",
                        "content": f"CONTEXTO:\n{context}\n\nPERGUNTA: {question}",
                    },
                ],
                temperature=0.0,
                timeout=60.0,
            )
            record_reported_usage("selfrag", resp)
            content = (resp.choices[0].message.content or "").strip()
            return content or REFUSAL_TEXT
        except Exception as exc:
            log.warning("SelfRAG: generate falhou: %s", exc)
            return REFUSAL_TEXT

    async def _check_support(self, answer: str, context: str) -> str:
        """ISSUP — retorna 'full', 'partial' ou 'none'."""
        result = await self._json_call(
            _ISSUP_PROMPT.format(context=context[:3000], answer=answer[:1000])
        )
        support = result.get("support", "full")
        return support if support in {"full", "partial", "none"} else "full"

    async def _refine_query(self, question: str, answer: str) -> str:
        """Gera query refinada para re-busca quando suporte é insuficiente."""
        result = await self._json_call(
            _REFINE_PROMPT.format(question=question, answer=answer[:300])
        )
        query = result.get("query", question)
        return query.strip() if isinstance(query, str) and query.strip() else question

    # ── Método principal ──────────────────────────────────────────────────────

    async def answer(
        self,
        question: str,
        sources: list[str],        # mantido por contrato com evaluate.py
        rewritten_query: str,
        is_labor_market: bool = False,
    ) -> tuple[str, list]:

        skill_block = build_domain_prompt_block(
            self._domain_skills,
            question,
            is_labor_market=is_labor_market,
            legacy_labor_skill=self._labor_skill,
        )

        source_nodes: list = []
        log.info("SelfRAGEngine: iniciando | question: %s", question[:80])

        # ── 1. RETRIEVE? ──────────────────────────────────────────────────────
        needs_retrieve = await self._decide_retrieve(question)
        log.info("SelfRAG RETRIEVE?: %s", needs_retrieve)

        if not needs_retrieve:
            log.info("SelfRAG: retrieval desnecessário — recusando")
            return REFUSAL_TEXT, []

        # ── 2. RETRIEVE ───────────────────────────────────────────────────────
        passages = await asyncio.to_thread(
            self._fetch_all, rewritten_query, source_nodes
        )
        log.info("SelfRAG RETRIEVE: %d passages", len(passages))

        if not passages:
            return REFUSAL_TEXT, []

        # ── 3. ISREL — filtra relevantes ──────────────────────────────────────
        relevant = await self._filter_relevant(question, passages)
        context = limit_context("\n\n---\n\n".join(relevant))

        # ── 4. GENERATE ───────────────────────────────────────────────────────
        answer_text = await self._generate(question, context, skill_block)
        log.info("SelfRAG GENERATE: %d chars gerados", len(answer_text))

        # ── 5. ISSUP — verifica suporte factual ───────────────────────────────
        support = await self._check_support(answer_text, context)
        log.info("SelfRAG ISSUP: support=%s", support)

        # ── 6. RETRY — re-busca se suporte nulo ──────────────────────────────
        for retry in range(_max_retries()):
            if support != "none":
                break
            log.info("SelfRAG RETRY: suporte insuficiente — refinando query")
            refined_query = await self._refine_query(question, answer_text)
            log.info("SelfRAG RETRY: query=%s", refined_query[:80])

            retry_nodes: list = []
            retry_passages = await asyncio.to_thread(
                self._fetch_all, refined_query, retry_nodes
            )
            source_nodes.extend(retry_nodes)

            if retry_passages:
                relevant_retry = await self._filter_relevant(question, retry_passages)
                context_retry = limit_context("\n\n---\n\n".join(relevant_retry))
                answer_retry = await self._generate(question, context_retry, skill_block)
                support_retry = await self._check_support(answer_retry, context_retry)
                log.info(
                    "SelfRAG RETRY %d/%d ISSUP: support=%s",
                    retry + 1, _max_retries(), support_retry,
                )
                answer_text = answer_retry
                support = support_retry
            else:
                break

        # Deduplica source_nodes
        seen: set[int] = set()
        unique_nodes = [
            n for n in source_nodes
            if not (id(n) in seen or seen.add(id(n)))  # type: ignore[func-returns-value]
        ]

        log.info("SelfRAGEngine: concluído | %d source nodes", len(unique_nodes))
        return sanitize_answer(answer_text, question=question), unique_nodes
