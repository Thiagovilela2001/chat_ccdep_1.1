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

from rag_core.llm import interp_model, openai_client_kwargs
from rag_core.logger import get_logger

log = get_logger(__name__)

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
Você é um analista especialista em dados econômicos e estatísticos do Estado de São Paulo.

Use SOMENTE o contexto abaixo para responder. Se a informação não constar no contexto,
responda exatamente: 'A informação não consta nos documentos fornecidos.'

Regras obrigatórias:
1. Conhecimento externo é proibido.
2. Cite a fonte de cada afirmação factual.
3. Não combine fragmentos de fontes distintas para criar afirmação não expressa por nenhuma.
4. Se a pergunta pede cálculo e os valores estão disponíveis, calcule e mostre.
{skill_block}
Linguagem clara, direta e profissional."""

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

_SKILL_BLOCK = """\

[Conhecimento Especializado — Mercado de Trabalho]
{skill_context}
[Fim do Conhecimento Especializado]
"""


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
        labor_market_skill=None,
    ):
        self._text         = text_retriever
        self._tables       = tables_retriever
        self._ts           = timeseries_retriever
        self._model        = getattr(llm, "model", "gpt-5-chat-latest")
        self._critic_model = interp_model()
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
            text = resp.choices[0].message.content or "{}"
            # Remove possíveis blocos ```json ... ```
            text = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", text, flags=re.DOTALL).strip()
            return json.loads(text)
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
                    passages.append(n.get_content())
        except Exception as exc:
            log.warning("SelfRAG: text retriever falhou: %s", exc)

        try:
            result = self._tables.retrieve(query)
            if result:
                data, nodes = result
                source_nodes.extend(nodes)
                if data:
                    passages.append(f"[Tabela]\n{data}")
        except Exception as exc:
            log.warning("SelfRAG: tables retriever falhou: %s", exc)

        try:
            result = self._ts.retrieve(query)
            if result:
                data, nodes = result
                source_nodes.extend(nodes)
                if data:
                    passages.append(f"[Série Temporal]\n{data}")
        except Exception as exc:
            log.warning("SelfRAG: timeseries retriever falhou: %s", exc)

        return passages

    # ── Fases Self-RAG ────────────────────────────────────────────────────────

    async def _decide_retrieve(self, question: str) -> bool:
        """RETRIEVE? — decide se busca é necessária para esta pergunta."""
        result = await self._json_call(_RETRIEVE_PROMPT.format(question=question))
        return bool(result.get("retrieve", True))  # default conservador: sempre busca

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
            content = (resp.choices[0].message.content or "").strip()
            return content or "A informação não consta nos documentos fornecidos."
        except Exception as exc:
            log.warning("SelfRAG: generate falhou: %s", exc)
            return "A informação não consta nos documentos fornecidos."

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
        return result.get("query", question)

    # ── Método principal ──────────────────────────────────────────────────────

    async def answer(
        self,
        question: str,
        sources: list[str],        # mantido por contrato com evaluate.py
        rewritten_query: str,
        is_labor_market: bool = False,
    ) -> tuple[str, list]:

        skill_block = ""
        if is_labor_market and self._labor_skill and self._labor_skill.is_loaded():
            skill_block = _SKILL_BLOCK.format(
                skill_context=self._labor_skill.get_context()
            )

        source_nodes: list = []
        log.info("SelfRAGEngine: iniciando | question: %s", question[:80])

        # ── 1. RETRIEVE? ──────────────────────────────────────────────────────
        needs_retrieve = await self._decide_retrieve(question)
        log.info("SelfRAG RETRIEVE?: %s", needs_retrieve)

        if not needs_retrieve:
            log.info("SelfRAG: retrieval desnecessário — recusando")
            return "A informação não consta nos documentos fornecidos.", []

        # ── 2. RETRIEVE ───────────────────────────────────────────────────────
        passages = await asyncio.to_thread(
            self._fetch_all, rewritten_query, source_nodes
        )
        log.info("SelfRAG RETRIEVE: %d passages", len(passages))

        if not passages:
            return "A informação não consta nos documentos fornecidos.", []

        # ── 3. ISREL — filtra relevantes ──────────────────────────────────────
        relevant = await self._filter_relevant(question, passages)
        context = "\n\n---\n\n".join(relevant)

        # ── 4. GENERATE ───────────────────────────────────────────────────────
        answer_text = await self._generate(question, context, skill_block)
        log.info("SelfRAG GENERATE: %d chars gerados", len(answer_text))

        # ── 5. ISSUP — verifica suporte factual ───────────────────────────────
        support = await self._check_support(answer_text, context)
        log.info("SelfRAG ISSUP: support=%s", support)

        # ── 6. RETRY — re-busca se suporte nulo ──────────────────────────────
        if support == "none":
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
                context_retry = "\n\n---\n\n".join(relevant_retry)
                answer_retry = await self._generate(question, context_retry, skill_block)
                support_retry = await self._check_support(answer_retry, context_retry)
                log.info("SelfRAG RETRY ISSUP: support=%s", support_retry)
                answer_text = answer_retry

        # Deduplica source_nodes
        seen: set[int] = set()
        unique_nodes = [
            n for n in source_nodes
            if not (id(n) in seen or seen.add(id(n)))  # type: ignore[func-returns-value]
        ]

        log.info("SelfRAGEngine: concluído | %d source nodes", len(unique_nodes))
        return answer_text.strip(), unique_nodes
