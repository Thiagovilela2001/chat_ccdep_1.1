"""
RaptorEngine — busca em índice hierárquico RAPTOR + síntese em chamada única.

Diferença em relação aos outros engines:

  rag_principal  → retrievers selecionados pelo interpreter → síntese única
  rag_agentic    → pre-fetch paralelo + loop iterativo de refinamento
  rag_raptor     → índice já contém múltiplos níveis de abstração (folhas +
                   resumos), então uma única busca recupera contexto tanto
                   específico quanto amplo → síntese única sem loop

O ganho vem do índice, não do número de chamadas ao LLM:
  - Perguntas específicas → busca retorna folhas (nível 0)
  - Perguntas amplas      → busca retorna resumos de alto nível
  - Perguntas mistas      → busca retorna ambos os níveis
"""
import asyncio
import os

from openai import AsyncOpenAI

from rag_core.answer_style import ANALYST_WRITING_GUIDE
from rag_core.domain_skills import build_domain_prompt_block
from rag_core.llm import openai_client_kwargs
from rag_core.runtime import limit_context
from rag_core.provenance import format_source_context, source_labels
from rag_core.metrics import record_reported_usage

from rag_core.logger import get_logger

log = get_logger(__name__)


def _raptor_level(value) -> int:
    """Normaliza metadados legados (``"1"``) e atuais (``1``)."""
    try:
        level = int(value)
    except (TypeError, ValueError):
        return 0
    return max(level, 0)

_SYSTEM_PROMPT = """\
Você é um analista de conjuntura econômica e estatística do Estado de São Paulo.
Sua tarefa é redigir uma análise que responda à pergunta do usuário com base
exclusivamente no contexto fornecido.

O contexto abaixo foi recuperado de um índice hierárquico (RAPTOR) que contém tanto
trechos específicos dos documentos (nível 0) quanto resumos de alto nível gerados
automaticamente (níveis superiores). Os resumos ajudam a enxergar a tendência
geral; os trechos originais fornecem os números e as evidências pontuais.

FIDELIDADE ÀS FONTES (inegociável)
1. Use SOMENTE o que está no contexto. Conhecimento externo é proibido.
2. Se a informação não estiver no contexto, responda exatamente:
   'A informação não consta nos documentos fornecidos.'
3. Todo fato e todo número deve ser rastreável ao contexto, com a fonte
   presente nos metadados citada no texto. Organizar, comparar e encadear
   fatos de trechos diferentes em uma mesma narrativa é permitido e esperado;
   criar fato novo, não: nenhuma afirmação causal, estimativa ou conclusão
   que nenhum trecho sustente, direta ou numericamente.
4. Se a pergunta pede cálculo e os dois valores estão disponíveis,
   calcule e mostre a conta (ex: 3,4% − 2,8% = 0,6 p.p.).
{skill_block}
""" + ANALYST_WRITING_GUIDE

class RaptorEngine:
    """
    Engine RAPTOR: recupera de índice hierárquico e sintetiza em chamada única.
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
        self._text   = text_retriever
        self._tables = tables_retriever
        self._ts     = timeseries_retriever
        self._model  = getattr(llm, "model", "gpt-5-chat-latest")
        self._domain_skills = domain_skills
        self._labor_skill = labor_market_skill
        self._client = AsyncOpenAI(**openai_client_kwargs())

    def _retrieve_all(self, query: str, source_nodes: list) -> str:
        """
        Recupera de todas as fontes e monta contexto estratificado por nível RAPTOR.
        Nós de níveis superiores (resumos) aparecem antes dos trechos originais
        para orientar o LLM com contexto amplo antes dos detalhes.
        """
        parts: list[str] = []

        # ── Texto narrativo + nós RAPTOR ──────────────────────────────────────
        try:
            nodes = self._text.retrieve(query)
            if nodes:
                source_nodes.extend(nodes)
                levels: dict[int, list[str]] = {}
                for n in nodes:
                    lvl = _raptor_level(n.metadata.get("raptor_level", 0))
                    levels.setdefault(lvl, []).append(format_source_context(n))

                # Ordena do nível mais alto (visão ampla) para o mais baixo (detalhe)
                for lvl in sorted(levels, reverse=True):
                    if lvl == 0:
                        label = "Trechos originais (nível 0)"
                    else:
                        label = f"Resumos automáticos — nível {lvl}"
                    parts.append(f"[{label}]\n" + "\n\n---\n\n".join(levels[lvl]))
        except Exception as exc:
            log.warning("RaptorEngine: text retriever falhou: %s", exc)

        # ── Tabelas ───────────────────────────────────────────────────────────
        try:
            result = self._tables.retrieve(query)
            if result:
                data, nodes = result
                source_nodes.extend(nodes)
                if data:
                    parts.append(f"[Tabelas]\n{source_labels(nodes)}\n{data}")
        except Exception as exc:
            log.warning("RaptorEngine: tables retriever falhou: %s", exc)

        # ── Séries temporais ──────────────────────────────────────────────────
        try:
            result = self._ts.retrieve(query)
            if result:
                data, nodes = result
                source_nodes.extend(nodes)
                if data:
                    parts.append(f"[Séries Temporais]\n{source_labels(nodes)}\n{data}")
        except Exception as exc:
            log.warning("RaptorEngine: timeseries retriever falhou: %s", exc)

        sep = "\n\n" + ("=" * 60) + "\n\n"
        return sep.join(parts) if parts else ""

    async def answer(
        self,
        question: str,
        sources: list[str],        # mantido por contrato de interface com evaluate.py
        rewritten_query: str,
        is_labor_market: bool = False,
    ) -> tuple[str, list]:

        skill_block = build_domain_prompt_block(
            self._domain_skills,
            question,
            is_labor_market=is_labor_market,
            legacy_labor_skill=self._labor_skill,
        )

        system_prompt = _SYSTEM_PROMPT.format(skill_block=skill_block)
        source_nodes: list = []

        log.info("RaptorEngine: iniciando | question: %s", question[:80])

        context = await asyncio.to_thread(
            self._retrieve_all, rewritten_query, source_nodes
        )
        context = limit_context(context)

        if not context:
            return "A informação não consta nos documentos fornecidos.", []

        user_message = (
            f"CONTEXTO RECUPERADO (índice RAPTOR — múltiplos níveis de abstração):\n"
            f"{context}\n\n"
            f"PERGUNTA: {question}"
        )

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                temperature=0.0,
                timeout=60.0,
            )
            record_reported_usage("raptor", response)
            answer_text = (
                response.choices[0].message.content
                or "A informação não consta nos documentos fornecidos."
            )
        except Exception as exc:
            log.warning("RaptorEngine: erro LLM: %s", exc)
            answer_text = "A informação não consta nos documentos fornecidos."

        # Deduplica source_nodes
        seen: set[int] = set()
        unique_nodes = [
            n for n in source_nodes
            if not (id(n) in seen or seen.add(id(n)))  # type: ignore[func-returns-value]
        ]

        log.info("RaptorEngine: concluído | %d source nodes", len(unique_nodes))
        return answer_text.strip(), unique_nodes
