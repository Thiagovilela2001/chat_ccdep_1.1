"""
jev_interpreter.py — seleção de fontes do rag_principal via Jev (System One).

O `interpret_query` do principal decide duas coisas de naturezas opostas no mesmo
prompt: (a) quais fontes consultar — julgamento semântico fechado; (b) a
reescrita da consulta — geração de texto. Aqui a primeira vira três perguntas
`Noul` independentes (tabelas, séries, grafo) em **uma única requisição** ao Jev;
a segunda continua no LLM, que é bom nisso.

O que muda em relação ao JSON do prompt único: a resposta vem como uma
probabilidade por fonte (nada para parsear), a lista nunca traz uma fonte
inventada e a incerteza fica explícita — `RAG_JEV_THRESHOLD` decide o corte.

"text" não é perguntado: incluir contexto narrativo é regra determinística do
projeto (`interpret_query` sempre força "text" na lista). Disponibilidade também
não se pergunta: se o grafo está desligado, o `analysis_engine` simplesmente
ignora a fonte.

Sem `TYPESAFE_API_KEY`, com `RAG_JEV_ENABLED=0` ou em qualquer falha, `select()`
devolve None e o `interpret_query` segue com as fontes que o LLM já devolvia.
"""
from __future__ import annotations

from typing import Any

from rag_core.jev import JevClient, JevResponse, jev_enabled, noul
from rag_core.logger import get_logger

log = get_logger(__name__)

AVAILABLE_SOURCES = ("tables", "timeseries", "graph")

_INSTRUCTIONS = {
    "tables": (
        "Responder `pergunta` exige consultar tabelas de dados — valores "
        "pontuais, rankings ou comparações entre regiões e setores?"
    ),
    "timeseries": (
        "Responder `pergunta` exige consultar séries temporais — evolução, "
        "tendência, crescimento ou variação ao longo do tempo?"
    ),
    "graph": (
        "Responder `pergunta` exige consultar o grafo de relações entre "
        "indicadores, setores e regiões?"
    ),
}

# Dizer o que NÃO conta evita inclusão por precaução (a doc recomenda descrever
# os dois lados do sim/não quando a fronteira é discutível).
_CRITERIA = {
    "tables": {
        "true": (
            "a resposta exige valores, rankings ou comparações pontuais que só "
            "aparecem em tabelas"
        ),
        "false": (
            "a resposta pode ser construída apenas com trechos narrativos ou "
            "séries temporais"
        ),
    },
    "timeseries": {
        "true": "a pergunta envolve evolução, tendência, variação ou sequência de períodos",
        "false": "a pergunta trata de um único período ou não pede comportamento ao longo do tempo",
    },
    "graph": {
        "true": (
            "a resposta exige relacionar indicadores, setores ou regiões "
            "(causalidade, correlação, comparação entre setores)"
        ),
        "false": "a resposta se resolve com documentos, tabelas e séries, sem relacionar entidades",
    },
}


def build_state(question: str) -> dict:
    """Estado enviado à Jev: a pergunta em campo próprio, referenciável por `pergunta`."""
    return {
        "pergunta": question,
        "corpus": {
            "fonte": "Boletins de Conjuntura Paulista e Seade Social (Fundação Seade)",
            "tema": "economia e estatísticas do Estado de São Paulo",
            "periodo": "2020 a 2025",
        },
    }


def build_questions() -> dict:
    """Uma pergunta Noul por fonte adicional a `text` — todas na mesma requisição."""
    return {
        source: noul(_INSTRUCTIONS[source], _CRITERIA[source])
        for source in AVAILABLE_SOURCES
    }


class JevSourceSelector:
    """Decide quais fontes, além de `text`, entram na consulta ao engine."""

    def __init__(self, client: Any = None):
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = JevClient()
        return self._client

    def select(self, question: str) -> dict | None:
        """
        Devolve `{"sources": [...], "probabilities": {...}, "confidence": float,
        "reasoning": str}` ou None quando a Jev não responde.

        `text` entra sempre; cada fonte adicional entra quando a probabilidade de
        precisar dela atinge o limiar (`RAG_JEV_THRESHOLD`).
        """
        if self._client is None and not jev_enabled():
            return None

        try:
            response = self._get_client().system_one(build_state(question), build_questions())
        except Exception as exc:  # contrato do cliente é "não lançar"; ainda assim
            log.warning("JevSourceSelector: chamada falhou (%s).", exc)
            return None

        if not isinstance(response, JevResponse):
            return None

        probabilities = {source: response.noul(source) for source in AVAILABLE_SOURCES}
        if any(probability is None for probability in probabilities.values()):
            log.warning("JevSourceSelector: resposta incompleta — mantendo as fontes do LLM.")
            return None

        chosen = [
            source
            for source in AVAILABLE_SOURCES
            if response.noul_decision(source) is True
        ]
        sources = ["text", *chosen]
        confidence = response.weakest_decisiveness(AVAILABLE_SOURCES) or 0.0
        reasoning = "Jev: " + ", ".join(
            f"{source}={probabilities[source]:.2f}{' (sim)' if source in chosen else ' (não)'}"
            for source in AVAILABLE_SOURCES
        )
        log.info("JevSourceSelector: fontes=%s | %s", sources, reasoning)
        return {
            "sources": sources,
            "probabilities": probabilities,
            "confidence": round(confidence, 3),
            "reasoning": reasoning,
        }
