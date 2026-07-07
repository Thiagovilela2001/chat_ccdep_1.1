"""
query_analyzer.py — Análise semântica da consulta (antes da recuperação).

Classifica a pergunta em dimensões que caracterizam **a própria consulta**
(nunca o corpus — princípio 2), produzindo o JSON consumido pelo `router`.

A via primária é uma única chamada LLM barata (JSON estrito). Há um fallback
heurístico apenas para resiliência (sem chave/rede/parse) — não é o caminho
principal e não substitui a análise semântica.

Não importa `transformers`/HuggingFace: usa apenas o SDK `openai`.
"""
from __future__ import annotations

import json
import os

from rag_core.llm import interp_model, openai_client_kwargs

_SYSTEM = (
    "Você é um roteador de consultas para um sistema RAG sobre os Boletins de "
    "Conjuntura Paulista (Fundação Seade, dados econômicos de São Paulo, "
    "2020–2025). Analise SEMANTICAMENTE a pergunta e classifique-a. Considere "
    "apenas as características da PERGUNTA, não o conteúdo dos documentos. "
    "Responda EXCLUSIVAMENTE com um objeto JSON válido, sem texto extra."
)

_INSTRUCTION = """Classifique a pergunta abaixo e devolva este JSON:

{{
  "intent": "consulta_dado | comparar | resumir | explicar | verificar",
  "query_type": "pontual | tabular | temporal | ampla | relacional | multi_hop | verificacao",
  "semantic_domain": "emprego | pib | industria | precos | comercio | servicos | geral",
  "specificity": "especifica | intermediaria | ampla",
  "expected_answer": "numerico | tabela | serie | narrativo | comparativo",
  "priority": "precisao | abrangencia",
  "retrieval_need": "lexical | semantica | hibrida",
  "technical_terms": true/false,
  "complexity": "baixa | media | alta",
  "linguistic_patterns": ["curtas descrições dos padrões da formulação"],
  "needs_multi_hop": true/false,
  "is_labor_market": true/false,
  "in_scope": true/false,
  "entities": ["..."],
  "period": "texto ou null",
  "confidence": 0.0-1.0,
  "reasoning": "1 frase"
}}

Diretrizes:
- query_type=pontual/tabular/temporal para dados específicos; ampla/comparativo
  para panoramas/resumos; relacional quando liga entidades; multi_hop quando exige
  encadear várias buscas; verificacao quando pede confirmar/direção de um dado.
- priority=precisao para números/fatos exatos; abrangencia para visão ampla.
- retrieval_need=lexical p/ siglas/termos exatos; semantica p/ conceitos; hibrida
  quando ambos importam.
- in_scope=false se a pergunta for sobre algo fora dos dados econômicos de SP
  (ex.: Selic, outro estado, política nacional).
- is_labor_market=true para emprego, desemprego, CAGED, RAIS, PNAD, rendimento.

Pergunta: {question}
"""

_DEFAULT_MODEL = os.getenv("ORCHESTRATOR_ANALYZER_MODEL") or interp_model()

# Chaves esperadas no resultado (para completar defaults com segurança).
_DEFAULTS = {
    "intent": "consulta_dado",
    "query_type": "pontual",
    "semantic_domain": "geral",
    "specificity": "intermediaria",
    "expected_answer": "narrativo",
    "priority": "precisao",
    "retrieval_need": "hibrida",
    "technical_terms": False,
    "complexity": "media",
    "linguistic_patterns": [],
    "needs_multi_hop": False,
    "is_labor_market": False,
    "in_scope": True,
    "entities": [],
    "period": None,
    "confidence": 0.5,
    "reasoning": "",
}


class QueryAnalyzer:
    """Classificador semântico de consultas. `client` injetável para testes."""

    def __init__(self, client=None, model: str = _DEFAULT_MODEL):
        self.model = model
        self._client = client  # openai.OpenAI-compatível; criado sob demanda

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI  # import tardio: evita custo se nunca usado
            self._client = OpenAI(**openai_client_kwargs())
        return self._client

    def analyze(self, question: str) -> dict:
        """Retorna o dict de classificação (sempre com todas as chaves)."""
        try:
            client = self._get_client()
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": _INSTRUCTION.format(question=question)},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
            data = json.loads(raw)
            return _merge_defaults(data)
        except Exception:
            # Fallback resiliente — mantém o sistema no ar mesmo sem LLM.
            return _heuristic_fallback(question)


def _merge_defaults(data: dict) -> dict:
    out = dict(_DEFAULTS)
    if isinstance(data, dict):
        out.update({k: v for k, v in data.items() if k in _DEFAULTS})
    try:
        out["confidence"] = float(out["confidence"])
    except (TypeError, ValueError):
        out["confidence"] = 0.5
    return out


def _heuristic_fallback(question: str) -> dict:
    """Classificação mínima e conservadora quando o LLM não está disponível."""
    out = dict(_DEFAULTS)
    q = question.lower()
    if any(t in q for t in ("panorama", "resumo", "resuma", "evolução", "visão geral")):
        out.update(query_type="ampla", priority="abrangencia", specificity="ampla",
                   expected_answer="narrativo")
    elif any(t in q for t in ("relaç", "impacto", "influ", "correlaç")):
        out.update(query_type="multi_hop", needs_multi_hop=True, complexity="alta")
    out["confidence"] = 0.3  # baixa: sinaliza que foi fallback, não análise real
    out["reasoning"] = "fallback heurístico (LLM indisponível)"
    return out
