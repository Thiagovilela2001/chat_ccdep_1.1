"""
router.py — Política de seleção da estratégia (Single Best Routing).

Recebe a classificação semântica da consulta (dict do `query_analyzer`) e decide
**qual engine** executa, pontuando cada `StrategyProfile` por compatibilidade com
as características da pergunta. É uma função **pura** (sem rede, sem LLM), portanto
totalmente testável isoladamente.

Princípios atendidos:
- 1 (Single Best): por padrão devolve **uma** engine; multi-engine só quando
  habilitado E a decisão é ambígua/baixa confiança.
- 2 (só a consulta): pontua apenas dimensões da pergunta, nunca o corpus.
- 4 (extensibilidade): o mapa de roteamento é derivado dos perfis — registrar um
  perfil novo estende o roteador automaticamente.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.registry import StrategyProfile, get_profiles

# Limiares (configuráveis) ------------------------------------------------------
TAU_HIGH = 0.75   # acima disto: single-best com alta confiança
TAU_LOW = 0.45    # abaixo disto: consulta considerada de baixa confiança
TIE_MARGIN = 1.0  # diferença mínima de score entre 1º e 2º p/ não ser empate


@dataclass
class RouteDecision:
    engines: list[str]                 # 1 (single-best) ou 2 (multi/desempate)
    mode: str                          # "single_best" | "multi" | "refuse"
    confidence: float
    query_type: str
    scores: dict[str, float]
    reasoning: str
    refused: bool = False


def _score(profile: StrategyProfile, cls: dict) -> float:
    """Compatibilidade entre um perfil de estratégia e a classificação da query."""
    s = 0.0
    if cls.get("query_type") in profile.good_for:
        s += 3.0
    if cls.get("needs_multi_hop") and "multi_hop" in profile.good_for:
        s += 2.0
    if cls.get("priority") in profile.priority:
        s += 1.0
    if cls.get("retrieval_need") in profile.retrieval:
        s += 1.0
    if cls.get("complexity") in profile.complexity:
        s += 1.0
    return s


def route(
    cls: dict,
    *,
    multi_engine: bool = False,
    tau_high: float = TAU_HIGH,
    tau_low: float = TAU_LOW,
    tie_margin: float = TIE_MARGIN,
) -> RouteDecision:
    """
    Decide a estratégia a partir da classificação `cls` (saída do query_analyzer).

    Parameters
    ----------
    multi_engine : bool
        Habilita o modo opcional de execução de 2 engines em caso de empate/baixa
        confiança (princípio 1 — desligado por padrão).
    """
    analyzer_conf = float(cls.get("confidence", 0.5))

    # ── Recusa por escopo (curto-circuito, custo zero) ────────────────────────
    if cls.get("in_scope") is False and analyzer_conf >= tau_high:
        return RouteDecision(
            engines=[],
            mode="refuse",
            confidence=analyzer_conf,
            query_type=cls.get("query_type", "?"),
            scores={},
            reasoning="Consulta fora do escopo dos Boletins de Conjuntura.",
            refused=True,
        )

    # ── Pontuação de todas as estratégias ─────────────────────────────────────
    profiles = get_profiles()
    scores = {k: _score(p, cls) for k, p in profiles.items()}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    top_key, top_score = ranked[0]
    second_key, second_score = ranked[1] if len(ranked) > 1 else (None, 0.0)

    # Se ninguém pontuou, cai no default seguro (principal cobre o caso geral).
    if top_score == 0.0:
        top_key = "principal" if "principal" in profiles else ranked[0][0]

    margin = top_score - second_score
    ambiguous = (margin < tie_margin) or (analyzer_conf < tau_low)

    # Confiança final: combina a do analyzer com a folga de score.
    margin_factor = min(margin / 3.0, 1.0)
    confidence = round(0.6 * analyzer_conf + 0.4 * margin_factor, 3)

    if ambiguous and multi_engine and second_key:
        engines = [top_key, second_key]
        mode = "multi"
        reasoning = (
            f"Ambígua (margem {margin:.1f}, conf {analyzer_conf:.2f}); "
            f"executando {top_key}+{second_key} e selecionando a melhor resposta."
        )
    else:
        engines = [top_key]
        mode = "single_best"
        reasoning = (
            f"query_type={cls.get('query_type')} → {top_key} "
            f"(score {top_score:.1f}, margem {margin:.1f})."
        )

    return RouteDecision(
        engines=engines,
        mode=mode,
        confidence=confidence,
        query_type=cls.get("query_type", "?"),
        scores=scores,
        reasoning=reasoning,
    )
