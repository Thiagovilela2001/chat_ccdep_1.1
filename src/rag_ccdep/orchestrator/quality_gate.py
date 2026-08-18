"""
quality_gate.py — Critérios de qualidade sobre a resposta de uma engine.

Reaproveita o que as engines já produzem: cada `/query` de variante roda o
`numerical_validator` e devolve `validation`. Aqui apenas se interpreta esse
resultado, detecta recusa e resume um sinal de qualidade — sem reprocessar nada
(princípio 5 — reuso máximo).
"""
from __future__ import annotations

REFUSAL_MARKERS = (
    "não consta nos documentos",
    "não fornecem evidência suficiente",
)


def is_refusal(answer: str) -> bool:
    lowered = (answer or "").lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def verify_ratio(response: dict) -> float:
    """Fração de números da resposta confirmados nas fontes (1.0 se não há números)."""
    v = response.get("validation") or {}
    total = v.get("total", 0) or 0
    verified = v.get("verified", 0) or 0
    return (verified / total) if total else 1.0


def quality_score(response: dict) -> float:
    """
    Pontuação usada para desempate/seleção (§8, requisito 8). Recusa recebe a
    menor pontuação; o restante privilegia verificação numérica e evidências
    rastreáveis fornecidas separadamente pela API.
    """
    if is_refusal(response.get("answer", "")):
        return -1.0
    n_sources = len(response.get("sources", []) or [])
    return verify_ratio(response) * 10.0 + min(n_sources, 5)


def summarize(response: dict) -> dict:
    """Resumo de qualidade anexado à resposta final (rastreabilidade)."""
    v = response.get("validation") or {}
    return {
        "refused": is_refusal(response.get("answer", "")),
        "verify_ratio": round(verify_ratio(response), 3),
        "numbers_verified": f"{v.get('verified', 0)}/{v.get('total', 0)}",
        "n_sources": len(response.get("sources", []) or []),
    }
