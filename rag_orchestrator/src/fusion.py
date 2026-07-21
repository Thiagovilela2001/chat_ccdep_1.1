"""
fusion.py — Seleção da melhor resposta no modo multi-engine (opcional).

Só é usado quando o roteador ativa o modo "multi" (empate/baixa confiança com a
flag de multi-engine ligada — princípio 1). Executa as engines candidatas em
paralelo e seleciona a melhor resposta por critérios determinísticos, **sem**
chamada LLM adicional (requisito 8).
"""
from __future__ import annotations

import asyncio

from .quality_gate import quality_score
from .registry import get_client, health_is_ready


async def run_engines(engines: list[str], question: str, timeout: int = 180) -> dict[str, dict]:
    """Executa as engines em paralelo; falhas viram entradas de erro (não abortam)."""
    async def _one(key: str):
        try:
            client = get_client(key, timeout=timeout)
            if not health_is_ready(await client.health()):
                return key, {"error": f"Engine '{key}' indisponível no health check."}
            return key, await client.query(question)
        except Exception as exc:  # rede/backend fora → registra e segue
            return key, {"error": str(exc)}

    pairs = await asyncio.gather(*(_one(k) for k in engines))
    return dict(pairs)


def select_best(results: dict[str, dict]) -> tuple[str, dict]:
    """
    Escolhe (engine, resposta) de maior `quality_score`. Ignora entradas com erro
    a menos que todas tenham falhado (nesse caso devolve a primeira com erro).
    """
    ok = {k: v for k, v in results.items() if "error" not in v}
    if not ok:
        key = next(iter(results))
        return key, results[key]
    best = max(ok.items(), key=lambda kv: quality_score(kv[1]))
    return best[0], best[1]
