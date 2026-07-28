"""
orchestrator.py — Pipeline do Meta RAG.

Fluxo: analisar a consulta → rotear (single-best) → encaminhar para a engine
escolhida → anexar metadados de rota/qualidade. Não recupera documentos nem
gera texto por conta própria: apenas decide e delega (princípio 6).
"""
from __future__ import annotations

import asyncio
import time

from rag_core.answer_policy import REFUSAL_TEXT

from .fusion import run_engines, select_best
from .query_analyzer import QueryAnalyzer
from .quality_gate import summarize
from .registry import get_client, get_profiles, health_is_ready, profile_dict
from .router import RouteDecision, route

class Orchestrator:
    def __init__(self, analyzer: QueryAnalyzer | None = None,
                 multi_engine: bool = False, timeout: int = 180):
        self.analyzer = analyzer or QueryAnalyzer()
        self.multi_engine = multi_engine
        self.timeout = timeout

    async def answer(self, question: str) -> dict:
        t_start = time.perf_counter()

        # 1. Análise semântica (LLM síncrono → thread, não bloqueia o loop).
        t0 = time.perf_counter()
        cls = await asyncio.to_thread(self.analyzer.analyze, question)
        analyzer_ms = (time.perf_counter() - t0) * 1000

        # 2. Roteamento (single-best por padrão).
        t0 = time.perf_counter()
        decision = route(cls, multi_engine=self.multi_engine)
        router_ms = (time.perf_counter() - t0) * 1000

        # 3. Recusa por escopo — curto-circuito, nenhuma engine chamada.
        if decision.mode == "refuse":
            timings = self._timings(analyzer_ms, router_ms, 0.0, t_start)
            return self._envelope(
                {"answer": REFUSAL_TEXT, "sources": [], "sources_used": [],
                 "rewritten_query": question, "validation": {"verified": 0, "total": 0, "unverified": []}},
                chosen=None, decision=decision, cls=cls, timings=timings,
            )

        # 4. Execução da engine escolhida.
        t0 = time.perf_counter()
        failover_from = None
        if decision.mode == "multi":
            results = await run_engines(decision.engines, question, self.timeout)
            chosen, resp = select_best(results)
        else:
            chosen = decision.engines[0]
            try:
                client = get_client(chosen, self.timeout)
                if not health_is_ready(await client.health()):
                    raise RuntimeError(f"Engine '{chosen}' indisponível no health check.")
                resp = await client.query(question)
            except Exception as exc:
                resp = {"error": str(exc)}

        # Falha/health negativo: tenta as demais estratégias pela pontuação da rota.
        if "error" in resp:
            failover_from = chosen
            fallback_key, fallback_resp = await self._fallback(
                question, decision, excluded=set(decision.engines)
            )
            if fallback_key is not None:
                chosen, resp = fallback_key, fallback_resp
        engine_ms = (time.perf_counter() - t0) * 1000
        timings = self._timings(analyzer_ms, router_ms, engine_ms, t_start)

        # 5. Erro de backend → envelope de erro (API traduz para 502).
        if "error" in resp:
            return self._envelope(resp, chosen=chosen, decision=decision, cls=cls,
                                  timings=timings, error=resp["error"],
                                  failover_from=failover_from)

        return self._envelope(
            resp, chosen=chosen, decision=decision, cls=cls, timings=timings,
            failover_from=failover_from,
        )

    async def _fallback(
        self, question: str, decision: RouteDecision, excluded: set[str]
    ) -> tuple[str | None, dict]:
        """Escolhe um backend saudável restante e tenta consultas em ordem de score."""
        candidates = [
            key for key, _score in sorted(
                decision.scores.items(), key=lambda item: item[1], reverse=True
            )
            if key not in excluded
        ]
        clients = {}
        for key in candidates:
            try:
                clients[key] = get_client(key, self.timeout)
            except Exception:
                continue
        candidates = [key for key in candidates if key in clients]
        if not candidates:
            return None, {"error": "Nenhuma engine disponível para failover."}
        checks = await asyncio.gather(
            *(client.health() for client in clients.values()),
            return_exceptions=True,
        )
        healthy = [
            key for key, check in zip(candidates, checks)
            if not isinstance(check, BaseException) and health_is_ready(check)
        ]
        last_error = "Nenhuma engine saudável disponível para failover."
        for key in healthy:
            try:
                response = await clients[key].query(question)
                if "error" not in response:
                    return key, response
                last_error = str(response["error"])
            except Exception as exc:
                last_error = str(exc)
        return None, {"error": last_error}

    @staticmethod
    def _timings(analyzer_ms: float, router_ms: float, engine_ms: float,
                 t_start: float) -> dict:
        return {
            "analyzer_ms": round(analyzer_ms, 1),
            "router_ms": round(router_ms, 1),
            "engine_ms": round(engine_ms, 1),
            "total_ms": round((time.perf_counter() - t_start) * 1000, 1),
        }

    async def route_only(self, question: str) -> dict:
        """Decisão de roteamento sem executar engine (debug/inspeção)."""
        cls = await asyncio.to_thread(self.analyzer.analyze, question)
        decision = route(cls, multi_engine=self.multi_engine)
        return {"analysis": cls, "route": _route_dict(decision)}

    # ── Montagem da resposta final ────────────────────────────────────────────

    def _envelope(self, resp: dict, *, chosen, decision: RouteDecision,
                  cls: dict, timings: dict, error: str | None = None,
                  failover_from: str | None = None) -> dict:
        profiles = get_profiles()
        engine_profile = profile_dict(chosen) if chosen else None
        final = dict(resp)
        final["route"] = {
            **_route_dict(decision),
            "engine": chosen,
            "engine_label": profiles[chosen].label if chosen in profiles else None,
            "failover_from": failover_from,
            "degraded": failover_from is not None,
        }
        final["analysis"] = cls               # QueryProfile
        final["engine_profile"] = engine_profile
        final["timings"] = timings
        final["models"] = {
            "analyzer": self.analyzer.model,
            "generation": engine_profile["llm_model"] if engine_profile else None,
            "embeddings": engine_profile["embed_model"] if engine_profile else None,
        }
        if error:
            final["error"] = error
        else:
            final["quality"] = summarize(resp)
        return final


def _route_dict(decision: RouteDecision) -> dict:
    return {
        "engines_used": decision.engines,
        "mode": decision.mode,
        "query_type": decision.query_type,
        "confidence": decision.confidence,
        "scores": decision.scores,
        "reasoning": decision.reasoning,
    }
