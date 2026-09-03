"""
Smoke do pipeline do orquestrador com analyzer e cliente HTTP falsos
(sem rede, sem LLM, sem transformers).

Executar:  python rag_orchestrator/tests/test_pipeline.py
"""
import asyncio
import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import fusion as fusion_mod        # noqa: E402
from src import orchestrator as orch_mod    # noqa: E402
from src.orchestrator import Orchestrator    # noqa: E402


class FakeAnalyzer:
    def __init__(self, cls):
        self.cls = cls
        self.model = "fake-analyzer-model"

    def analyze(self, q):
        return self.cls


class FakeClient:
    def __init__(self, resp):
        self.resp = resp

    async def query(self, q):
        return self.resp

    async def health(self):
        return {"engine_ready": True}


class HealthClient(FakeClient):
    def __init__(self, resp, healthy):
        super().__init__(resp)
        self.healthy = healthy

    async def health(self):
        return {"engine_ready": True} if self.healthy else None


def test_engine_client_headers():
    """A credencial interna deve ser encaminhada às engines."""
    from src.registry import EngineClient

    old = os.environ.get("RAG_BACKEND_API_KEY")
    os.environ["RAG_BACKEND_API_KEY"] = "internal-test-key"
    try:
        check("header interno encaminhado",
              EngineClient._headers() == {"x-api-key": "internal-test-key"})
    finally:
        if old is None:
            os.environ.pop("RAG_BACKEND_API_KEY", None)
        else:
            os.environ["RAG_BACKEND_API_KEY"] = old


def check(nome, cond):
    print(f"  [{'OK ' if cond else 'FALHOU'}] {nome}")
    assert cond, nome


def main():
    print("Smoke do pipeline:")

    test_engine_client_headers()

    # 1. Single-best encaminha para principal e monta o envelope.
    cls = {"query_type": "pontual", "confidence": 0.9, "in_scope": True,
           "priority": "precisao", "retrieval_need": "hibrida"}
    canned = {"answer": "A taxa foi 7,9%.",
              "sources": [{"file": "boletim.pdf", "score": 0.9}],
              "validation": {"verified": 1, "total": 1, "unverified": []}}
    orch_mod.get_client = lambda key, timeout=180: FakeClient(canned)
    res = asyncio.run(Orchestrator(analyzer=FakeAnalyzer(cls)).answer("qual a taxa?"))
    check("single-best → engine=principal", res["route"]["engine"] == "principal")
    check("resposta encaminhada", res["answer"] == "A taxa foi 7,9%.")
    check("quality anexada", res["quality"]["refused"] is False)
    check("route no envelope", res["route"]["mode"] == "single_best")

    # 2. Recusa por escopo (nenhuma engine chamada).
    cls_out = {"query_type": "pontual", "confidence": 0.95, "in_scope": False}
    res = asyncio.run(Orchestrator(analyzer=FakeAnalyzer(cls_out)).answer("qual a Selic?"))
    check("fora de escopo → refuse", res["route"]["mode"] == "refuse")
    check("texto de limite", "não é possível determinar" in res["answer"].lower())

    # 3. Multi-engine (opcional): escolhe a melhor resposta por qualidade.
    cls_amb = {"query_type": "ampla", "confidence": 0.3, "in_scope": True,
               "priority": "precisao", "retrieval_need": "lexical"}
    boa = {"answer": "Panorama detalhado.", "sources": [{"file": "a.pdf", "score": 0.9}],
           "validation": {"verified": 2, "total": 2, "unverified": []}}
    recusa = {"answer": "A informação não consta nos documentos fornecidos.",
              "sources": [], "validation": {"verified": 0, "total": 0, "unverified": []}}
    respostas = {"raptor": boa, "principal": recusa}
    fusion_mod.get_client = lambda key, timeout=180: FakeClient(respostas[key])
    res = asyncio.run(
        Orchestrator(analyzer=FakeAnalyzer(cls_amb), multi_engine=True).answer("como está a economia?")
    )
    check("multi → 2 engines na rota", len(res["route"]["engines_used"]) == 2)
    check("multi → escolhe a não-recusa", res["answer"] == "Panorama detalhado.")

    # 4. Backend primário fora do ar → failover para engine saudável.
    fallback = {"answer": "Resposta via failover.", "sources": [],
                "validation": {"verified": 0, "total": 0, "unverified": []}}
    clients = {
        "principal": HealthClient({}, False),
        "agentic": HealthClient({}, False),
        "selfrag": HealthClient({}, False),
        "raptor": HealthClient(fallback, True),
    }
    orch_mod.get_client = lambda key, timeout=180: clients[key]
    res = asyncio.run(Orchestrator(analyzer=FakeAnalyzer(cls)).answer("qual a taxa?"))
    check("failover escolhe engine saudável", res["route"]["engine"] == "raptor")
    check("failover sinalizado", res["route"]["failover_from"] == "principal")

    # 5. Todos os backends fora do ar → envelope de erro.
    def _boom(key, timeout=180):
        raise RuntimeError("connection refused")
    orch_mod.get_client = _boom
    res = asyncio.run(Orchestrator(analyzer=FakeAnalyzer(cls)).answer("qual a taxa?"))
    check("erro de backend capturado", "error" in res)

    print("\nPipeline OK.")


if __name__ == "__main__":
    main()
