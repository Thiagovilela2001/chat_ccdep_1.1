"""
Teste do roteador (pura lógica de decisão — sem rede, sem LLM).

Executar:  python rag_orchestrator/tests/test_router.py
"""
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from rag_ccdep.orchestrator.router import route


def _cls(**kw):
    base = {"confidence": 0.9, "in_scope": True}
    base.update(kw)
    return base


def check(nome, cond):
    status = "OK " if cond else "FALHOU"
    print(f"  [{status}] {nome}")
    assert cond, nome


def main():
    print("Testes do roteador:")

    d = route(_cls(query_type="pontual", priority="precisao", retrieval_need="hibrida"))
    check("pontual → principal", d.engines == ["principal"] and d.mode == "single_best")

    d = route(_cls(query_type="tabular", priority="precisao", retrieval_need="lexical"))
    check("tabular → principal", d.engines == ["principal"])

    d = route(_cls(query_type="ampla", priority="abrangencia", retrieval_need="semantica"))
    check("ampla → raptor", d.engines == ["raptor"])

    d = route(_cls(query_type="comparativo", priority="abrangencia", retrieval_need="semantica"))
    check("comparativo → raptor", d.engines == ["raptor"])

    d = route(_cls(query_type="multi_hop", needs_multi_hop=True, complexity="alta"))
    check("multi_hop → agentic", d.engines == ["agentic"])

    d = route(_cls(query_type="verificacao", priority="precisao", retrieval_need="semantica"))
    check("verificacao → selfrag", d.engines == ["selfrag"])

    d = route(_cls(query_type="relacional", priority="precisao", retrieval_need="hibrida"))
    check("relacional → principal", d.engines == ["principal"])

    # Fora de escopo com alta confiança → recusa (nenhuma engine).
    d = route(_cls(query_type="pontual", in_scope=False, confidence=0.95))
    check("fora de escopo → refuse", d.mode == "refuse" and d.engines == [])

    # Single-best é o padrão mesmo em ambiguidade (multi desligado).
    d = route(_cls(query_type="pontual", confidence=0.3), multi_engine=False)
    check("ambígua sem multi → 1 engine", len(d.engines) == 1 and d.mode == "single_best")

    # Ambiguidade + multi habilitado → 2 engines.
    d = route(_cls(query_type="ampla", priority="precisao", retrieval_need="lexical",
                   confidence=0.3), multi_engine=True)
    check("ambígua com multi → 2 engines", len(d.engines) == 2 and d.mode == "multi")

    print("\nTodos os testes passaram.")


if __name__ == "__main__":
    main()
