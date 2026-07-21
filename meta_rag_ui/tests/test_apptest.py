"""
Smoke de renderização do frontend via streamlit.testing (AppTest).

Executa o app.py num ambiente simulado (sem navegador) e verifica que carrega
sem exceção — com backend offline (estado degradado esperado).

Executar:  python meta_rag_ui/tests/test_apptest.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def main():
    print("AppTest: carregando app.py (backend offline)…")
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception, f"App levantou exceção: {at.exception}"
    print(f"  [OK ] app carregou sem exceção (títulos: {len(at.title)})")

    # Habilita o modo desenvolvedor e re-executa.
    at.session_state["dev_mode"] = True
    at.run()
    assert not at.exception, f"Modo dev levantou exceção: {at.exception}"
    print("  [OK ] modo desenvolvedor carregou sem exceção")

    # Simula um histórico com resposta+telemetria e re-executa (render de métricas/dev).
    at.session_state["messages"] = [
        {"role": "user", "content": "qual a taxa de desocupação?", "meta": None},
        {"role": "assistant", "content": "A taxa foi 7,9%.", "meta": {
            "answer": "A taxa foi 7,9%.",
            "sources": [{"file": "boletim.pdf", "score": 0.9}],
            "route": {"engine": "principal", "engine_label": "RAG Principal",
                      "query_type": "pontual", "confidence": 0.88, "mode": "single_best",
                      "scores": {"principal": 5.0, "raptor": 2.0}, "reasoning": "..."},
            "analysis": {"query_type": "pontual", "confidence": 0.88},
            "engine_profile": {"label": "RAG Principal", "description": "híbrido",
                               "retrieval": ["hibrida"], "good_for": ["pontual"],
                               "limitations": ["síntese ampla"], "llm_model": "gpt-5-chat-latest",
                               "embed_model": "BAAI/bge-m3"},
            "timings": {"analyzer_ms": 300, "router_ms": 1, "engine_ms": 900, "total_ms": 1201},
            "models": {"analyzer": "gpt-5-mini", "generation": "gpt-5-chat-latest",
                       "embeddings": "BAAI/bge-m3"},
            "quality": {"numbers_verified": "1/1"},
        }},
    ]
    at.run()
    assert not at.exception, f"Render com histórico levantou exceção: {at.exception}"
    print("  [OK ] render de métricas + painel dev com dados simulados")

    # Resposta de engine DIRETA (sem route): fontes + validação numérica.
    at.session_state["messages"] = [
        {"role": "user", "content": "qual o PIB de SP?", "meta": None},
        {"role": "assistant", "content": "O PIB cresceu 0,4%.", "meta": {
            "answer": "O PIB cresceu 0,4%.",
            "rag_type": "principal",
            "rag_label": "RAG Principal",
            "sources_used": ["text", "tables"],
            "rewritten_query": "variação PIB São Paulo 1º trimestre 2024",
            "sources": [{"file": "boletim_1trim2024.pdf", "score": 0.87}],
            "validation": {"verified": 1, "total": 2, "unverified": ["3,1%"]},
            "citation_validation": {"verified": 1, "total": 1, "unverified": []},
            "_client_roundtrip_ms": 950.0,
        }},
    ]
    at.run()
    assert not at.exception, f"Render de engine direta levantou exceção: {at.exception}"
    print("  [OK ] render de resposta de engine direta (fontes + validação)")

    print("\nFrontend OK.")


if __name__ == "__main__":
    main()
