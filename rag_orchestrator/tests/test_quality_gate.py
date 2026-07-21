from rag_orchestrator.src.quality_gate import quality_score, summarize


def _response(citations):
    return {
        "answer": "Resposta factual.",
        "sources": [{"file": "a.pdf"}],
        "validation": {"verified": 1, "total": 1},
        "citation_validation": citations,
    }


def test_citacoes_verificadas_melhoram_score():
    cited = _response({"verified": 1, "total": 1})
    uncited = _response({"verified": 0, "total": 0})
    assert quality_score(cited) > quality_score(uncited)


def test_resumo_sinaliza_ausencia_de_citacoes():
    quality = summarize(_response({"verified": 0, "total": 0}))
    assert quality["citation_precision"] == 0.0
    assert quality["citation_coverage"] is False
