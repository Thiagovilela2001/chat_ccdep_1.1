from rag_orchestrator.src.quality_gate import is_refusal, quality_score, summarize


def _response(*, sources=None):
    return {
        "answer": "Resposta factual.",
        "sources": sources if sources is not None else [{"file": "a.pdf"}],
        "validation": {"verified": 1, "total": 1},
    }


def test_evidencias_separadas_melhoram_score():
    with_sources = _response()
    without_sources = _response(sources=[])
    assert quality_score(with_sources) > quality_score(without_sources)


def test_resumo_nao_exige_citacoes_inline():
    quality = summarize(_response())
    assert quality["n_sources"] == 1
    assert "citation_precision" not in quality
    assert "citation_coverage" not in quality


def test_nova_mensagem_de_evidencia_insuficiente_e_recusa():
    assert is_refusal(
        "Os documentos disponíveis não fornecem evidência suficiente para responder."
    )
