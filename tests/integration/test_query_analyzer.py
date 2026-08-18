from rag_ccdep.orchestrator.query_analyzer import _heuristic_fallback, _merge_defaults


def test_comparativo_pertence_ao_contrato():
    result = _merge_defaults({"query_type": "comparativo", "confidence": 0.8})
    assert result["query_type"] == "comparativo"


def test_saida_invalida_e_normalizada():
    result = _merge_defaults({
        "query_type": ["ampla"],
        "confidence": 4,
        "in_scope": "false",
        "entities": "PIB",
    })
    assert result["query_type"] == "pontual"
    assert result["confidence"] == 1.0
    assert result["in_scope"] is True
    assert result["entities"] == []


def test_fallback_reconhece_comparacao():
    result = _heuristic_fallback("Compare o PIB de 2023 versus 2024")
    assert result["query_type"] == "comparativo"
    assert result["intent"] == "comparar"
