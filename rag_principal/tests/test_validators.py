from types import SimpleNamespace

from rag_core.citation_validator import validate_citations
from rag_core.numerical_validator import validate_numbers
from rag_core.provenance import format_source_context


def _node(text: str, *, file="regional/boletim.pdf", page=3, score=0.9):
    return SimpleNamespace(
        metadata={"source_file": file, "page": page},
        score=score,
        get_content=lambda: text,
    )


def test_citacao_confere_arquivo_e_pagina():
    nodes = [_node("A taxa foi 7,9%.")]
    checks = validate_citations(
        "A taxa foi 7,9% (Fonte: boletim.pdf, p. 3). ", nodes
    )
    assert len(checks) == 1
    assert checks[0].verified


def test_citacao_com_pagina_incorreta_e_rejeitada():
    checks = validate_citations(
        "A taxa foi 7,9% (Fonte: boletim.pdf, p. 9).", [_node("7,9%")]
    )
    assert not checks[0].verified


def test_citacao_normaliza_markdown_e_intervalo_de_paginas():
    nodes = [
        _node("Primeiro trecho.", file="regional/boletim.pdf", page=3),
        _node("Segundo trecho.", file="regional/boletim.pdf", page=4),
    ]
    checks = validate_citations(
        "Síntese (**Fonte:** `boletim.pdf`, p. 3–4).",
        nodes,
    )
    assert len(checks) == 1
    assert checks[0].verified


def test_citacao_rejeita_intervalo_com_pagina_nao_recuperada():
    checks = validate_citations(
        "Síntese (Fonte: boletim.pdf, p. 3–5).",
        [_node("Trecho.", page=3), _node("Trecho.", page=5)],
    )
    assert not checks[0].verified


def test_contexto_enviado_ao_llm_inclui_proveniencia():
    context = format_source_context(_node("Trecho documental."))
    assert "Fonte: regional/boletim.pdf" in context
    assert "p./aba 3" in context
    assert context.endswith("Trecho documental.")


def test_resultado_aritmetico_explicito_e_verificado():
    checks = validate_numbers(
        "A diferença foi 3,4% − 2,8% = 0,6 p.p.",
        [_node("Taxas observadas: 3,4% e 2,8%.")],
    )
    by_value = {check.value: check for check in checks}
    assert by_value["0,6"].verified
    assert by_value["0,6"].derived


def test_resultado_aritmetico_incorreto_e_rejeitado():
    checks = validate_numbers(
        "A diferença foi 3,4% − 2,8% = 0,9 p.p.",
        [_node("Taxas observadas: 3,4% e 2,8%.")],
    )
    by_value = {check.value: check for check in checks}
    assert not by_value["0,9"].verified
