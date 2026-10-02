from types import SimpleNamespace

from rag_core.argument_validator import validate_arguments
from rag_core.citation_validator import validate_citations
from rag_core.numerical_validator import validate_numbers
from rag_core.provenance import format_source_context, source_file


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


def test_numero_verificado_preserva_posicao_e_fonte_exatas():
    response = "O primeiro indicador foi 12,5%, mas o principal chegou a 18,7%."
    checks = validate_numbers(
        response,
        [
            _node("Série secundária com resultado de 12,5%.", page=2),
            _node("O indicador principal chegou a 18,7%.", page=8),
        ],
    )
    by_value = {check.value: check for check in checks}

    assert by_value["12,5%"].source_index == 0
    assert response[
        by_value["12,5%"].response_start:by_value["12,5%"].response_end
    ] == "12,5%"
    assert by_value["18,7%"].source_index == 1


def test_numero_curto_nao_valida_por_substring_em_ano():
    checks = validate_numbers(
        "Foram gerados 16 mil postos.",
        [_node("Estado de Sao Paulo - janeiro de 2016 a julho de 2018.")],
    )

    assert checks[0].value == "16"
    assert not checks[0].verified


def test_inteiro_de_um_digito_em_valor_e_obrigatoriamente_validado():
    checks = validate_numbers(
        "Santos registrou saldo de 9 mil postos.",
        [_node("Santos passou a integrar o grupo de maior destaque.")],
    )

    assert len(checks) == 1
    assert checks[0].value == "9"
    assert not checks[0].verified


def test_inteiro_de_um_digito_e_aceito_quando_consta_na_fonte():
    checks = validate_numbers(
        "Santos registrou saldo de 9 mil postos.",
        [_node("Santos registrou saldo de 9 mil postos.")],
    )

    assert len(checks) == 1
    assert checks[0].value == "9"
    assert checks[0].verified


def test_marcadores_estruturais_de_um_digito_nao_viram_dados():
    checks = validate_numbers(
        "1. Campinas liderou. Veja a fonte [2], p. 3.",
        [_node("Campinas liderou.")],
    )

    assert checks == []


def test_valores_repetidos_sao_validados_por_ocorrencia_e_contexto():
    response = (
        "Sorocaba registrou 11 mil postos. "
        "Sao Jose dos Campos registrou 11 mil postos."
    )
    checks = validate_numbers(
        response,
        [
            _node("Sorocaba registrou 11 mil postos.", page=4),
            _node("Sao Jose dos Campos registrou 11 mil postos.", page=7),
        ],
    )

    assert len(checks) == 2
    assert all(check.verified for check in checks)
    assert [check.source_index for check in checks] == [0, 1]
    assert checks[0].response_start != checks[1].response_start


def test_valor_repetido_nao_empresta_validacao_a_outro_contexto():
    checks = validate_numbers(
        "Sorocaba registrou 11 mil postos. Campinas registrou 11 mil postos.",
        [_node("Sorocaba registrou 11 mil postos.")],
    )

    assert len(checks) == 2
    assert checks[0].verified
    assert not checks[1].verified


def test_mesmo_numero_com_unidade_diferente_nao_valida():
    checks = validate_numbers(
        "O saldo foi de 11 mil postos.",
        [_node("A taxa foi de 11%.")],
    )

    assert len(checks) == 1
    assert not checks[0].verified


def test_trecho_do_numero_curto_usa_match_exato_nao_primeira_substring():
    checks = validate_numbers(
        "Desde 2016, foram gerados 16 mil postos.",
        [_node("Periodo de 2016 a 2018. Saldo observado: 16 mil postos.")],
    )
    check = {item.value: item for item in checks}["16"]

    assert check.verified
    assert "Saldo observado: 16 mil" in check.source_snippet
    assert "foram gerados 16 mil" in check.response_snippet


def test_citacao_tabular_preserva_linha_estruturada_completa():
    table_node = _node(
        "Setor: Indústria\nEmpregos: 125.400\nVariação: 3,2%\nFonte: tabela.pdf",
        page=4,
    )
    table_node.metadata["type"] = "table"
    checks = validate_numbers(
        "A indústria registrou 125.400 empregos.",
        [table_node],
    )

    assert checks[0].source_snippet == (
        "Setor: Indústria\nEmpregos: 125.400\nVariação: 3,2%"
    )


def test_argumento_textual_rejeita_vocabulario_sem_suporte():
    checks = validate_arguments(
        "A composicao setorial reforca recuperacao parcial apos retracao de fim de ano.",
        [_node("As admissoes superaram os desligamentos na industria e na construcao.")],
    )

    assert not checks[0].verified
    assert "recuperacao" in checks[0].missing_terms


def test_argumento_textual_aceita_termos_presentes_na_fonte():
    checks = validate_arguments(
        "As admissoes superaram os desligamentos na industria e na construcao.",
        [_node("As admissoes superaram os desligamentos na industria e na construcao.")],
    )

    assert checks[0].verified


# ── Casos obrigatórios Frente B ───────────────────────────────────────────────

def test_semantic_equivalence_1_60_versus_1_6():
    checks = validate_numbers(
        "A taxa de crescimento foi de 1,60%.",
        [_node("A taxa de crescimento da economia foi de 1,6%.")],
    )
    by_val = {c.value: c for c in checks}
    assert by_val["1,60%"].verified


def test_direction_language_recuou_1_10_versus_minus_1_1():
    checks = validate_numbers(
        "A indústria recuou 1,10% na comparação trimestral.",
        [_node("No mesmo período, a indústria registrou -1,1% na comparação trimestral.")],
    )
    by_val = {c.value: c for c in checks}
    assert by_val["1,10%"].verified


def test_table_header_unit_inheritance_cell_minus_2_8_under_em_percent():
    table_node = _node(
        "Tabela 1 - Variação do PIB por setor (em %)\n"
        "Setor | Taxa\n"
        "Indústria | -2,8\n"
        "Serviços | 1,8\n"
    )
    table_node.metadata["type"] = "table"
    checks = validate_numbers(
        "O setor industrial apresentou variação de -2,8%.",
        [table_node],
    )
    by_val = {c.value: c for c in checks}
    assert by_val["-2,8%"].verified
    assert by_val["-2,8%"].unit == "%"


def test_period_conflict_1_6_in_april_does_not_support_1_6_in_q1():
    checks = validate_numbers(
        "O indicador avançou 1,6% no primeiro trimestre de 2026.",
        [_node("O indicador registrou crescimento de 1,6% em abril de 2026.")],
    )
    by_val = {c.value: c for c in checks}
    assert not by_val["1,6%"].verified


def test_refusal_with_question_year_marked_not_applicable():
    question = "Qual foi a taxa de mortalidade infantil em 2024?"
    response = "Não foram encontrados dados de mortalidade infantil para o ano de 2024 no acervo consultado."
    checks = validate_numbers(
        response,
        [_node("Taxa de mortalidade infantil em 2023 foi de 10,8 por mil nascidos vivos.")],
        question=question,
    )
    by_val = {c.value: c for c in checks}
    assert "2024" in by_val
    assert not by_val["2024"].applicable
    applicable = [c for c in checks if getattr(c, "applicable", True)]
    assert len(applicable) == 0


def test_repeated_numbers_in_different_periods_not_misconsumed():
    response = "O PIB cresceu 1,5% no 1º trimestre e a inflação foi de 1,5% em abril."
    node_q1 = _node("O PIB apresentou crescimento de 1,5% no 1º trimestre.")
    node_apr = _node("A inflação oficial registrou 1,5% em abril.")
    checks = validate_numbers(response, [node_q1, node_apr])
    assert len(checks) == 2
    assert all(c.verified for c in checks)
    sources = [c.source_index for c in checks]
    assert sources == [0, 1]

