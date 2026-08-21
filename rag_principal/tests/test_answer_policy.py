from types import SimpleNamespace

from rag_core.answer_policy import (
    CALCULATION_FAILURE_PROVENANCE_TEXT,
    CALCULATION_CLARIFICATION_PERIOD_TEXT,
    CALCULATION_CLARIFICATION_TERRITORY_TEXT,
    CALCULATION_PROVENANCE_TEXT,
    CALCULATION_SOURCES_HEADER,
    REFUSAL_TEXT,
    asks_for_calculation,
    asks_for_sources,
    calculation_clarification,
    calculation_missing_data_refusal,
    enforce_calculation_provenance,
    sanitize_answer,
)


def test_remove_citacao_inline_sem_perder_conteudo():
    answer = (
        "O setor de serviços liderou o crescimento "
        "(Fonte: Boletim_Conjuntura_2024.pdf, p. 18)."
    )
    assert sanitize_answer(answer, question="Qual setor liderou?") == (
        "O setor de serviços liderou o crescimento."
    )


def test_remove_todos_formatos_inline_proibidos():
    answer = (
        "A indústria avançou (pdf, página 12). "
        "Os serviços cresceram (boletim.pdf, p. 18). "
        "O comércio ficou estável [1]."
    )
    assert sanitize_answer(answer, question="Como evoluíram os setores?") == (
        "A indústria avançou. Os serviços cresceram. O comércio ficou estável."
    )


def test_remove_sentenca_de_fonte_sem_apagar_analise():
    assert sanitize_answer(
        "O PIB avançou. Fonte: boletim.pdf, página 2.",
        question="Como evoluiu o PIB?",
    ) == "O PIB avançou."
    assert sanitize_answer(
        "Fonte: boletim.pdf, página 2. O PIB avançou.",
        question="Como evoluiu o PIB?",
    ) == "O PIB avançou."


def test_remove_bloco_de_fontes_e_metadados():
    answer = """A atividade econômica avançou.

### Fontes
- Boletim_2024.pdf, página 18, score 0,91
- tabela.xlsx, aba 2
"""
    assert sanitize_answer(answer, question="Como evoluiu a atividade?") == (
        "A atividade econômica avançou."
    )


def test_remove_atribuicao_e_nome_de_arquivo():
    answer = (
        "Segundo o documento Boletim_2024.pdf, o emprego cresceu. "
        "A confirmação aparece na página 12."
    )
    cleaned = sanitize_answer(answer, question="Como evoluiu o emprego?")
    assert cleaned == "O emprego cresceu."


def test_remove_detalhes_internos_sem_apagar_recuperacao_economica():
    answer = (
        "A recuperação econômica ganhou força. "
        "Foram usados 8 chunks com score de similaridade elevado. "
        "O nível de atividade avançou."
    )
    assert sanitize_answer(answer, question="Como evoluiu a economia?") == (
        "A recuperação econômica ganhou força. O nível de atividade avançou."
    )


def test_pedido_explicito_preserva_referencia_mas_nao_detalhes_internos():
    question = "Quais foram as fontes?"
    answer = (
        "Fonte: Boletim_2024.pdf, página 18. "
        "O contexto recuperado teve 8 chunks."
    )
    assert asks_for_sources(question)
    assert sanitize_answer(answer, question=question) == (
        "Fonte: Boletim_2024.pdf, página 18."
    )


def test_resposta_vazia_usa_mensagem_padrao():
    assert sanitize_answer("", question="Qual o dado?") == REFUSAL_TEXT


def test_recusa_isolada_permanece_padronizada():
    assert sanitize_answer(
        f"  {REFUSAL_TEXT}\n", question="Qual o dado?"
    ) == REFUSAL_TEXT


def test_remove_recusa_global_anexada_a_resposta_sustentada():
    answer = (
        "A população residente era de 44,4 milhões.\n\n"
        f"{REFUSAL_TEXT}"
    )
    assert sanitize_answer(answer, question="Qual era a população?") == (
        "A população residente era de 44,4 milhões."
    )


def test_remove_recusa_global_antes_de_resposta_sustentada():
    answer = f"{REFUSAL_TEXT}\n\nA taxa anual foi de 0,62%."
    assert sanitize_answer(answer, question="Qual foi a taxa?") == (
        "A taxa anual foi de 0,62%."
    )


def test_sanitizacao_de_recusa_mista_e_idempotente():
    answer = f"O indicador recuou. {REFUSAL_TEXT}"
    cleaned = sanitize_answer(answer, question="Como evoluiu?")
    assert sanitize_answer(cleaned, question="Como evoluiu?") == cleaned


def test_pergunta_generica_sobre_documentos_nao_libera_metadados():
    assert not asks_for_sources("Analise os documentos sobre atividade econômica.")


def test_variantes_de_pedido_explicito_de_fontes():
    assert asks_for_sources("Qual documento afirma isso?")
    assert asks_for_sources("De onde veio essa informação?")
    assert asks_for_sources("Mostre as referências.")


def test_detecta_pedido_de_calculo_sem_herdar_historico_antigo():
    assert asks_for_calculation("Calcule a variação percentual.")
    assert asks_for_calculation("Qual o índice de envelhecimento?")
    assert not asks_for_calculation(
        "<HISTORICO_DA_CONVERSA>USUÁRIO: calcule o índice</HISTORICO_DA_CONVERSA>"
        "<PERGUNTA_ATUAL>Qual foi o PIB?</PERGUNTA_ATUAL>"
    )


def test_proveniencia_obrigatoria_em_calculo_validado_e_idempotente():
    checks = [SimpleNamespace(verified=True), SimpleNamespace(verified=True)]
    answer = "A conta foi 10 + 20 = 30."

    result = enforce_calculation_provenance(
        answer,
        question="Calcule a soma.",
        checks=checks,
        sources=["tabela.pdf, p./aba 4"],
    )

    assert result.endswith(CALCULATION_PROVENANCE_TEXT)
    assert CALCULATION_SOURCES_HEADER in result
    assert "- tabela.pdf, p./aba 4" in result
    assert enforce_calculation_provenance(
        result,
        question="Calcule a soma.",
        checks=checks,
        sources=["tabela.pdf, p./aba 4"],
    ) == result


def test_calculo_nao_validado_e_bloqueado_com_proveniencia():
    result = enforce_calculation_provenance(
        "A conta foi 10 + 20 = 99.",
        question="Calcule a soma.",
        checks=[SimpleNamespace(verified=False)],
    )

    assert result.startswith(REFUSAL_TEXT)
    assert result.endswith(CALCULATION_FAILURE_PROVENANCE_TEXT)
    assert "99" not in result


def test_recusa_de_calculo_tambem_explicita_proveniencia():
    result = enforce_calculation_provenance(
        REFUSAL_TEXT,
        question="Calcule a razão.",
        checks=[],
    )

    assert result == f"{REFUSAL_TEXT}\n\n{CALCULATION_FAILURE_PROVENANCE_TEXT}"


def test_calculo_sem_operacao_explicita_e_bloqueado():
    result = enforce_calculation_provenance(
        "O resultado calculado foi 30.",
        question="Calcule a soma de 10 e 20.",
        checks=[SimpleNamespace(verified=True)],
        sources=["tabela.pdf"],
    )

    assert result.startswith(REFUSAL_TEXT)
    assert "30" not in result


def test_calculo_sem_documento_identificavel_e_bloqueado():
    result = enforce_calculation_provenance(
        "A conta foi 10 + 20 = 30.",
        question="Calcule a soma.",
        checks=[SimpleNamespace(verified=True)],
        sources=[],
    )

    assert result.startswith(REFUSAL_TEXT)


def test_operacao_com_percentuais_e_reconhecida():
    result = enforce_calculation_provenance(
        "A diferença foi 3,4% − 2,8% = 0,6 p.p.",
        question="Calcule a diferença.",
        checks=[SimpleNamespace(verified=True)],
        sources=["boletim.pdf, p./aba 2"],
    )

    assert CALCULATION_SOURCES_HEADER in result


def test_sanitizacao_posterior_preserva_auditoria_do_calculo():
    answer = enforce_calculation_provenance(
        "A conta foi 10 + 20 = 30.",
        question="Calcule a soma.",
        checks=[SimpleNamespace(verified=True)],
        sources=["tabela.pdf, p./aba 4"],
    )

    sanitized = sanitize_answer(answer, question="Calcule a soma.")

    assert "10 + 20 = 30" in sanitized
    assert CALCULATION_SOURCES_HEADER in sanitized
    assert "tabela.pdf, p./aba 4" in sanitized
    assert CALCULATION_PROVENANCE_TEXT in sanitized


def test_resposta_ambigua_pede_periodos_em_vez_de_calcular():
    answer = (
        "O índice passou de 43,3 para 49,0, diferença de 5,7 pontos. "
        "Os trechos não especificam os anos correspondentes. "
        "Sem definição clara dos períodos, a comparação não é rastreável."
    )

    result = enforce_calculation_provenance(
        answer,
        question="Compare os índices e calcule a diferença absoluta.",
        checks=[SimpleNamespace(verified=True)],
        sources=["demografia.pdf, p./aba 8"],
    )

    assert result == CALCULATION_CLARIFICATION_PERIOD_TEXT
    assert "43,3" not in result
    assert "5,7" not in result


def test_ambiguidade_territorial_pede_recorte_geografico():
    assert calculation_clarification(
        "O território não foi especificado e há vários municípios."
    ) == CALCULATION_CLARIFICATION_TERRITORY_TEXT


def test_falta_real_de_dados_nao_e_tratada_como_ambiguidade():
    assert calculation_clarification(
        "Os documentos não contêm os valores necessários para o cálculo."
    ) is None


def test_falta_de_periodo_usa_recusa_curta_sem_valores_ou_anos_laterais():
    answer = (
        "O índice foi 23,2 em 2000, 36,5 em 2010 e 66,3 em 2022. "
        "Para a comparação pedida, o dado de 2022 está disponível, mas o de "
        "2023 está ausente."
    )

    result = enforce_calculation_provenance(
        answer,
        question=(
            "Compare os índices de envelhecimento de 2022 e 2023 e calcule "
            "a diferença absoluta."
        ),
        checks=[],
    )

    assert result == (
        "Não é possível calcular a diferença absoluta.\n\n"
        "- Dado encontrado: Índice de envelhecimento para 2022.\n"
        "- Dado ausente: Índice de envelhecimento para 2023.\n"
        "Operação cancelada por ausência de dados na fonte."
    )
    assert "23,2" not in result
    assert "2000" not in result
    assert "2010" not in result
    assert CALCULATION_FAILURE_PROVENANCE_TEXT not in result


def test_recusa_generica_identifica_que_nenhum_periodo_foi_encontrado():
    result = calculation_missing_data_refusal(
        REFUSAL_TEXT,
        question="Calcule a variação percentual entre 2019 e 2020.",
    )

    assert result == (
        "Não é possível calcular a variação percentual.\n\n"
        "- Dado encontrado: Nenhum dos períodos solicitados.\n"
        "- Dado ausente: Dado solicitado para 2019, 2020.\n"
        "Operação cancelada por ausência de dados na fonte."
    )


def test_recusa_sem_periodos_explicitos_preserva_fluxo_anterior():
    assert calculation_missing_data_refusal(
        "Os documentos não contêm os valores necessários.",
        question="Calcule a razão.",
    ) is None
