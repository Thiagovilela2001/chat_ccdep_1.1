from rag_core.answer_policy import (
    REFUSAL_TEXT,
    asks_for_sources,
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


def test_colapsa_folga_no_separador_de_milhar():
    assert sanitize_answer(
        "O rendimento médio foi de R$ 4. 038.", question="Qual o rendimento?"
    ) == "O rendimento médio foi de R$ 4.038."
    assert sanitize_answer(
        "O rendimento médio foi de R$ 4 . 038.", question="Qual o rendimento?"
    ) == "O rendimento médio foi de R$ 4.038."
    assert sanitize_answer(
        "A população era de 44. 400. 000 habitantes.", question="Qual a população?"
    ) == "A população era de 44.400.000 habitantes."
    assert sanitize_answer(
        "O grupo de 90 a 99 anos saltou de 83. 421 para 170. 679 pessoas.",
        question="Como evoluiu o grupo?",
    ) == "O grupo de 90 a 99 anos saltou de 83.421 para 170.679 pessoas."


def test_preserva_fim_de_frase_e_decimal_com_virgula():
    answer = "O PIB cresceu 1,5% em 2024. Em 2025, subiu 0,6%."
    assert sanitize_answer(answer, question="Como evoluiu o PIB?") == answer
    answer = "O déficit foi de US$ 12,41 bilhões."
    assert sanitize_answer(answer, question="Qual o saldo?") == answer


def test_colapsa_ponto_percentual_espacado():
    assert sanitize_answer(
        "A taxa recuou 2,6 p. p. no trimestre.", question="Como evoluiu a taxa?"
    ) == "A taxa recuou 2,6 p.p. no trimestre."


def test_remove_conectivo_orfao_apos_frase_interna_removida():
    answer = (
        "Os trechos recuperados não informam a taxa mais recente. "
        "Ainda assim, os documentos permitem traçar o quadro anterior."
    )
    assert sanitize_answer(answer, question="Qual a taxa?") == (
        "Os documentos permitem traçar o quadro anterior."
    )
    answer = (
        "Os documentos recuperados mostram deterioração. "
        "Em contrapartida, o saldo comercial caiu."
    )
    assert sanitize_answer(answer, question="Qual o saldo?") == (
        "O saldo comercial caiu."
    )


def test_conectivo_preservado_no_meio_da_resposta():
    answer = "O PIB caiu. Ainda assim, o emprego subiu."
    assert sanitize_answer(answer, question="Como evoluiu a economia?") == answer


def test_conectivo_legitimo_no_inicio_nao_e_removido():
    answer = "Assim, o nível de atividade avançou."
    assert sanitize_answer(answer, question="Como evoluiu a atividade?") == answer
