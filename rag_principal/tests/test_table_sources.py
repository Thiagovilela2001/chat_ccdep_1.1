from types import SimpleNamespace

from rag_core.numerical_validator import validate_numbers
from rag_core.table_sources import annotate_table_sources, document_label


def _node(text, file, page):
    return SimpleNamespace(
        metadata={"source_file": file, "page": page},
        get_content=lambda: text,
        text=text,
        score=0.9,
    )


NODES = [
    _node("Em 2025 a infraestrutura somou R$ 63,4 bilhões, 57% do total.",
          "investimentos/seade-piesp-anual-2025.pdf", 2),
    _node("A indústria recebeu R$ 23,1 bilhões, ou 21% do total anunciado.",
          "investimentos/seade-piesp-setores.pdf", 4),
]

ANSWER = (
    "## Infraestrutura concentra os investimentos de 2025\n\n"
    "A infraestrutura liderou os anúncios.\n\n"
    "| Setor | Valor | Participação |\n"
    "|---|---:|---:|\n"
    "| Infraestrutura | R$ 63,4 bilhões | 57% |\n"
    "| Indústria | R$ 23,1 bilhões | 21% |\n\n"
    "A indústria ficou em segundo lugar, com 21% do total."
)


def test_nota_de_fonte_lista_os_documentos_dos_numeros_da_tabela():
    checks = validate_numbers(ANSWER, NODES)
    answer, shifted = annotate_table_sources(ANSWER, checks, NODES)

    note = ("*Fonte: Fundação Seade, seade piesp anual 2025, p. 2; "
            "Fundação Seade, seade piesp setores, p. 4.*")
    assert note in answer
    assert answer.index(note) > answer.index("| Indústria |")
    # Cada citação continua apontando o mesmo número depois do deslocamento.
    for check in shifted:
        if check.response_start is not None:
            assert answer[check.response_start:check.response_end].strip() == check.value.strip()


def test_sem_tabela_ou_sem_numero_verificado_nao_altera():
    plain = "Texto sem tabela, com 57% do total."
    checks = validate_numbers(plain, NODES)
    assert annotate_table_sources(plain, checks, NODES) == (plain, checks)

    unsourced = "| A | B |\n|---|---|\n| x | 999 |"
    checks = validate_numbers(unsourced, NODES)
    assert annotate_table_sources(unsourced, checks, NODES)[0] == unsourced


def test_celulas_sem_unidade_ainda_encontram_a_fonte():
    # Caso real: a coluna diz "R$ bilhões" e a célula traz só "63,4".
    table = (
        "| Setor | Valor (R$ bilhões) | Participação (%) |\n"
        "|---|---:|---:|\n"
        "| Infraestrutura | 63,4 | 57 |\n"
        "| Indústria | 23,1 | 21 |\n"
        "| Comércio | 13,9 | 13 |"
    )
    other = _node("Comércio exterior somou 999 em 2024.", "outro.pdf", 1)
    nodes = [other, NODES[0], NODES[1]]
    answer, _ = annotate_table_sources(table, [], nodes)
    assert answer.endswith("*Fonte: Fundação Seade, seade piesp anual 2025, p. 2; "
                           "Fundação Seade, seade piesp setores, p. 4.*")
    assert "outro" not in answer


def test_query_service_anexa_fonte_so_no_modo_artigo(monkeypatch):
    import asyncio

    from rag_core.query_service import execute_engine_query

    monkeypatch.setenv("RAG_NUMERIC_VALIDATION", "1")
    monkeypatch.setenv("RAG_ANSWER_RETRIES", "0")
    monkeypatch.setenv("RAG_POPUP_EXPLANATIONS", "0")

    class Engine:
        supports_article_mode = True

        async def answer(self, **_kwargs):
            return ANSWER, NODES

    def interpreter(question, _llm):
        return {"sources": ["text"], "rewritten_query": question, "is_labor_market": False}

    def run(article_mode):
        response, _ = asyncio.run(execute_engine_query(
            question="Como se distribuíram os investimentos em 2025?", engine=Engine(),
            interp_llm=object(), interpreter=interpreter, rag_type="t", rag_label="T",
            article_mode=article_mode,
        ))
        return response

    article, normal = run(True), run(False)
    assert article.answer.startswith("## Infraestrutura")
    assert "*Fonte: Fundação Seade, seade piesp anual 2025, p. 2" in article.answer
    assert "Fonte:" not in normal.answer
    for citation in article.numeric_citations:
        assert article.answer[citation.start:citation.end].strip() == citation.value.strip()


def test_rotulo_legivel_sem_extensao_e_sem_pagina_ausente():
    assert document_label(_node("", "a/seade_trabalho_regional.pdf", None)) == "Fundação Seade, seade trabalho regional"
