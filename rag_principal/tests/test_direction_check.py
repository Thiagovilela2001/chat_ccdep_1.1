import importlib

from rag_core import answer_style
from rag_core.direction_check import find_direction_conflicts


def test_flags_inverted_share_from_real_answer():
    text = ("A comparação evidencia que houve redução expressiva na participação relativa "
            "do setor, que passou de 48% para 57% do total.")
    assert len(find_direction_conflicts(text)) == 1


def test_consistent_and_ambiguous_sentences_pass():
    ok = [
        "A taxa de desocupação recuou de 6,2% para 5,0% no período.",
        "O saldo cresceu de 900 mil para 1,2 milhão de postos.",
        "O investimento caiu de R$ 88,6 bilhões para R$ 63,4 bilhões, mas a participação subiu de 48% para 57%.",
        "A população de 18 a 24 anos diminuiu entre 2010 e 2024.",
        "Houve queda no indicador de 2023 para 2024.",
        "Famílias de baixa renda passaram de 25,0% para 26,1% do cadastro.",
        "O crescimento em quatro trimestres desacelerou de 3,1% para 2,4%.",
    ]
    for sentence in ok:
        assert find_direction_conflicts(sentence) == [], sentence


def test_scale_shared_between_operands():
    assert find_direction_conflicts("O cadastro aumentou de 14,5 para 14,1 milhões de pessoas.")


def test_strict_inference_is_default_and_can_be_disabled(monkeypatch):
    monkeypatch.delenv("RAG_STRICT_INFERENCE", raising=False)
    strict = importlib.reload(answer_style)
    assert strict.INTEGRATIVE_CLOSE not in strict.ANALYST_WRITING_GUIDE
    assert "DISCIPLINA INFERENCIAL" in strict.ANALYST_WRITING_GUIDE
    monkeypatch.setenv("RAG_STRICT_INFERENCE", "0")
    try:
        legacy = importlib.reload(answer_style)
        assert legacy.INTEGRATIVE_CLOSE in legacy.ANALYST_WRITING_GUIDE
    finally:
        monkeypatch.delenv("RAG_STRICT_INFERENCE")
        importlib.reload(answer_style)


def test_article_mode_uses_reasoning_llm_only_when_requested():
    from types import SimpleNamespace
    import asyncio

    from rag_principal.src.analysis_engine import AnalysisEngine

    class Retriever:
        def retrieve(self, _query):
            return ("Indicador: 10", [SimpleNamespace(metadata={"source_file": "f.pdf"})])

    class LLM:
        def __init__(self, name):
            self.name, self.calls = name, 0

        def complete(self, _prompt):
            self.calls += 1
            return SimpleNamespace(text=f"Resposta {self.name}.")

    base, article = LLM("base"), LLM("artigo")
    engine = AnalysisEngine(Retriever(), Retriever(), Retriever(), base, article_llm=article)
    ask = dict(question="Qual o indicador?", sources=["tables"], rewritten_query="indicador")

    assert engine.supports_article_mode
    assert asyncio.run(engine.answer(**ask))[0] == "Resposta base."
    assert asyncio.run(engine.answer(**ask, article_mode=True))[0] == "Resposta artigo."
    assert (base.calls, article.calls) == (1, 1)
    assert not AnalysisEngine(Retriever(), Retriever(), Retriever(), base).supports_article_mode


def test_query_service_sends_article_mode_only_to_supporting_engines(monkeypatch):
    import asyncio

    from rag_core.query_service import execute_engine_query

    monkeypatch.setenv("RAG_ANSWER_RETRIES", "0")
    seen = []

    class Legacy:  # assinatura sem article_mode, como as outras engines
        async def answer(self, *, question, sources, rewritten_query, is_labor_market):
            seen.append("legacy")
            return "Sem números.", []

    class Principal:
        supports_article_mode = True

        async def answer(self, *, article_mode=False, **_kwargs):
            seen.append(article_mode)
            return "Sem números.", []

    def interpreter(question, _llm):
        return {"sources": ["text"], "rewritten_query": question, "is_labor_market": False}

    for engine in (Legacy(), Principal()):
        asyncio.run(execute_engine_query(
            question="Pergunta?", engine=engine, interp_llm=object(), interpreter=interpreter,
            rag_type="t", rag_label="T", article_mode=True,
        ))
    assert seen == ["legacy", True]
