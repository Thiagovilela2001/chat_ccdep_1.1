import asyncio
from pathlib import Path
from types import SimpleNamespace

from rag_core.domain_skills import DomainSkillRegistry, build_domain_prompt_block
from rag_principal.src.analysis_engine import AnalysisEngine, _build_context_block


REPO_ROOT = Path(__file__).resolve().parents[2]


def _registry(monkeypatch):
    monkeypatch.setenv("RAG_MAX_DOMAIN_SKILLS", "2")
    return DomainSkillRegistry(str(REPO_ROOT / "rag_principal"))


def test_registry_discovers_all_domain_skills(monkeypatch):
    registry = _registry(monkeypatch)

    assert set(registry.available_domains()) == {
        "demography",
        "economic_conjuncture",
        "investment_trade",
        "labor_market",
        "social_protection",
        "sectoral_regional",
    }


def test_registry_matches_accents_and_combines_domains(monkeypatch):
    registry = _registry(monkeypatch)

    matches = registry.match(
        "Compare investimentos anunciados na indústria automotiva paulista."
    )

    assert [skill.domain for skill in matches] == [
        "investment_trade",
        "sectoral_regional",
    ]


def test_registry_prioritizes_labor_and_sector_context(monkeypatch):
    registry = _registry(monkeypatch)

    block = registry.get_prompt_block(
        "Como evoluiu o emprego formal na indústria automotiva?"
    )

    assert "Mercado de Trabalho" in block
    assert "Análise Setorial e Regional" in block
    assert "Não introduzir fatos setoriais externos" in block


def test_registry_combines_investment_with_sector_question(monkeypatch):
    registry = _registry(monkeypatch)

    matches = registry.match(
        "Quais setores receberam mais investimentos anunciados?"
    )

    assert [skill.domain for skill in matches] == [
        "investment_trade",
        "sectoral_regional",
    ]


def test_registry_combina_conjuntura_e_regioes_em_dinamismo_economico(monkeypatch):
    registry = _registry(monkeypatch)

    matches = registry.match(
        "Quais regiões paulistas apresentaram maior dinamismo econômico?"
    )

    assert [skill.domain for skill in matches] == [
        "economic_conjuncture",
        "sectoral_regional",
    ]


def test_registry_matches_demographic_question(monkeypatch):
    registry = _registry(monkeypatch)

    matches = registry.match(
        "Como o envelhecimento populacional mudou a razão de dependência em São Paulo?"
    )

    assert matches[0].domain == "demography"
    assert "Análise Demográfica" in registry.get_prompt_block(
        "Como evoluiu a estrutura etária paulista?"
    )
    assert "Não tratar Censo, estimativa intercensitária e projeção" in (
        matches[0].context
    )


def test_registry_matches_social_protection_question(monkeypatch):
    registry = _registry(monkeypatch)

    matches = registry.match(
        "Qual era o perfil dos inscritos no CadÚnico paulista?"
    )

    assert matches[0].domain == "social_protection"
    assert "Proteção Social" in registry.get_prompt_block(
        "Como evoluiu o Programa Bolsa Família em São Paulo?"
    )


def test_registry_combines_social_protection_and_demography(monkeypatch):
    registry = _registry(monkeypatch)

    matches = registry.match(
        "Compare o BPC para idosos com o envelhecimento populacional."
    )

    assert [skill.domain for skill in matches] == [
        "demography",
        "social_protection",
    ]


def test_generic_income_does_not_trigger_social_protection(monkeypatch):
    registry = _registry(monkeypatch)

    matches = registry.match("Como evoluiu a renda do trabalho?")

    assert [skill.domain for skill in matches] == ["labor_market"]


def test_forced_labor_domain_preserves_legacy_interpreter_signal(monkeypatch):
    registry = _registry(monkeypatch)

    block = build_domain_prompt_block(
        registry,
        "Qual foi o resultado?",
        is_labor_market=True,
    )

    assert "Mercado de Trabalho" in block


def test_unmatched_question_does_not_inject_context(monkeypatch):
    registry = _registry(monkeypatch)

    assert registry.get_prompt_block("Qual documento foi publicado primeiro?") == ""


def test_analysis_engine_injects_registry_context():
    class Retriever:
        def __init__(self, result):
            self.result = result

        def retrieve(self, _question):
            return self.result

    class Registry:
        def is_loaded(self):
            return True

        def get_prompt_block(self, _question, *, forced_domains=()):
            assert forced_domains == ()
            return "\n[DOMÍNIO ECONÔMICO]\n"

    class LLM:
        prompt = ""

        def complete(self, prompt):
            self.prompt = prompt
            return SimpleNamespace(text="Resposta.")

    node = SimpleNamespace(metadata={"source_file": "fonte.pdf", "page": 1})
    llm = LLM()
    engine = AnalysisEngine(
        Retriever([]),
        Retriever(("Indicador: 10", [node])),
        Retriever(None),
        llm,
        domain_skills=Registry(),
    )

    answer, _sources = asyncio.run(
        engine.answer(
            question="Como evoluiu o PIB?",
            sources=["tables"],
            rewritten_query="PIB",
        )
    )

    assert answer == "Resposta."
    assert "[DOMÍNIO ECONÔMICO]" in llm.prompt


def test_contexto_prioriza_series_e_tabelas_antes_da_narrativa():
    node = SimpleNamespace(
        metadata={"source_file": "fonte.pdf", "page": 1},
        get_content=lambda: "Narrativa extensa.",
    )

    block = _build_context_block(
        [node],
        "Tabela regional completa",
        [node],
        "Série temporal completa",
        [node],
    )

    assert block.index("[Dados de Séries Temporais]") < block.index(
        "[Dados Estruturados de Tabelas]"
    )
    assert block.index("[Dados Estruturados de Tabelas]") < block.index(
        "[Contexto Narrativo dos Documentos]"
    )
