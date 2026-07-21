from rag_core import llm


def test_maritaca_is_the_default_provider(monkeypatch):
    for name in (
        "RAG_LLM_PROVIDER",
        "RAG_LLM_MODEL",
        "RAG_INTERP_MODEL",
        "RAG_LLM_BASE_URL",
        "RAG_LLM_API_KEY",
        "MARITACA_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    assert llm.provider_name() == "maritaca"
    assert llm.main_model() == "sabia-4"
    assert llm.interp_model() == "sabia-4"
    assert llm._cfg()["base_url"] == "https://chat.maritaca.ai/api"
    assert llm._cfg()["context_window"] == 128000


def test_llm_environment_overrides_are_preserved(monkeypatch):
    monkeypatch.setenv("RAG_LLM_PROVIDER", "maritaca")
    monkeypatch.setenv("RAG_LLM_MODEL", "modelo-principal")
    monkeypatch.setenv("RAG_INTERP_MODEL", "modelo-interprete")
    monkeypatch.setenv("RAG_LLM_BASE_URL", "https://example.invalid/api")
    monkeypatch.setenv("RAG_LLM_API_KEY", "test-key")

    config = llm._cfg()

    assert config["main_model"] == "modelo-principal"
    assert config["interp_model"] == "modelo-interprete"
    assert config["base_url"] == "https://example.invalid/api"
    assert config["api_key"] == "test-key"
