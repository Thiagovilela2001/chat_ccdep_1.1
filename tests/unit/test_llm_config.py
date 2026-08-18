from rag_ccdep.core import llm


def test_maritaca_is_the_default_provider(monkeypatch):
    for name in (
        "RAG_LLM_PROVIDER",
        "RAG_LLM_MODEL",
        "RAG_INTERP_MODEL",
        "RAG_LLM_BASE_URL",
        "RAG_LLM_API_KEY",
        "RAG_LLM_MAX_TOKENS",
        "RAG_REASONING_EFFORT",
        "RAG_REASONING_EXCLUDE",
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


def test_ollama_uses_local_defaults_without_api_key(monkeypatch):
    monkeypatch.setenv("RAG_LLM_PROVIDER", "ollama")
    for name in (
        "RAG_LLM_MODEL",
        "RAG_INTERP_MODEL",
        "RAG_LLM_BASE_URL",
        "RAG_LLM_API_KEY",
        "OLLAMA_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    config = llm._cfg()

    assert config["base_url"] == "http://127.0.0.1:11434/v1"
    assert config["main_model"] == "qwen3:4b-instruct"
    assert config["interp_model"] == "qwen3:4b-instruct"
    assert config["api_key"] == "ollama"
    llm.require_api_key()


def test_ollama_limits_local_concurrency(monkeypatch):
    monkeypatch.setenv("RAG_LLM_PROVIDER", "ollama")
    monkeypatch.delenv("RAG_LLM_CONCURRENCY", raising=False)
    assert llm.llm_concurrency(default=8) == 1

    monkeypatch.setenv("RAG_LLM_CONCURRENCY", "2")
    assert llm.llm_concurrency(default=8) == 2


def test_openrouter_defaults_to_gpt_oss_with_reasoning(monkeypatch):
    monkeypatch.setenv("RAG_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    for name in (
        "RAG_LLM_MODEL",
        "RAG_INTERP_MODEL",
        "RAG_POPUP_MODEL",
        "RAG_LLM_BASE_URL",
        "RAG_LLM_API_KEY",
        "RAG_LLM_MAX_TOKENS",
        "RAG_REASONING_EFFORT",
        "RAG_REASONING_EXCLUDE",
    ):
        monkeypatch.delenv(name, raising=False)

    config = llm._cfg()

    assert config["base_url"] == "https://openrouter.ai/api/v1"
    assert config["main_model"] == "openai/gpt-oss-120b"
    assert config["api_key"] == "test-key"
    assert config["max_tokens"] == 4096
    assert llm.chat_completion_kwargs() == {
        "extra_body": {"reasoning": {"effort": "low", "exclude": True}}
    }


def test_nvidia_provider_accepts_nemotron_overrides(monkeypatch):
    monkeypatch.setenv("RAG_LLM_PROVIDER", "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setenv("RAG_LLM_MODEL", "nvidia/nemotron-3-super-120b-a12b")
    monkeypatch.setenv("RAG_INTERP_MODEL", "nvidia/nemotron-3-super-120b-a12b")
    monkeypatch.setenv("RAG_REASONING_EFFORT", "medium")
    monkeypatch.setenv("RAG_REASONING_EXCLUDE", "0")

    config = llm._cfg()

    assert config["base_url"] == "https://integrate.api.nvidia.com/v1"
    assert config["main_model"] == "nvidia/nemotron-3-super-120b-a12b"
    assert config["interp_model"] == "nvidia/nemotron-3-super-120b-a12b"
    assert llm.chat_completion_kwargs() == {
        "extra_body": {"reasoning": {"effort": "medium", "exclude": False}}
    }
