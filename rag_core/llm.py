"""
rag_core/llm.py — fábrica única de LLM (OpenAI ou provedores compatíveis).

Todo o projeto fala com a API no formato OpenAI. Este módulo concentra a
escolha do provedor para que trocar de LLM seja só configuração de ambiente —
sem tocar em nenhuma engine. Os embeddings continuam locais (HuggingFace
bge-m3) e não passam por aqui.

Configuração por ambiente
--------------------------
    RAG_LLM_PROVIDER   "maritaca" (padrão) | "openai"
    RAG_LLM_MODEL      sobrescreve o modelo de síntese
    RAG_INTERP_MODEL   sobrescreve o modelo de interpretação/crítica
    RAG_LLM_BASE_URL   sobrescreve a base URL do provedor
    RAG_LLM_API_KEY    sobrescreve a chave (senão usa a chave padrão do provedor)

Maritaca usa a chave `MARITACA_API_KEY`; OpenAI usa `OPENAI_API_KEY`.

Dois consumidores
-----------------
- `make_llm(...)`          → objeto LLM do LlamaIndex (startups, processing).
                             Usa OpenAILike quando o provedor tem base URL
                             própria (nomes como "sabia-4" ficam fora do
                             catálogo validado pela classe OpenAI do LlamaIndex).
- `openai_client_kwargs()` → kwargs para `openai.OpenAI`/`AsyncOpenAI` cru
                             (agentic, self-rag, raptor, orchestrator analyzer).
"""
from __future__ import annotations

import os

_PROVIDERS: dict[str, dict] = {
    "maritaca": {
        "base_url": "https://chat.maritaca.ai/api",
        "key_env": "MARITACA_API_KEY",
        "main_model": "sabia-4",
        "interp_model": "sabia-4",
        "context_window": 32000,
    },
    "openai": {
        "base_url": None,  # SDK usa o endpoint padrão da OpenAI
        "key_env": "OPENAI_API_KEY",
        "main_model": "gpt-5-chat-latest",
        "interp_model": "gpt-5-mini",
        "context_window": 128000,
    },
}


def provider_name() -> str:
    return os.getenv("RAG_LLM_PROVIDER", "maritaca").lower()


def _cfg() -> dict:
    cfg = dict(_PROVIDERS.get(provider_name(), _PROVIDERS["openai"]))
    if os.getenv("RAG_LLM_BASE_URL"):
        cfg["base_url"] = os.getenv("RAG_LLM_BASE_URL")
    cfg["api_key"] = os.getenv("RAG_LLM_API_KEY") or os.getenv(cfg["key_env"])
    cfg["main_model"] = os.getenv("RAG_LLM_MODEL", cfg["main_model"])
    cfg["interp_model"] = os.getenv("RAG_INTERP_MODEL", cfg["interp_model"])
    return cfg


def main_model() -> str:
    return _cfg()["main_model"]


def interp_model() -> str:
    return _cfg()["interp_model"]


def require_api_key() -> None:
    """Valida a chave do provedor ativo; lança EnvironmentError com dica clara."""
    cfg = _cfg()
    if not cfg["api_key"]:
        env = "RAG_LLM_API_KEY" if os.getenv("RAG_LLM_API_KEY") is not None else cfg["key_env"]
        raise EnvironmentError(
            f"Chave do provedor '{provider_name()}' não encontrada. "
            f"Defina {env} no .env (ou troque RAG_LLM_PROVIDER)."
        )


def openai_client_kwargs() -> dict:
    """kwargs para instanciar `openai.OpenAI(**kwargs)` / `AsyncOpenAI(**kwargs)`."""
    cfg = _cfg()
    kwargs: dict = {"api_key": cfg["api_key"]}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]
    return kwargs


def make_llm(
    *,
    interp: bool = False,
    temperature: float = 0.0,
    timeout: float = 60.0,
    model: str | None = None,
):
    """
    Cria o LLM do LlamaIndex para o provedor configurado.

    interp=True usa o modelo de interpretação/crítica (mais leve, quando o
    provedor distingue); caso contrário, o modelo de síntese.
    """
    cfg = _cfg()
    mdl = model or (cfg["interp_model"] if interp else cfg["main_model"])

    if cfg["base_url"]:
        # Provedor compatível com nome de modelo fora do catálogo OpenAI.
        from llama_index.llms.openai_like import OpenAILike

        return OpenAILike(
            model=mdl,
            api_base=cfg["base_url"],
            api_key=cfg["api_key"],
            is_chat_model=True,
            context_window=cfg["context_window"],
            temperature=temperature,
            timeout=timeout,
        )

    from llama_index.llms.openai import OpenAI

    return OpenAI(model=mdl, api_key=cfg["api_key"], temperature=temperature, timeout=timeout)
