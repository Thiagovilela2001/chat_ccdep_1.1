"""
rag_core/llm.py — fábrica única de LLM (OpenAI ou provedores compatíveis).

Todo o projeto fala com a API no formato OpenAI. Este módulo concentra a
escolha do provedor para que trocar de LLM seja só configuração de ambiente —
sem tocar em nenhuma engine. Os embeddings continuam locais (HuggingFace
bge-m3) e não passam por aqui.

Configuração por ambiente
--------------------------
    RAG_LLM_PROVIDER   "maritaca" (padrão) | "openai" | "ollama"
    RAG_LLM_MODEL      sobrescreve o modelo de síntese
    RAG_INTERP_MODEL   sobrescreve o modelo de interpretação/crítica
    RAG_EXTRACT_MODEL  sobrescreve o modelo da extração estruturada
                       (tabelas/séries temporais); padrão: o de síntese
    RAG_POPUP_MODEL    sobrescreve o modelo das explicações de citações
    RAG_LLM_BASE_URL   sobrescreve a base URL do provedor
    RAG_LLM_API_KEY    sobrescreve a chave (senão usa a chave padrão do provedor)
    RAG_LLM_REASONING_EFFORT
                       esforço de raciocínio enviado ao provedor. Vazio (padrão)
                       ou "auto"/"default" não enviar nada. Em modelo de
                       raciocínio, "none" corta o "pensamento" dos passos
                       mecânicos: medido no DeepSeek, a extração de série
                       temporal caiu de 9,7s (JSON vazio, orçamento consumido
                       pelo raciocínio) para 2,3s com o JSON completo.
                       "low" e "minimal" não reduziram nada lá.

Maritaca usa a chave `MARITACA_API_KEY`; OpenAI usa `OPENAI_API_KEY`.
Ollama roda localmente e não exige chave.

Dois consumidores
-----------------
- `make_llm(...)`          → objeto LLM do LlamaIndex (startups, processing).
                             Para provedores com base URL própria, registra o
                             modelo (ex.: "sabia-4") no catálogo do LlamaIndex
                             e usa a classe `OpenAI` base — sem depender do
                             pacote `openai-like`, cuja versão colide com o
                             `llama-index-llms-openai` já resolvido pelo core.
- `openai_client_kwargs()` → kwargs para `openai.OpenAI`/`AsyncOpenAI` cru
                             (agentic, self-rag, raptor, orchestrator analyzer).
"""
from __future__ import annotations

import os

from .logger import get_logger

log = get_logger(__name__)

_PROVIDERS: dict[str, dict] = {
    "maritaca": {
        "base_url": "https://chat.maritaca.ai/api",
        "key_env": "MARITACA_API_KEY",
        "main_model": "sabia-4",
        "interp_model": "sabia-4",
        "popup_model": "sabiazinho-4",
        "context_window": 128000,
    },
    "openai": {
        "base_url": None,  # SDK usa o endpoint padrão da OpenAI
        "key_env": "OPENAI_API_KEY",
        "main_model": "gpt-5-chat-latest",
        "interp_model": "gpt-5-mini",
        "popup_model": "gpt-5-mini",
        "context_window": 128000,
    },
    "ollama": {
        "base_url": "http://127.0.0.1:11434/v1",
        "key_env": "OLLAMA_API_KEY",
        # O cliente OpenAI exige um valor, embora o Ollama o ignore.
        "default_api_key": "ollama",
        "main_model": "qwen3:4b-instruct",
        "interp_model": "qwen3:4b-instruct",
        "popup_model": "qwen3:4b-instruct",
        "context_window": 32768,
    },
}


def provider_name() -> str:
    return os.getenv("RAG_LLM_PROVIDER", "maritaca").lower()


def _cfg() -> dict:
    cfg = dict(_PROVIDERS.get(provider_name(), _PROVIDERS["openai"]))
    if os.getenv("RAG_LLM_BASE_URL"):
        cfg["base_url"] = os.getenv("RAG_LLM_BASE_URL")
    cfg["api_key"] = (
        os.getenv("RAG_LLM_API_KEY")
        or os.getenv(cfg["key_env"])
        or cfg.get("default_api_key")
    )
    cfg["main_model"] = os.getenv("RAG_LLM_MODEL", cfg["main_model"])
    cfg["interp_model"] = os.getenv("RAG_INTERP_MODEL", cfg["interp_model"])
    cfg["extract_model"] = os.getenv("RAG_EXTRACT_MODEL") or cfg["main_model"]
    cfg["popup_model"] = os.getenv("RAG_POPUP_MODEL", cfg["popup_model"])
    return cfg


def main_model() -> str:
    return _cfg()["main_model"]


def interp_model() -> str:
    return _cfg()["interp_model"]


def extract_model() -> str:
    """Modelo dos passos mecânicos de extração estruturada (tabelas/séries)."""
    return _cfg()["extract_model"]


# Valores aceitos pela API compatível com OpenAI. Qualquer outro é ignorado para
# não transformar uma configuração errada em erro 400 na consulta.
_REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high"})
# Valores que significam "não enviar nada" e manter o padrão do provedor.
_REASONING_OFF = frozenset({"", "auto", "default"})


def _parse_reasoning_effort(var_name: str) -> str | None:
    raw = os.getenv(var_name, "").strip().lower()
    if raw in _REASONING_OFF:
        return None
    if raw not in _REASONING_EFFORTS:
        log.warning(
            "%s=%r ignorado; use %s",
            var_name,
            raw,
            ", ".join(sorted(_REASONING_EFFORTS)),
        )
        return None
    return raw


def main_reasoning_effort() -> str | None:
    """Esforço de raciocínio da síntese principal (`RAG_LLM_REASONING_EFFORT`)."""
    return _parse_reasoning_effort("RAG_LLM_REASONING_EFFORT")


def interp_reasoning_effort() -> str | None:
    """Esforço de raciocínio da interpretação/roteamento (`RAG_INTERP_REASONING_EFFORT`)."""
    return _parse_reasoning_effort("RAG_INTERP_REASONING_EFFORT")


def extract_reasoning_effort() -> str | None:
    """Esforço de raciocínio da extração estruturada (`RAG_EXTRACT_REASONING_EFFORT`)."""
    return _parse_reasoning_effort("RAG_EXTRACT_REASONING_EFFORT")


def article_reasoning_effort() -> str:
    """Esforço da síntese no modo artigo (`RAG_ARTICLE_REASONING_EFFORT`, padrão `low`)."""
    return _parse_reasoning_effort("RAG_ARTICLE_REASONING_EFFORT") or "low"


def reasoning_effort(role: str = "main") -> str | None:
    """
    Esforço de raciocínio por papel: 'main' (síntese), 'interp' (interpretação)
    ou 'extract' (extração). Preserva compatibilidade quando chamado sem argumentos.
    """
    if role in ("interp", "interpretation"):
        return interp_reasoning_effort()
    if role in ("extract", "extraction"):
        return extract_reasoning_effort()
    return main_reasoning_effort()


def popup_model() -> str:
    """Modelo leve dedicado à redação das explicações de citações numéricas."""
    return _cfg()["popup_model"]


def llm_concurrency(default: int = 4) -> int:
    """Limite de chamadas paralelas; Ollama local usa uma por vez por padrão."""
    raw = os.getenv("RAG_LLM_CONCURRENCY")
    if raw:
        try:
            return max(1, min(int(raw), 16))
        except ValueError:
            pass
    return 1 if provider_name() == "ollama" else default


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


def _register_model(model: str, context_window: int) -> None:
    """
    Registra um modelo fora do catálogo OpenAI (ex.: "sabia-4") no LlamaIndex,
    para a classe `OpenAI` base aceitá-lo sem exigir o pacote `openai-like`.
    Idempotente (`setdefault`).
    """
    from llama_index.llms.openai import utils as _ou

    _ou.ALL_AVAILABLE_MODELS.setdefault(model, context_window)
    _ou.CHAT_MODELS.setdefault(model, context_window)


def make_llm(
    *,
    interp: bool = False,
    extract: bool = False,
    role: str | None = None,
    temperature: float = 0.0,
    timeout: float = 60.0,
    model: str | None = None,
    max_retries: int | None = None,
    reasoning: str | None = None,
):
    """
    Cria o LLM do LlamaIndex para o provedor configurado.

    Papéis suportados:
    - síntese (padrão): model=main_model(), reasoning=RAG_LLM_REASONING_EFFORT
    - interp=True / role='interp': model=interp_model(), reasoning=RAG_INTERP_REASONING_EFFORT
    - extract=True / role='extract': model=extract_model(), reasoning=RAG_EXTRACT_REASONING_EFFORT
    `reasoning` sobrescreve o esforço do papel (ex.: LLM do modo artigo).
    """
    if interp and extract:
        raise ValueError("interp=True e extract=True são mutuamente exclusivos.")

    eff_role = role or ("interp" if interp else ("extract" if extract else "main"))

    from llama_index.llms.openai import OpenAI

    cfg = _cfg()
    if model:
        mdl = model
    elif eff_role == "interp":
        mdl = cfg["interp_model"]
    elif eff_role == "extract":
        mdl = cfg["extract_model"]
    else:
        mdl = cfg["main_model"]

    effort = reasoning or reasoning_effort(eff_role)
    adicional: dict = {"reasoning_effort": effort} if effort else {}
    retries: dict = {} if max_retries is None else {"max_retries": max_retries}

    if cfg["base_url"]:
        # Provedor compatível (Maritaca): registra o modelo e aponta a base URL.
        _register_model(mdl, cfg["context_window"])
        return OpenAI(
            model=mdl,
            api_base=cfg["base_url"],
            api_key=cfg["api_key"],
            temperature=temperature,
            timeout=timeout,
            additional_kwargs=adicional,
            **retries,
        )

    return OpenAI(
        model=mdl,
        api_key=cfg["api_key"],
        temperature=temperature,
        timeout=timeout,
        additional_kwargs=adicional,
        **retries,
    )


def make_rerank_llm():
    """LLM do rerank: falha rápido, sem retentativa do cliente.

    O rerank roda na frente da síntese e o TolerantLLMReranker já devolve a
    ordem híbrida quando um lote falha. Com o timeout de 30 s e as 3
    retentativas padrão do interp_llm, um único lote chegou a travar 32 s.
    """
    from .runtime import bounded_float

    return make_llm(
        interp=True,
        temperature=0.0,
        timeout=bounded_float("RAG_RERANK_TIMEOUT", 10.0, 2.0, 60.0),
        max_retries=0,
    )
