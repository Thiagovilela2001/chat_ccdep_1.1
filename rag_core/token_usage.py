"""
Contabilidade de tokens reais (campo `usage` devolvido pela API) por modelo.

Todo LLM do projeto passa pelo SDK `openai` (LlamaIndex por baixo ou cliente
cru), então basta envolver `chat.completions.create` sync e async. Uso:

    token_usage.install()
    antes = token_usage.snapshot()
    ...  # chamadas
    gasto = token_usage.diff(token_usage.snapshot(), antes)

Chamadas em streaming sem `usage` são contadas em `calls_without_usage`.
"""
import copy
import functools
import threading
from collections import defaultdict

_FIELDS = ("calls", "prompt_tokens", "completion_tokens", "cached_prompt_tokens",
           "reasoning_tokens", "calls_without_usage")
_lock = threading.Lock()
_totals: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(_FIELDS, 0))
_installed = False


def _record(model: str | None, response) -> None:
    usage = getattr(response, "usage", None)
    with _lock:
        row = _totals[model or "unknown"]
        row["calls"] += 1
        if usage is None:
            row["calls_without_usage"] += 1
            return
        row["prompt_tokens"] += usage.prompt_tokens or 0
        row["completion_tokens"] += usage.completion_tokens or 0
        # DeepSeek: prompt_cache_hit_tokens; OpenAI: prompt_tokens_details.cached_tokens
        cached = getattr(usage, "prompt_cache_hit_tokens", None)
        if cached is None:
            cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None)
        row["cached_prompt_tokens"] += cached or 0
        details = getattr(usage, "completion_tokens_details", None)
        row["reasoning_tokens"] += getattr(details, "reasoning_tokens", None) or 0


def install() -> None:
    """Idempotente: envolve create() das classes do SDK uma única vez."""
    global _installed
    if _installed:
        return
    from openai.resources.chat.completions import AsyncCompletions, Completions

    sync_create = Completions.create
    async_create = AsyncCompletions.create

    @functools.wraps(sync_create)
    def create(self, *args, **kwargs):
        response = sync_create(self, *args, **kwargs)
        _record(getattr(response, "model", None) or kwargs.get("model"), response)
        return response

    @functools.wraps(async_create)
    async def acreate(self, *args, **kwargs):
        response = await async_create(self, *args, **kwargs)
        _record(getattr(response, "model", None) or kwargs.get("model"), response)
        return response

    Completions.create = create
    AsyncCompletions.create = acreate
    _installed = True


def snapshot() -> dict[str, dict[str, int]]:
    with _lock:
        return copy.deepcopy(dict(_totals))


def diff(after: dict, before: dict) -> dict[str, dict[str, int]]:
    out = {}
    for model, row in after.items():
        base = before.get(model, {})
        delta = {k: row.get(k, 0) - base.get(k, 0) for k in _FIELDS}
        if any(delta.values()):
            out[model] = delta
    return out


def merge(*usages: dict) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(_FIELDS, 0))
    for usage in usages:
        for model, row in (usage or {}).items():
            for k in _FIELDS:
                out[model][k] += row.get(k, 0)
    return dict(out)
