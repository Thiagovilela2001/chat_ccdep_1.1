"""
TimeSeries Retriever — recupera chunks de séries temporais via pandas.
"""
import re
import pandas as pd

from src.logger import get_logger
from src.safe_exec import safe_exec

log = get_logger(__name__)
_FALLBACK_TOP_N = 3

def _sanitize(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

_PERIOD_PATTERN = re.compile(
    r"([1-4][oO°]\s*trim)"
    r"|(\b(19|20)\d{2}\b)"
    r"|(jan(eiro)?|fev(ereiro)?|mar(ço)?|abr(il)?"
    r"|mai(o)?|jun(ho)?|jul(ho)?|ago(sto)?"
    r"|set(embro)?|out(ubro)?|nov(embro)?|dez(embro)?)"
    r"|(\b(período|data|mês|mes|trimestre|semestre|anual|mensal)\b)",
    re.IGNORECASE,
)

_TEMPORAL_KEYWORDS = {
    "mensal", "trimestral", "semestral", "bimestral",
    "semanal", "diário", "diaria", "diária",
}

_EXTRACT_PROMPT = """\
Você é um extrator de séries temporais. Leia os trechos abaixo e extraia os dados \
em formato de série temporal para responder à pergunta.

Retorne SOMENTE um bloco de código Python entre ```python e ```.
O código deve definir uma variável `df` como DataFrame pandas com colunas de período e valor.
Se os dados forem muito simples, defina um dicionário `data` com pares período→valor.
Regras:
- Não importe pandas — ele já está disponível como `pd`.
- Ordene os dados cronologicamente quando possível.
- Use nomes de colunas em português (ex: "Período", "Valor", "Variação").
- Não inclua nada fora do bloco de código.

Trechos:
{context}

Pergunta: {question}

```python
```"""

_ANALYZE_PROMPT = """\
Você tem uma série temporal disponível como `df` (DataFrame pandas) ou `data` (dict).
Escreva SOMENTE o código Python (entre ```python e ```) que analisa a série para responder à pergunta.
Armazene o resultado final como string descritiva na variável `resultado`.
Regras:
- Não importe nada. `pd` e `df`/`data` já estão disponíveis.
- Não use print().
- Ao acessar um valor de Series de um único elemento, use `.iloc[0]`.
- Calcule variações, tendências e estatísticas relevantes para a pergunta.
- Formate números com separador de milhar e 2 casas decimais.

Pergunta: {question}

Série temporal disponível:
{data_preview}

```python
```"""


def _is_temporal_table(node) -> bool:
    if node.metadata.get("type") != "table":
        return False
    gran = node.metadata.get("table_granularidade", "").lower()
    return any(kw in gran for kw in _TEMPORAL_KEYWORDS)


def _context_has_temporal_labels(context: str) -> bool:
    return bool(_PERIOD_PATTERN.search(context))


def _df_is_valid_timeseries(df) -> bool:
    if df is None or df.shape[0] < 2 or df.shape[1] < 2:
        return False
    first_col = df.iloc[:, 0]
    if pd.api.types.is_numeric_dtype(first_col):
        return False
    return True


def _extract_code_block(text: str) -> str:
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


class TimeSeriesRetriever:
    def __init__(self, retriever, reranker, llm):
        self._retriever = retriever
        self._reranker = reranker
        self._llm = llm

    def retrieve(self, question: str) -> tuple[str, list] | None:
        nodes = self._retriever.retrieve(question)

        ts_nodes = [n for n in nodes if _is_temporal_table(n)]
        if not ts_nodes:
            return None

        for n in ts_nodes:
            n.node.text = _sanitize(n.node.text)

        try:
            reranked = self._reranker.postprocess_nodes(ts_nodes, query_str=question)
        except Exception:
            log.warning("Reranker falhou em timeseries — usando fallback", extra={"fallback": True})
            reranked = []

        if not reranked:
            reranked = ts_nodes[:_FALLBACK_TOP_N]

        context = "\n\n---\n\n".join(n.get_content() for n in reranked)

        if not _context_has_temporal_labels(context):
            log.warning(
                "Contexto sem rotulos temporais — revertendo para narrativa",
                extra={"event": "ts_no_period_labels"},
            )
            return None, reranked

        structured = self._extract_and_analyze(question, context)
        if structured is None:
            log.warning(
                "Extracao retornou estrutura invalida — revertendo para narrativa",
                extra={"event": "ts_extraction_fallback"},
            )
            return None, reranked
        return structured, reranked

    def _extract_and_analyze(self, question: str, context: str) -> str:
        extract_resp = self._llm.complete(
            _EXTRACT_PROMPT.format(context=context, question=question)
        )
        extract_code = _extract_code_block(extract_resp.text)

        ns = {"pd": pd}
        try:
            safe_exec(extract_code, ns)
        except (ValueError, RuntimeError) as exc:
            log.warning("Extracao de serie temporal falhou: %s", exc)
            return f"[Erro na extração da série temporal: {exc}]"

        df = ns.get("df")
        data = ns.get("data")

        if df is not None:
            if not _df_is_valid_timeseries(df):
                log.warning(
                    "DataFrame extraido sem estrutura de serie temporal valida "
                    "(colunas: %s, shape: %s)",
                    list(df.columns), df.shape,
                    extra={"event": "ts_invalid_df"},
                )
                return None
            data_preview = df.to_string(max_rows=40)
        elif data is not None:
            data_preview = str(data)
        else:
            return "[Sem dados de série temporal extraídos]"

        analyze_resp = self._llm.complete(
            _ANALYZE_PROMPT.format(question=question, data_preview=data_preview)
        )
        analyze_code = _extract_code_block(analyze_resp.text)

        try:
            safe_exec(analyze_code, ns)
        except (ValueError, RuntimeError) as exc:
            log.warning("Analise de serie temporal falhou: %s", exc)
            return f"[Erro na análise da série temporal: {exc}]"

        return str(ns.get("resultado", data_preview))