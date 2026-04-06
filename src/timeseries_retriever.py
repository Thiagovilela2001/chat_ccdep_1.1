"""
TimeSeries Retriever — recupera chunks de séries temporais (mensal/trimestral)
e os estrutura via pandas para o Analysis Engine.

"Temporal" = tabelas com granularidade sequencial ao longo do tempo, adequadas para
análise de tendências, crescimento e evolução de indicadores econômicos.
"""
import re
import pandas as pd

# Granularidades que indicam série temporal → incluídas neste retriever
_TEMPORAL_KEYWORDS = {
    "mensal", "trimestral", "semestral", "bimestral",
    "semanal", "diário", "diaria", "diária",
}

# ── Prompts ───────────────────────────────────────────────────────────────────

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
- Calcule variações, tendências e estatísticas relevantes para a pergunta.
- Formate números com separador de milhar e 2 casas decimais.

Pergunta: {question}

Série temporal disponível:
{data_preview}

```python
```"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_temporal_table(node) -> bool:
    """True se o node for tabela com granularidade temporal."""
    if node.metadata.get("type") != "table":
        return False
    gran = node.metadata.get("table_granularidade", "").lower()
    return any(kw in gran for kw in _TEMPORAL_KEYWORDS)


def _extract_code_block(text: str) -> str:
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


# ── Retriever ─────────────────────────────────────────────────────────────────

class TimeSeriesRetriever:
    """
    Recupera chunks de séries temporais e extrai dados estruturados via pandas.

    Fluxo: retrieve (top-20) → filtra séries temporais → rerank → extração + análise pandas
    Retorna (structured_data: str, nodes: list) ou None se sem dados temporais relevantes.
    """

    def __init__(self, retriever, reranker, llm):
        self._retriever = retriever
        self._reranker = reranker
        self._llm = llm

    def retrieve(self, question: str) -> tuple[str, list] | None:
        nodes = self._retriever.retrieve(question)

        ts_nodes = [n for n in nodes if _is_temporal_table(n)]
        if not ts_nodes:
            return None

        ts_nodes = self._reranker.postprocess_nodes(ts_nodes, query_str=question)
        if not ts_nodes:
            return None

        context = "\n\n---\n\n".join(n.get_content() for n in ts_nodes)
        structured = self._extract_and_analyze(question, context)
        return structured, ts_nodes

    def _extract_and_analyze(self, question: str, context: str) -> str:
        # Fase 1: extração da série temporal em DataFrame
        extract_resp = self._llm.complete(
            _EXTRACT_PROMPT.format(context=context, question=question)
        )
        extract_code = _extract_code_block(extract_resp.text)

        ns = {"pd": pd}
        try:
            exec(extract_code, ns)  # noqa: S102
        except Exception as exc:
            return f"[Erro na extração da série temporal: {exc}]"

        df = ns.get("df")
        data = ns.get("data")

        if df is not None:
            data_preview = df.to_string(max_rows=40)
        elif data is not None:
            data_preview = str(data)
        else:
            return "[Sem dados de série temporal extraídos]"

        # Fase 2: análise da série (tendências, variações)
        analyze_resp = self._llm.complete(
            _ANALYZE_PROMPT.format(question=question, data_preview=data_preview)
        )
        analyze_code = _extract_code_block(analyze_resp.text)

        try:
            exec(analyze_code, ns)  # noqa: S102
        except Exception as exc:
            return f"[Erro na análise da série temporal: {exc}]"

        return str(ns.get("resultado", data_preview))
