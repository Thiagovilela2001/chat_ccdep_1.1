"""
Tables Retriever — recupera dados de tabelas estáticas (não temporais) e os
estrutura via pandas para o Analysis Engine.

"Estática" = tabela com valores pontuais, rankings ou comparações entre categorias
(ex: emprego por setor, PIB por região). Distingue-se de TimeSeries pela granularidade.
"""
import re
import pandas as pd

# Granularidades que indicam série temporal → excluídas deste retriever
_TEMPORAL_KEYWORDS = {
    "mensal", "trimestral", "semestral", "bimestral",
    "semanal", "diário", "diaria", "diária",
}

# ── Prompts ───────────────────────────────────────────────────────────────────

_EXTRACT_PROMPT = """\
Você é um extrator de dados. Leia os trechos de tabelas abaixo e extraia os dados \
numéricos necessários para responder à pergunta.

Retorne SOMENTE um bloco de código Python entre ```python e ```.
O código deve definir uma variável chamada `df` como um pandas DataFrame com os dados extraídos.
Se os dados forem poucos para uma tabela, defina um dicionário simples chamado `data`.
Regras:
- Não importe pandas — ele já está disponível como `pd`.
- Não inclua nada fora do bloco de código.
- Use nomes de colunas em português quando possível.

Trechos:
{context}

Pergunta: {question}

```python
```"""

_CALCULATE_PROMPT = """\
Você tem os dados abaixo disponíveis como variável `df` (DataFrame pandas) ou `data` (dict).
Escreva SOMENTE o código Python (entre ```python e ```) que calcula a resposta para a pergunta.
Armazene o resultado final como string formatada e legível na variável `resultado`.
Regras:
- Não importe nada. `pd` e `df`/`data` já estão disponíveis.
- Não use print().
- Formate números com separador de milhar e 2 casas decimais quando aplicável.

Pergunta: {question}

Dados disponíveis:
{data_preview}

```python
```"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_static_table(node) -> bool:
    """True se o node for tabela com granularidade não-temporal."""
    if node.metadata.get("type") != "table":
        return False
    gran = node.metadata.get("table_granularidade", "").lower()
    return not any(kw in gran for kw in _TEMPORAL_KEYWORDS)


def _extract_code_block(text: str) -> str:
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


# ── Retriever ─────────────────────────────────────────────────────────────────

class TablesRetriever:
    """
    Recupera chunks de tabelas estáticas e extrai dados estruturados via pandas.

    Fluxo: retrieve (top-20) → filtra tabelas estáticas → rerank → extração pandas
    Retorna (structured_data: str, nodes: list) ou None se sem tabelas relevantes.
    """

    def __init__(self, retriever, reranker, llm):
        self._retriever = retriever
        self._reranker = reranker
        self._llm = llm

    def retrieve(self, question: str) -> tuple[str, list] | None:
        nodes = self._retriever.retrieve(question)

        table_nodes = [n for n in nodes if _is_static_table(n)]
        if not table_nodes:
            return None

        table_nodes = self._reranker.postprocess_nodes(table_nodes, query_str=question)
        if not table_nodes:
            return None

        context = "\n\n---\n\n".join(n.get_content() for n in table_nodes)
        structured = self._extract_and_calculate(question, context)
        return structured, table_nodes

    def _extract_and_calculate(self, question: str, context: str) -> str:
        # Fase 1: extração de dados em DataFrame
        extract_resp = self._llm.complete(
            _EXTRACT_PROMPT.format(context=context, question=question)
        )
        extract_code = _extract_code_block(extract_resp.text)

        ns = {"pd": pd}
        try:
            exec(extract_code, ns)  # noqa: S102
        except Exception as exc:
            return f"[Erro na extração de dados da tabela: {exc}]"

        df = ns.get("df")
        data = ns.get("data")

        if df is not None:
            data_preview = df.to_string(max_rows=20)
        elif data is not None:
            data_preview = str(data)
        else:
            return "[Sem dados estruturados extraídos da tabela]"

        # Fase 2: cálculo via pandas
        calc_resp = self._llm.complete(
            _CALCULATE_PROMPT.format(question=question, data_preview=data_preview)
        )
        calc_code = _extract_code_block(calc_resp.text)

        try:
            exec(calc_code, ns)  # noqa: S102
        except Exception as exc:
            return f"[Erro no cálculo sobre a tabela: {exc}]"

        return str(ns.get("resultado", data_preview))
