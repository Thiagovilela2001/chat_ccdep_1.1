import re
import pandas as pd
from llama_index.core.postprocessor import SentenceTransformerRerank

# ── Prompts ──────────────────────────────────────────────────────────────────

EXTRACT_PROMPT = """\
Você é um extrator de dados. Leia os trechos abaixo e extraia os dados numéricos \
necessários para responder à pergunta.

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

CALCULATE_PROMPT = """\
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

SYNTHESIZE_PROMPT = """\
Você é um analista econômico. Use o resultado do cálculo abaixo para redigir uma \
resposta clara, direta e profissional.
Não invente informações além do que está no resultado e nos trechos fornecidos.
Se o resultado contiver números, destaque-os explicitamente.

Citação de fontes (obrigatório):
- Toda afirmação com dado numérico ou fato específico deve ser seguida de uma citação inline.
- Formato: (Fonte: [nome_do_arquivo], p. [página])
- Use exatamente o nome do arquivo e número de página que aparecem nos trechos de referência.
- Exemplo: 'O crescimento foi de 3,1% no período (Fonte: Boletim_Conjuntura_3Trim.pdf, p. 22).'

Pergunta: {question}

Resultado calculado: {resultado}

Trechos de referência:
{context}

Resposta:"""

# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_code_block(text: str) -> str:
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

# ── Engine ───────────────────────────────────────────────────────────────────

class CalculationEngine:
    """Pipeline para perguntas que exigem cálculo numérico.

    Fluxo:
        retrieve (top 20) → rerank (top 5) → LLM extrai dados → exec(pandas)
        → LLM sintetiza resposta
    """

    def __init__(self, retriever, reranker: SentenceTransformerRerank, llm):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm

    def answer(self, question: str) -> tuple:
        """Retorna (resposta_texto: str, source_nodes: list)."""

        # 1. Retrieve + rerank
        nodes = self.retriever.retrieve(question)
        nodes = self.reranker.postprocess_nodes(nodes, query_str=question)
        context = "\n\n---\n\n".join(n.get_content() for n in nodes)

        # 2. Extrair dados estruturados
        extract_resp = self.llm.complete(
            EXTRACT_PROMPT.format(context=context, question=question)
        )
        extract_code = _extract_code_block(extract_resp.text)

        namespace = {"pd": pd}
        try:
            exec(extract_code, namespace)  # noqa: S102
        except Exception as exc:
            return f"[ERRO na extração de dados] {exc}", nodes

        df = namespace.get("df")
        data = namespace.get("data")

        if df is not None:
            data_preview = df.to_string(max_rows=20)
        elif data is not None:
            data_preview = str(data)
        else:
            return "Não foi possível extrair dados estruturados para o cálculo.", nodes

        # 3. Calcular via pandas
        calc_resp = self.llm.complete(
            CALCULATE_PROMPT.format(question=question, data_preview=data_preview)
        )
        calc_code = _extract_code_block(calc_resp.text)

        try:
            exec(calc_code, namespace)  # noqa: S102
        except Exception as exc:
            return f"[ERRO no cálculo] {exc}", nodes

        resultado = namespace.get("resultado", "Resultado não disponível.")

        # 4. Síntese em linguagem natural
        synth_resp = self.llm.complete(
            SYNTHESIZE_PROMPT.format(
                question=question,
                resultado=resultado,
                context=context,
            )
        )

        return synth_resp.text.strip(), nodes
