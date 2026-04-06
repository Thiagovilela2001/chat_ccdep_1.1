"""
Analysis Engine — agrega resultados dos três retrievers e sintetiza
uma única resposta via LLM.

Fluxo:
    sources selecionados → retrievers em paralelo (asyncio)
    → contexto unificado → LLM (síntese única) → (resposta, source_nodes)
"""
import asyncio

# ── Prompt de síntese ─────────────────────────────────────────────────────────

_SYNTHESIS_PROMPT = """\
Você é um analista especialista em dados econômicos e estatísticos do Estado de São Paulo.
Responda SOMENTE com base nas informações fornecidas abaixo. É proibido usar conhecimento externo.

Regras obrigatórias:
- Não invente números, datas, nomes, períodos ou relações causais.
- Se a resposta não estiver disponível nos dados fornecidos, diga: \
'A informação não consta nos documentos fornecidos.'
- Se houver evidência parcial, responda apenas com o que está documentado e explicite a limitação.
- Toda afirmação com dado numérico deve ser seguida de citação inline.
  Formato: (Fonte: [nome_do_arquivo], p. [página])
- Linguagem clara, direta e profissional. Destaque números e tendências relevantes.
{skill_block}
{context_block}

Pergunta: {question}

Resposta:"""

_SKILL_HEADER = """\

[Conhecimento Especializado — Mercado de Trabalho]
Use as definições e o checklist abaixo para interpretar corretamente os dados recuperados.
{skill_context}
[Fim do Conhecimento Especializado]
"""


def _build_context_block(
    text_nodes: list,
    tables_data: str | None,
    ts_data: str | None,
) -> str:
    sections = []

    if text_nodes:
        text_context = "\n\n---\n\n".join(n.get_content() for n in text_nodes)
        sections.append(f"[Contexto Narrativo dos Documentos]\n{text_context}")

    if tables_data:
        sections.append(f"[Dados Estruturados de Tabelas]\n{tables_data}")

    if ts_data:
        sections.append(f"[Dados de Séries Temporais]\n{ts_data}")

    return "\n\n" + "\n\n".join(sections) if sections else ""


# ── Engine ────────────────────────────────────────────────────────────────────

class AnalysisEngine:
    """
    Orquestra os retrievers em paralelo e sintetiza a resposta final com um
    único LLM call, combinando contexto narrativo + dados estruturados.
    """

    def __init__(self, text_retriever, tables_retriever, timeseries_retriever, llm,
                 labor_market_skill=None):
        self._text = text_retriever
        self._tables = tables_retriever
        self._ts = timeseries_retriever
        self._llm = llm
        self._labor_skill = labor_market_skill

    async def answer(
        self,
        question: str,
        sources: list[str],
        rewritten_query: str,
        is_labor_market: bool = False,
    ) -> tuple[str, list]:
        """
        Executa os retrievers necessários em paralelo e retorna
        (resposta_texto, all_source_nodes).
        """
        # Monta corrotinas apenas para as fontes selecionadas
        keys: list[str] = []
        coros = []
        if "text" in sources:
            keys.append("text")
            coros.append(asyncio.to_thread(self._text.retrieve, rewritten_query))
        if "tables" in sources:
            keys.append("tables")
            coros.append(asyncio.to_thread(self._tables.retrieve, rewritten_query))
        if "timeseries" in sources:
            keys.append("ts")
            coros.append(asyncio.to_thread(self._ts.retrieve, rewritten_query))

        results = await asyncio.gather(*coros)
        result_map = dict(zip(keys, results))

        # Coleta resultados
        text_nodes: list = result_map.get("text") or []
        tables_result = result_map.get("tables")
        ts_result = result_map.get("ts")

        tables_data: str | None = None
        tables_nodes: list = []
        if tables_result is not None:
            tables_data, tables_nodes = tables_result

        ts_data: str | None = None
        ts_nodes: list = []
        if ts_result is not None:
            ts_data, ts_nodes = ts_result

        all_source_nodes = text_nodes + tables_nodes + ts_nodes

        # Síntese: único LLM call com contexto unificado
        context_block = _build_context_block(text_nodes, tables_data, ts_data)

        if not context_block.strip():
            return "A informação não consta nos documentos fornecidos.", []

        skill_block = ""
        if is_labor_market and self._labor_skill and self._labor_skill.is_loaded():
            skill_block = _SKILL_HEADER.format(
                skill_context=self._labor_skill.get_context()
            )

        response = self._llm.complete(
            _SYNTHESIS_PROMPT.format(
                skill_block=skill_block,
                context_block=context_block,
                question=question,
            )
        )

        return response.text.strip(), all_source_nodes
