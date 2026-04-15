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
Responda SOMENTE com base nas informações fornecidas abaixo. Conhecimento externo é proibido.

Regras obrigatórias:

1. CITAÇÃO REAL — Toda afirmação factual deve citar o arquivo PDF de origem e a página.
   Formato: (Fonte: nome_do_arquivo.pdf, p. X)
   Nunca cite "[Dados de Séries Temporais]" ou "[Dados Estruturados de Tabelas]" como fonte.
   Se um valor numérico extraído de tabela/série não tiver arquivo PDF identificável no \
contexto narrativo adjacente, não o utilize na resposta.

2. UMA FONTE POR AFIRMAÇÃO — Cada afirmação deve ser verificável em um único trecho do \
contexto. Não combine fragmentos de trechos distintos para criar uma afirmação nova que \
nenhum trecho expressa diretamente.

3. DADOS ESTRUTURADOS SEM RÓTULOS — Se a seção de séries temporais ou tabelas contiver \
apenas números sem rótulos claros de indicador e período, ignore essa seção inteiramente \
e baseie a resposta somente no contexto narrativo.

4. CONFLITO DE DADOS — Se um valor numérico na seção estruturada divergir do contexto \
narrativo, prevaleça o contexto narrativo.

5. AUSÊNCIA DE DADOS — Se a informação não está no contexto, responda exatamente:
   'A informação não consta nos documentos fornecidos.'

6. EVIDÊNCIA PARCIAL — Responda apenas o que está documentado e declare explicitamente \
o que está faltando.

7. CÁLCULOS — Se a pergunta pede diferença, variação ou comparação e os dois valores \
estão no contexto com fontes identificáveis, calcule e mostre (ex: 3,4% − 2,8% = 0,6 p.p.).

8. SEM CONCLUSÕES ALÉM DO TEXTO — Não escreva parágrafos de síntese com afirmações \
que vão além do que está literal ou numericamente nos trechos fornecidos.

Linguagem clara, direta e profissional.
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
    ts_nodes: list | None = None,
) -> str:
    sections = []

    # Consolida texto narrativo: nodes de texto + nodes de timeseries sem dados estruturados
    narrative_parts = []
    if text_nodes:
        narrative_parts.extend(n.get_content() for n in text_nodes)
    if not ts_data and ts_nodes:
        # Timeseries não produziu dados estruturados — usa conteúdo bruto como narrativa
        narrative_parts.extend(n.get_content() for n in ts_nodes)
    if narrative_parts:
        sections.append(
            "[Contexto Narrativo dos Documentos]\n" + "\n\n---\n\n".join(narrative_parts)
        )

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

        results = await asyncio.wait_for(asyncio.gather(*coros), timeout=120.0)
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
        context_block = _build_context_block(text_nodes, tables_data, ts_data, ts_nodes)

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
