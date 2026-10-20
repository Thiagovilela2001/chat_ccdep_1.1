"""
Analysis Engine — agrega resultados dos três retrievers e sintetiza
uma única resposta via LLM.

Fluxo:
    sources selecionados → retrievers em paralelo (asyncio)
    → contexto unificado → LLM (síntese única) → (resposta, source_nodes)
"""
import asyncio

from rag_core.answer_policy import REFUSAL_TEXT, sanitize_answer
from rag_core.answer_style import ANALYST_WRITING_GUIDE
from rag_core.direction_check import (
    REVISION_INSTRUCTION, direction_check_enabled, find_direction_conflicts,
)
from rag_core.domain_skills import build_domain_prompt_block
from rag_core.logger import get_logger
from rag_core.runtime import limit_context, request_timeout_seconds
from rag_core.provenance import format_source_context, source_labels
from rag_core.text_retriever import deduplicate_nodes, context_top_k

log = get_logger(__name__)

# ── Prompt de síntese ─────────────────────────────────────────────────────────

_SYNTHESIS_PROMPT = """\
Você é um analista de conjuntura econômica e estatística do Estado de São Paulo.
Sua tarefa é redigir uma análise que responda à pergunta do usuário com base
exclusivamente nas fontes fornecidas abaixo.

FIDELIDADE ÀS FONTES (inegociável)

1. Use somente informações presentes no contexto. Conhecimento externo é proibido.

2. RASTREABILIDADE — Todo fato e todo número da resposta deve ser rastreável a um \
trecho específico do contexto. Organizar, comparar e encadear fatos de trechos \
diferentes em uma mesma narrativa é permitido e esperado; criar fato novo, não: \
nenhuma afirmação causal, estimativa ou conclusão que nenhum trecho sustente, \
direta ou numericamente.

2A. NUMEROS NA RESPOSTA - Todo numero deve ser copiado literalmente de algum \
trecho do contexto recuperado ou ser resultado de uma conta explicita com operandos \
tambem presentes no contexto. Nao arredonde, nao converta unidades e nao estime. \
Se o numero necessario nao estiver no contexto recuperado, omita esse numero ou recuse.

3. SEPARAÇÃO DA EVIDÊNCIA — Use os rótulos de origem apenas internamente para \
verificar o suporte. Não os copie para a resposta. Não escreva citações, nomes \
de arquivos, páginas, abas ou listas de referências, salvo pedido explícito do usuário.
   Se um valor numérico extraído de tabela/série não tiver suporte identificável no \
contexto narrativo adjacente, não o utilize na resposta.

4. DADOS ESTRUTURADOS SEM RÓTULOS — Se a seção de séries temporais ou tabelas contiver \
apenas números sem rótulos claros de indicador e período, ignore essa seção inteiramente \
e baseie a resposta somente no contexto narrativo.

5. CONFLITO DE DADOS — Se um valor numérico na seção estruturada divergir do contexto \
narrativo, prevaleça o contexto narrativo.

6. AUSÊNCIA DE DADOS — Use exatamente a mensagem abaixo somente quando nenhum \
ponto central da pergunta puder ser respondido pelo contexto:
   '""" + REFUSAL_TEXT + """'

7. EVIDÊNCIA PARCIAL — Responda apenas o que está sustentado. Nunca concatene a \
mensagem do item 6 a uma resposta factual. Em perguntas com vários subitens, \
identifique brevemente apenas o subitem não determinado quando essa omissão puder \
induzir erro.

8. CÁLCULOS — Se a pergunta pede diferença, variação ou comparação e os dois valores \
estão sustentados no contexto, calcule e mostre a conta \
(ex: 3,4% − 2,8% = 0,6 p.p.).

""" + ANALYST_WRITING_GUIDE + """
{skill_block}
{context_block}

Pergunta: {question}

Resposta:"""

_ARTICLE_MODE_INSTRUCTION = """

FORMATO DE ARTIGO
- A primeira linha da resposta é um título no formato `## <título>`: até 12
  palavras, informativo (diz o achado principal, não repete a pergunta), sem
  ponto final.
- Quando apresentar dados em tabela Markdown, não escreva a fonte: ela é
  anexada automaticamente sob a tabela.
- Profundidade: o rigor inferencial restringe o que se conclui, não a extensão
  da análise. Não resuma nem condense. Desenvolva cada item, período e
  dimensão sustentados no contexto em parágrafo próprio, com seus números e
  a leitura que as fontes dão a eles. Uma tabela complementa o texto
  analítico, não o substitui."""


def _build_context_block(
    text_nodes: list,
    tables_data: str | None,
    tables_nodes: list | None,
    ts_data: str | None,
    ts_nodes: list | None = None,
    graph_nodes: list | None = None,
) -> str:
    sections = []

    # Consolida texto narrativo: nodes de texto + grafo + nodes de timeseries sem dados estruturados
    narrative_parts = []
    if text_nodes:
        narrative_parts.extend(format_source_context(n) for n in text_nodes)
    if graph_nodes:
        narrative_parts.extend(format_source_context(n) for n in graph_nodes)
    if not ts_data and ts_nodes:
        # Timeseries não produziu dados estruturados — usa conteúdo bruto como narrativa
        narrative_parts.extend(format_source_context(n) for n in ts_nodes)
    # Dados temporais e tabulares vêm primeiro para não serem truncados depois
    # de um bloco narrativo extenso. A síntese ainda usa a narrativa para contexto.
    if ts_data:
        labels = source_labels(ts_nodes or [])
        sections.append(f"[Dados de Séries Temporais]\n{labels}\n{ts_data}")

    if tables_data:
        labels = source_labels(tables_nodes or [])
        sections.append(f"[Dados Estruturados de Tabelas]\n{labels}\n{tables_data}")

    if narrative_parts:
        sections.append(
            "[Contexto Narrativo dos Documentos]\n" + "\n\n---\n\n".join(narrative_parts)
        )

    return "\n\n" + "\n\n".join(sections) if sections else ""


# ── Engine ────────────────────────────────────────────────────────────────────

class AnalysisEngine:
    """
    Orquestra os retrievers em paralelo e sintetiza a resposta final com um
    único LLM call, combinando contexto narrativo + dados estruturados.
    """

    def __init__(
        self,
        text_retriever,
        tables_retriever,
        timeseries_retriever,
        llm,
        domain_skills=None,
        labor_market_skill=None,
        graph_retriever=None,
        reranker=None,
        article_llm=None,
    ):
        self._article_llm = article_llm
        self._text = text_retriever
        self._tables = tables_retriever
        self._ts = timeseries_retriever
        self._llm = llm
        self._domain_skills = domain_skills
        self._labor_skill = labor_market_skill
        self._graph = graph_retriever
        self._reranker = reranker

    @property
    def supports_article_mode(self) -> bool:
        return self._article_llm is not None

    async def answer(
        self,
        question: str,
        sources: list[str],
        rewritten_query: str,
        is_labor_market: bool = False,
        article_mode: bool = False,
    ) -> tuple[str, list]:
        """
        Executa os retrievers necessários em paralelo e retorna
        (resposta_texto, all_source_nodes).

        `article_mode` troca só o LLM da síntese pelo de raciocínio: a avaliação
        científica mediu inferência 0,84 → 0,95 a ~2× a latência.
        """
        synth_llm = self._article_llm if article_mode and self._article_llm is not None else self._llm
        # Só o prompt de síntese recebe a instrução; o reranker continua com a pergunta pura.
        prompt_question = question + _ARTICLE_MODE_INSTRUCTION if article_mode else question
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

        results = await asyncio.wait_for(
            asyncio.gather(*coros, return_exceptions=True),
            timeout=request_timeout_seconds(),
        )
        result_map = {}
        for key, result in zip(keys, results):
            if isinstance(result, BaseException):
                log.warning("Retriever %s falhou; continuando com fontes parciais: %s", key, result)
            else:
                result_map[key] = result

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

        # Grafo: executa separadamente (precisa dos IDs dos nós já coletados para deduplicar)
        graph_nodes: list = []
        if "graph" in sources and self._graph is not None:
            existing_ids = {
                getattr(n.node if hasattr(n, "node") else n, "node_id", None)
                for n in text_nodes + tables_nodes + ts_nodes
            }
            try:
                graph_nodes = await asyncio.wait_for(
                    asyncio.to_thread(self._graph.retrieve, rewritten_query, existing_ids),
                    timeout=min(60.0, request_timeout_seconds()),
                )
            except Exception as exc:
                log.warning("GraphRetriever falhou; continuando sem grafo: %s", exc)

        # Unifica nós brutos de todas as fontes
        raw_candidates = deduplicate_nodes(
            text_nodes + tables_nodes + ts_nodes + graph_nodes
        )

        # Reranking global sobre os candidatos unificados
        if self._reranker is not None and raw_candidates:
            try:
                reranked = await asyncio.to_thread(
                    self._reranker.postprocess_nodes,
                    raw_candidates,
                    query_str=question,
                )
                if reranked:
                    all_source_nodes = list(reranked)
                else:
                    all_source_nodes = raw_candidates
            except Exception as exc:
                log.warning("Reranking global de fontes falhou: %s", exc)
                all_source_nodes = raw_candidates
        else:
            all_source_nodes = raw_candidates

        # Poda para o Top-K mais relevante
        all_source_nodes = all_source_nodes[:context_top_k()]

        # Filtra text_nodes para a síntese manter apenas trechos do Top-K
        kept_ids = {
            getattr(n.node if hasattr(n, "node") else n, "node_id", None)
            for n in all_source_nodes
        }
        filtered_text_nodes = [
            n for n in text_nodes
            if getattr(n.node if hasattr(n, "node") else n, "node_id", None) in kept_ids
        ]
        if not filtered_text_nodes and not tables_data and not ts_data:
            filtered_text_nodes = [
                n for n in all_source_nodes
                if getattr(n.node if hasattr(n, "node") else n, "metadata", {}).get("type") != "table"
            ]

        # Síntese: único LLM call com contexto unificado
        context_block = limit_context(
            _build_context_block(
                filtered_text_nodes, tables_data, tables_nodes, ts_data, ts_nodes, graph_nodes
            )
        )

        if not context_block.strip():
            return REFUSAL_TEXT, []

        skill_block = build_domain_prompt_block(
            self._domain_skills,
            question,
            is_labor_market=is_labor_market,
            legacy_labor_skill=self._labor_skill,
        )

        prompt = _SYNTHESIS_PROMPT.format(
            skill_block=skill_block,
            context_block=context_block,
            question=prompt_question,
        )
        response = await asyncio.to_thread(synth_llm.complete, prompt)
        answer = sanitize_answer(response.text, question=question)

        # Número certo, direção errada ("de 48% para 57%" chamado de redução):
        # uma única reescrita, mantida só se reduzir os conflitos.
        conflicts = find_direction_conflicts(answer) if direction_check_enabled() else []
        if conflicts:
            log.info("Checagem de direção: %d frase(s) em conflito; reescrevendo", len(conflicts))
            revision_prompt = _SYNTHESIS_PROMPT.format(
                skill_block=skill_block,
                context_block=context_block,
                question=prompt_question + REVISION_INSTRUCTION.format(
                    conflicts="\n".join(f"- {c}" for c in conflicts)
                ),
            )
            revised = await asyncio.to_thread(synth_llm.complete, revision_prompt)
            revised_answer = sanitize_answer(revised.text, question=question)
            if len(find_direction_conflicts(revised_answer)) < len(conflicts):
                answer = revised_answer

        return answer, all_source_nodes

