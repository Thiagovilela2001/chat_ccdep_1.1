"""
Text Retriever — retrieval híbrido (Vector + BM25) para chunks de texto narrativo.

Expõe também build_hybrid_retriever(), usado pelos outros retrievers.
"""
import os
import re
from collections import defaultdict

from llama_index.core.indices.utils import default_parse_choice_select_answer_fn
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer

from .logger import get_logger
from .llm import provider_name
from .runtime import bounded_int

log = get_logger(__name__)
_FALLBACK_TOP_N = 5

# Repetição do reranking quando o LLM não devolve veredito utilizável: em vez de
# aceitar a perda silenciosa, tenta uma vez com menos candidatos por chamada.
_RERANK_BATCH_SIZE = 30
_RERANK_RETRY_WINDOW = 20
_RERANK_RETRY_BATCH = 10


def _relevance(node) -> float:
    """Nota do reranker normalizada para ordenação, tolerando score ausente."""
    try:
        return float(getattr(node, "score", None) or 0.0)
    except (TypeError, ValueError):
        return 0.0

_QUERY_GEN_PROMPT = """\
Gere {num_queries} versões alternativas da consulta abaixo para ampliar a busca em
documentos econômicos em português. Preserve setor, local, período e indicador.
Retorne somente uma consulta por linha, sem numeração ou explicações.

Consulta: {query}
"""


def retrieval_top_k() -> int:
    return bounded_int("RAG_RETRIEVAL_TOP_K", 80, 20, 200)


def query_fusion_queries() -> int:
    return bounded_int("RAG_QUERY_FUSION_QUERIES", 2, 1, 4)


def rerank_candidate_limit() -> int:
    return bounded_int("RAG_RERANK_CANDIDATE_LIMIT", 40, 10, 100)


def rerank_top_n() -> int:
    return bounded_int("RAG_RERANK_TOP_N", 24, 5, 50)


def text_top_n() -> int:
    return bounded_int("RAG_TEXT_TOP_N", 20, 5, 40)


def structured_top_n() -> int:
    return bounded_int("RAG_STRUCTURED_TOP_N", 10, 3, 20)


def max_chunks_per_document() -> int:
    return bounded_int("RAG_MAX_CHUNKS_PER_DOCUMENT", 3, 1, 10)


def complete_coverage_chunks_per_document() -> int:
    return bounded_int("RAG_COMPLETE_COVERAGE_CHUNKS_PER_DOCUMENT", 8, 3, 20)


def llm_reranking_enabled() -> bool:
    """Define se os candidatos devem ser reranqueados por um LLM.

    No Ollama local, o lote de candidatos excede facilmente a janela ativa do
    modelo e o processamento em CPU adiciona dezenas de segundos à consulta.
    O ranking híbrido Vector+BM25 é usado por padrão nesse caso.
    """


def llm_reranking_enabled() -> bool:
    """Define se os candidatos devem ser reranqueados por um LLM.

    No Ollama local, o lote de candidatos excede facilmente a janela ativa do
    modelo e o processamento em CPU adiciona dezenas de segundos à consulta.
    O ranking híbrido Vector+BM25 é usado por padrão nesse caso.
    """
    raw = os.getenv("RAG_LLM_RERANK")
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return provider_name() != "ollama"


class ScoreReranker:
    """Reranker determinístico que preserva a ordem do retriever híbrido."""

    def __init__(self, top_n: int = _FALLBACK_TOP_N):
        self.top_n = top_n

    def postprocess_nodes(self, nodes, query_str: str | None = None):
        del query_str
        return list(nodes[:self.top_n])


class _RerankDiagnosis:
    """Guarda a última resposta crua do reranker para explicar saídas vazias."""

    def __init__(self) -> None:
        self.raw = ""


def _diagnosing_parser(diagnosis: _RerankDiagnosis):
    """Embrulha o parser padrão e registra por que ele não extraiu escolhas."""

    def parse(answer: str, num_choices: int, raise_error: bool = False):
        choices, relevances = default_parse_choice_select_answer_fn(
            answer, num_choices, raise_error
        )
        diagnosis.raw = answer or ""
        if not choices:
            excerpt = " ".join((answer or "").split())[:200]
            log.warning(
                "Reranker nao extraiu escolhas da resposta do LLM: %s",
                excerpt or "<resposta vazia>",
                extra={"fallback": True},
            )
        return choices, relevances

    return parse


class TolerantLLMReranker:
    """Reranker por LLM que não descarta candidatos quando um lote é recusado.

    O LLM às vezes responde em prosa, sem as linhas ``Doc: N, Relevance: R`` que o
    parser exige, e o lote inteiro era descartado — inclusive trechos que a resposta
    final acabava usando. Aqui o lote sem veredito volta na ordem híbrida, depois
    dos trechos que o modelo ranqueou, e uma repetição em janela menor só acontece
    quando nenhum lote devolveu veredito.

    Como os trechos aceitos vêm primeiro, a mudança é inerte quando o modelo
    ranqueia pelo menos ``top_n`` candidatos: nada além deles chega ao chamador.
    """

    def __init__(
        self,
        reranker,
        retry_reranker=None,
        batch_size: int = _RERANK_BATCH_SIZE,
        retry_window: int = _RERANK_RETRY_WINDOW,
        top_n: int | None = None,
    ):
        self._batch_reranker = reranker
        self._retry_reranker = retry_reranker or reranker
        self._batch_size = batch_size
        self._retry_window = retry_window
        self._top_n = top_n or rerank_top_n()

    def _accepted_in_batches(self, nodes, reranker, batch_size: int, query_str):
        """Devolve só os trechos ranqueados; lote recusado ou com erro fica de fora."""
        accepted: list = []
        for start in range(0, len(nodes), batch_size):
            batch = nodes[start : start + batch_size]
            try:
                result = list(reranker.postprocess_nodes(batch, query_str=query_str))
            except Exception as exc:
                log.warning(
                    "Reranker falhou no lote de %d (%s: %s) — lote segue na ordem híbrida",
                    len(batch),
                    type(exc).__name__,
                    exc,
                    extra={"fallback": True},
                )
                result = []
            accepted.extend(result)
        return accepted

    def postprocess_nodes(self, nodes, query_str: str | None = None):
        nodes = list(nodes or [])
        accepted = self._accepted_in_batches(
            nodes, self._batch_reranker, self._batch_size, query_str
        )

        if not accepted and len(nodes) > self._retry_window:
            log.warning(
                "Reranker sem veredito em %d candidatos — repetindo em janela de %d",
                len(nodes),
                self._retry_window,
                extra={"fallback": True},
            )
            accepted = self._accepted_in_batches(
                nodes[: self._retry_window],
                self._retry_reranker,
                min(_RERANK_RETRY_BATCH, self._retry_window),
                query_str,
            )

        if not accepted:
            log.warning(
                "Reranker sem veredito — mantendo %d candidatos na ordem híbrida",
                len(nodes),
                extra={"fallback": True},
            )
            return nodes

        accepted = sorted(accepted, key=_relevance, reverse=True)[: self._top_n]
        chosen = {id(node) for node in accepted}
        return accepted + [node for node in nodes if id(node) not in chosen]


def build_llm_reranker(
    llm, top_n: int | None = None, batch_size: int = _RERANK_BATCH_SIZE
) -> TolerantLLMReranker:
    """Cria o reranker por LLM com diagnóstico, tolerância por lote e uma repetição."""
    top_n = top_n or rerank_top_n()
    diagnosis = _RerankDiagnosis()
    # O lote é a unidade do LLMRerank: `top_n` igual ao lote evita perder escolhas
    # dentro da chamada; o corte final para `top_n` é do embrulho tolerante.
    reranker = LLMRerank(
        top_n=batch_size,
        choice_batch_size=batch_size,
        llm=llm,
        parse_choice_select_answer_fn=_diagnosing_parser(diagnosis),
    )
    retry_batch = min(_RERANK_RETRY_BATCH, batch_size)
    retry_reranker = LLMRerank(
        top_n=retry_batch,
        choice_batch_size=retry_batch,
        llm=llm,
        parse_choice_select_answer_fn=_diagnosing_parser(diagnosis),
    )
    return TolerantLLMReranker(
        reranker,
        retry_reranker,
        batch_size=batch_size,
        retry_window=_RERANK_RETRY_WINDOW,
        top_n=top_n,
    )


def _sanitize(text: str) -> str:
    """Remove caracteres de controle inválidos que podem quebrar o JSON da API."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


def _nodes_by_type(nodes, node_type: str | None) -> list:
    if node_type is None:
        return list(nodes or [])
    return [
        node for node in (nodes or [])
        if (getattr(node, "metadata", {}) or {}).get("type") == node_type
    ]


def deduplicate_nodes(nodes: list) -> list:
    """Remove nodes duplicados preservando ordem de relevância."""
    seen_ids = set()
    seen_texts = set()
    deduped = []
    for node in nodes or []:
        node_obj = getattr(node, "node", node)
        node_id = getattr(node_obj, "node_id", None) or getattr(node, "id_", None)
        text = str(getattr(node_obj, "text", "") or getattr(node, "text", "") or "").strip()
        text_key = re.sub(r"\s+", " ", text)[:300]

        if node_id and node_id in seen_ids:
            continue
        if text_key and text_key in seen_texts:
            continue

        if node_id:
            seen_ids.add(node_id)
        if text_key:
            seen_texts.add(text_key)
        deduped.append(node)
    return deduped


def _diversify_by_document(
    nodes,
    limit: int | None = None,
    per_document: int | None = None,
) -> list:
    """Prioriza variedade documental sem descartar resultados relevantes."""
    limit = limit or text_top_n()
    per_document = per_document or max_chunks_per_document()
    nodes = deduplicate_nodes(nodes)
    selected, overflow = [], []
    counts: dict[str, int] = defaultdict(int)

    for node in nodes:
        metadata = getattr(node, "metadata", {}) or {}
        source = str(
            metadata.get("source_files")
            or metadata.get("source_file")
            or metadata.get("file_name")
            or "?"
        ).lower()
        if counts[source] < per_document:
            selected.append(node)
            counts[source] += 1
        else:
            overflow.append(node)

    return (selected + overflow)[:limit]


def build_hybrid_retriever(index, bm25_nodes, *, node_type: str | None = None, llm=None):
    """
    Cria retriever híbrido (Vector + BM25 com fusão Reciprocal Rank).
    Se não houver nós BM25, retorna apenas o retriever vetorial.
    """
    top_k = retrieval_top_k()
    filters = None
    if node_type is not None:
        filters = MetadataFilters(filters=[MetadataFilter(key="type", value=node_type)])
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
        filters=filters,
    )

    typed_bm25_nodes = _nodes_by_type(bm25_nodes, node_type)
    if typed_bm25_nodes:
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=typed_bm25_nodes,
            similarity_top_k=top_k,
            language="portuguese",
            stemmer=Stemmer.Stemmer("portuguese"),
        )
        return QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            llm=llm,
            query_gen_prompt=_QUERY_GEN_PROMPT,
            similarity_top_k=top_k,
            num_queries=query_fusion_queries(),
            mode="reciprocal_rerank",
            use_async=False,
        )

    return vector_retriever


class TextRetriever:
    """
    Recupera chunks de texto narrativo relevantes para a query.

    Fluxo: retrieve (top-K configurável) → filtra texto → deduplica → rerank → diversidade.
    Fallback: se o reranker falhar ou retornar vazio, preserva o ranking híbrido.
    """

    def __init__(self, retriever, reranker):
        self._retriever = retriever
        self._reranker = reranker

    def retrieve(self, question: str) -> list:
        """Retorna nodes de texto reranqueados. Lista vazia se sem resultados."""
        nodes = self._retriever.retrieve(question)

        # Filtra apenas chunks narrativos (não tabelas) e deduplica
        text_nodes = deduplicate_nodes([n for n in nodes if n.metadata.get("type") != "table"])
        if not text_nodes:
            return []

        # Sanitiza conteúdo antes do reranker (evita erro 400 na API)
        for n in text_nodes:
            n.node.text = _sanitize(n.node.text)

        try:
            reranked = self._reranker.postprocess_nodes(
                text_nodes[:rerank_candidate_limit()],
                query_str=question,
            )
        except Exception as exc:
            log.warning(
                "Reranker falhou (%s: %s) — usando fallback por score",
                type(exc).__name__,
                exc,
                extra={"fallback": True},
            )
            reranked = []

        # Fallback: reranker vazio → top-N por score de recuperação
        per_document = (
            complete_coverage_chunks_per_document()
            if "todos os periodos disponiveis" in question.lower()
            else None
        )

        if not reranked:
            log.warning(
                "Reranker retornou vazio — usando fallback top-%d", text_top_n(),
                extra={"fallback": True},
            )
            return _diversify_by_document(
                text_nodes, text_top_n(), per_document=per_document
            )

        return _diversify_by_document(
            reranked, text_top_n(), per_document=per_document
        )
