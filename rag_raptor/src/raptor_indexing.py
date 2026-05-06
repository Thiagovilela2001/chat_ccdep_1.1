"""
RAPTOR tree builder.

Algoritmo (Recursive Abstractive Processing for Tree-Organized Retrieval):
1. Parte dos leaf nodes (chunks originais, nível 0)
2. Para cada nível:
   a. Gera embeddings dos nós correntes
   b. Agrupa por similaridade semântica (KMeans)
   c. Sumariza cada cluster via LLM → gera summary nodes (nível N)
3. Os summary nodes do nível N viram os leaf nodes do nível N+1
4. Todos os nós (folhas + resumos) são indexados juntos no ChromaDB

Resultado: a busca semântica recupera tanto detalhes específicos (nível 0)
quanto resumos de alto nível (nível N), melhorando o recall em perguntas
que exigem visão ampla do corpus.

Referência: Sarthi et al., "RAPTOR: Recursive Abstractive Processing
for Tree-Organized Retrieval" (2024).
"""
import math

import numpy as np
from openai import OpenAI
from sklearn.cluster import KMeans
from llama_index.core.schema import TextNode

from src.logger import get_logger

log = get_logger(__name__)

_SUMMARY_PROMPT = """\
Você é um analista de dados econômicos e estatísticos do Estado de São Paulo.
Abaixo estão trechos de documentos do Boletim de Conjuntura Paulista.
Escreva um resumo coeso e factual que integre as informações principais.
Preserve números, percentuais e datas relevantes. Máximo 400 palavras.

TRECHOS:
{chunks}

RESUMO:"""


# ── Helpers internos ─────────────────────────────────────────────────────────

def _get_embeddings(texts: list[str], embed_model) -> np.ndarray:
    """Gera matriz de embeddings para uma lista de textos."""
    vectors = [embed_model.get_text_embedding(t) for t in texts]
    return np.array(vectors, dtype=np.float32)


def _cluster_texts(embeddings: np.ndarray, n_clusters: int) -> list[list[int]]:
    """
    Agrupa embeddings em n_clusters com KMeans.
    Retorna lista de listas de índices — um grupo por cluster.
    """
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    clusters: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(idx)
    return list(clusters.values())


def _summarize_cluster(texts: list[str], client: OpenAI, model: str) -> str:
    """Sumariza um cluster de textos em um único parágrafo coeso."""
    # Limita a 20 trechos para não estourar o contexto do LLM
    combined = "\n\n---\n\n".join(texts[:20])
    prompt = _SUMMARY_PROMPT.format(chunks=combined)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=600,
    )
    return response.choices[0].message.content.strip()


# ── Função principal ──────────────────────────────────────────────────────────

def build_raptor_tree(
    leaf_nodes: list,
    embed_model,
    api_key: str,
    llm_model: str = "gpt-5-mini",
    max_levels: int = 3,
    min_cluster_size: int = 4,
) -> list:
    """
    Constrói a árvore RAPTOR sobre os leaf_nodes.
    Retorna todos os nós (folhas + resumos de todos os níveis).

    Parâmetros
    ----------
    leaf_nodes       : nós folha (chunks originais do corpus)
    embed_model      : modelo de embedding já configurado (BAAI/bge-m3)
    api_key          : chave OpenAI para sumarização
    llm_model        : modelo usado para sumarizar clusters
    max_levels       : profundidade máxima da árvore (padrão 3)
    min_cluster_size : mínimo de nós para tentar uma nova camada
    """
    client = OpenAI(api_key=api_key)

    # Marca leaf nodes com raptor_level=0
    for node in leaf_nodes:
        node.metadata["raptor_level"] = 0

    all_nodes: list = list(leaf_nodes)
    current_nodes: list = list(leaf_nodes)

    for level in range(1, max_levels + 1):
        if len(current_nodes) < min_cluster_size:
            log.info(
                "RAPTOR: nível %d — poucos nós (%d < %d), parando",
                level, len(current_nodes), min_cluster_size,
            )
            break

        # n_clusters ≈ √(n), mínimo 2, máximo n//2
        n_clusters = max(2, int(math.sqrt(len(current_nodes))))
        n_clusters = min(n_clusters, len(current_nodes) // 2)

        log.info(
            "RAPTOR: nível %d — %d nós → %d clusters",
            level, len(current_nodes), n_clusters,
        )

        texts = [n.get_content() for n in current_nodes]
        embeddings = _get_embeddings(texts, embed_model)
        cluster_indices = _cluster_texts(embeddings, n_clusters)

        summary_nodes: list = []
        for i, indices in enumerate(cluster_indices):
            cluster_texts = [texts[idx] for idx in indices]
            log.info(
                "RAPTOR: sumarizando cluster %d/%d (%d chunks)...",
                i + 1, n_clusters, len(cluster_texts),
            )
            summary = _summarize_cluster(cluster_texts, client, llm_model)

            # Agrega source_files dos nós do cluster
            source_files = list({
                current_nodes[idx].metadata.get("source_file", "raptor_summary")
                for idx in indices
            })

            node = TextNode(
                text=summary,
                metadata={
                    "raptor_level": level,
                    "source_file": source_files[0] if source_files else "raptor_summary",
                    "source_files": ", ".join(sorted(source_files)),
                    "cluster_size": len(indices),
                    "type": "text",
                },
            )
            summary_nodes.append(node)

        all_nodes.extend(summary_nodes)
        current_nodes = summary_nodes

        log.info(
            "RAPTOR: nível %d concluído — %d resumos gerados | %d nós no total",
            level, len(summary_nodes), len(all_nodes),
        )

    n_leaves = len(leaf_nodes)
    n_summaries = len(all_nodes) - n_leaves
    log.info(
        "RAPTOR: árvore completa — %d nós (%d folhas + %d resumos em %d nível/níveis)",
        len(all_nodes), n_leaves, n_summaries, min(max_levels, level),
    )
    return all_nodes
