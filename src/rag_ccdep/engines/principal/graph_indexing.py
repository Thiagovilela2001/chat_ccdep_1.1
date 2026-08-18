"""
Graph Indexing — constrói um PropertyGraphIndex de conhecimento sobre os
chunks de texto usando extração de entidades e relações via LLM.

Entidades: Indicador, Setor, Região, Período, FonteDados
Relações:  CRESCEU_EM, RECUOU_EM, PERTENCE_A, APLICA_SE_A, MEDIDO_POR, RELACIONA_COM

O grafo é persistido em {base_dir}/graph_store/ e reutilizado nas execuções
seguintes. É reconstruído automaticamente quando os documentos mudam
(force_rebuild=True).
"""
import os

from llama_index.core import PropertyGraphIndex
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.graph_stores.simple_labelled import LabelledPropertyGraph
from llama_index.core.graph_stores.types import KG_NODES_KEY, KG_RELATIONS_KEY
from llama_index.core.indices.property_graph import DynamicLLMPathExtractor
from llama_index.core.schema import TransformComponent
from llama_index.core.vector_stores import SimpleVectorStore

from rag_ccdep.core.logger import get_logger

try:
    import networkx as nx
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False

log = get_logger(__name__)

_GRAPH_DIR = "graph_store"
_GRAPH_FILE = "graph_store.json"
_GRAPH_VEC_FILE = "graph_vectors.json"

_ENTITY_TYPES = [
    "Indicador",   # ex: taxa de desocupação, PIB, IPCA, saldo de empregos
    "Setor",       # ex: indústria de transformação, comércio, construção civil
    "Região",      # ex: Estado de SP, RMSP, interior paulista
    "Período",     # ex: 1T2023, 2024, primeiro trimestre de 2022
    "FonteDados",  # ex: CAGED, PNAD Contínua, RAIS, IBGE, SEADE
]

_RELATION_TYPES = [
    "CRESCEU_EM",    # indicador cresceu em determinado período/região/setor
    "RECUOU_EM",     # indicador recuou em determinado período/região/setor
    "PERTENCE_A",    # indicador/dado pertence a setor ou categoria
    "APLICA_SE_A",   # dado se aplica a uma região geográfica
    "MEDIDO_POR",    # indicador é medido/divulgado por uma fonte de dados
    "RELACIONA_COM", # indicador se relaciona com outro indicador ou setor
]


def _normalize_entity_name(name: str) -> str:
    """
    Normaliza o nome de uma entidade para o formato que o LLMSynonymRetriever
    aplica às keywords da query (`.strip().capitalize()`). O matching do
    retriever é literal por id (= nome), então gravar em outro formato
    significa nunca casar.
    """
    return " ".join(str(name).split()).capitalize()


class EntityNormalizer(TransformComponent):
    """
    Normaliza nomes de entidades e endpoints de relações após a extração.

    O DynamicLLMPathExtractor gera variantes do mesmo conceito ("Taxa de
    desocupação" / "taxa de desocupação" / "TAXA DE DESOCUPAÇÃO"), o que
    fragmenta o grafo. Como o id do EntityNode é o próprio nome, normalizar
    aqui faz o upsert fundir as duplicatas em um único nó.
    """

    def __call__(self, nodes, **kwargs):
        for node in nodes:
            for ent in node.metadata.get(KG_NODES_KEY, []):
                if hasattr(ent, "name"):
                    ent.name = _normalize_entity_name(ent.name)
            for rel in node.metadata.get(KG_RELATIONS_KEY, []):
                rel.source_id = _normalize_entity_name(rel.source_id)
                rel.target_id = _normalize_entity_name(rel.target_id)
        return nodes


def _save_graph_store(graph_store: SimplePropertyGraphStore, path: str) -> None:
    """Persiste o grafo em UTF-8 (contorna limite do cp1252 no Windows)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(graph_store.graph.model_dump_json())


def _load_graph_store(path: str) -> SimplePropertyGraphStore:
    """Carrega o grafo de um JSON salvo em UTF-8."""
    with open(path, "r", encoding="utf-8") as f:
        graph = LabelledPropertyGraph.model_validate_json(f.read())
    return SimplePropertyGraphStore(graph=graph)


def export_graph_image(graph_store: SimplePropertyGraphStore, output_path: str) -> None:
    """Exporta o grafo como PNG usando networkx + matplotlib."""
    if not _HAS_NETWORKX:
        log.warning("[Graph] networkx/matplotlib não instalados — imagem não gerada")
        return

    triplets = graph_store.graph.triplets
    if not triplets:
        log.warning("[Graph] Grafo vazio — imagem não gerada")
        return

    G = nx.DiGraph()
    for triplet in triplets:
        src = triplet[0].name if hasattr(triplet[0], "name") else str(triplet[0])
        rel = triplet[1].label if hasattr(triplet[1], "label") else str(triplet[1])
        dst = triplet[2].name if hasattr(triplet[2], "name") else str(triplet[2])
        G.add_edge(src, dst, label=rel)

    fig, ax = plt.subplots(figsize=(24, 18))
    pos = nx.spring_layout(G, seed=42, k=2.5)
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color="#4C9BE8", alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=6, font_color="white", ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="#888", arrows=True, arrowsize=10, ax=ax)
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5, ax=ax)
    ax.axis("off")
    ax.set_title(f"Knowledge Graph — {G.number_of_nodes()} nós, {G.number_of_edges()} arestas", fontsize=12)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("[Graph] Imagem salva em %s", output_path)


def build_or_load_graph(
    text_nodes: list,
    base_dir: str,
    llm,
    force_rebuild: bool = False,
) -> tuple[PropertyGraphIndex, SimpleVectorStore | None]:
    """
    Constrói o PropertyGraphIndex a partir dos nós de texto ou carrega do disco.

    As entidades são embedadas (modelo local, via Settings.embed_model) e o
    vector store resultante é persistido ao lado do grafo — ele habilita o
    VectorContextRetriever como fallback quando o matching literal de
    sinônimos não encontra a entidade.

    Parâmetros
    ----------
    text_nodes : list
        Nós TextNode já processados (saída de process_documents).
    base_dir : str
        Diretório raiz do RAG (onde fica /chroma_db).
    llm :
        LLM para extração de entidades/relações.
    force_rebuild : bool
        Se True, reconstrói mesmo que o cache exista.

    Retorna
    -------
    (index, vector_store) — vector_store é None em caches antigos sem
    embeddings (retrieval segue funcionando só com sinônimos).
    """
    graph_dir = os.path.join(base_dir, _GRAPH_DIR)
    graph_path = os.path.join(graph_dir, _GRAPH_FILE)
    vec_path = os.path.join(graph_dir, _GRAPH_VEC_FILE)

    if not force_rebuild and os.path.exists(graph_path):
        try:
            log.info("[Graph] Carregando grafo existente de %s", graph_path)
            graph_store = _load_graph_store(graph_path)
            if not graph_store.graph.triplets:
                log.warning("[Graph] Grafo em cache vazio — reconstruindo")
            else:
                vec_store = None
                if os.path.exists(vec_path):
                    vec_store = SimpleVectorStore.from_persist_path(vec_path)
                else:
                    log.warning(
                        "[Graph] Cache sem embeddings de entidades — retrieval "
                        "vetorial desativado (reconstrua o grafo para habilitar)"
                    )
                image_path = os.path.join(graph_dir, "graph_store.png")
                if not os.path.exists(image_path):
                    export_graph_image(graph_store, image_path)
                index = PropertyGraphIndex.from_existing(
                    property_graph_store=graph_store,
                    vector_store=vec_store,
                    embed_kg_nodes=vec_store is not None,
                )
                return index, vec_store
        except Exception as exc:
            log.warning("[Graph] Cache inválido (%s) — reconstruindo", exc)

    log.info("[Graph] Construindo grafo de conhecimento...")
    os.makedirs(graph_dir, exist_ok=True)

    # Usa apenas nós de texto narrativo — tabelas/timeseries têm pouco contexto relacional
    narrative_nodes = [
        n for n in text_nodes
        if getattr(n, "metadata", {}).get("type", "text") == "text"
    ]
    if not narrative_nodes:
        narrative_nodes = text_nodes

    log.info("[Graph] Extraindo entidades e relações de %d nós", len(narrative_nodes))

    extractor = DynamicLLMPathExtractor(
        llm=llm,
        max_triplets_per_chunk=6,
        allowed_entity_types=_ENTITY_TYPES,
        allowed_relation_types=_RELATION_TYPES,
    )

    graph_store = SimplePropertyGraphStore()
    vec_store = SimpleVectorStore()
    index = PropertyGraphIndex(
        nodes=narrative_nodes,
        # EntityNormalizer roda após o extractor: funde variantes do mesmo
        # conceito e alinha os nomes ao formato esperado pelo synonym matching
        kg_extractors=[extractor, EntityNormalizer()],
        property_graph_store=graph_store,
        vector_store=vec_store,
        embed_kg_nodes=True,
        show_progress=True,
    )

    _save_graph_store(graph_store, graph_path)
    vec_store.persist(vec_path)
    log.info("[Graph] Grafo salvo em %s (embeddings em %s)", graph_path, vec_path)

    image_path = os.path.join(graph_dir, "graph_store.png")
    export_graph_image(graph_store, image_path)

    return index, vec_store
