import json
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser
from llama_index.core.extractors import TitleExtractor, KeywordExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import TextNode
from rag_core.llm import interp_model, make_llm


# Tabelas com até este número de linhas são indexadas como um único chunk.
# Acima disso cada linha vira um chunk independente (série histórica).
SMALL_TABLE_MAX_ROWS = 10

_ENRICH_PROMPT = """\
Analise a tabela abaixo e retorne APENAS um JSON válido, sem blocos markdown, com os campos:

- "descricao": o que a tabela representa (1-2 frases)
- "periodo": intervalo de tempo coberto, se identificável (ex: "2010-2022"), senão ""
- "indicadores": lista com os principais indicadores/métricas presentes (máx. 5)
- "granularidade": nível de detalhe das linhas (ex: "anual por região", "mensal", "por faixa etária")

Tabela:
{table_text}

Exemplo de resposta:
{{"descricao": "Taxa de desemprego por região do Brasil.", "periodo": "2012-2022", "indicadores": ["taxa de desemprego", "região"], "granularidade": "anual por região"}}"""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _markdown_to_df(doc_text: str) -> pd.DataFrame:
    """Extrai o bloco de tabela markdown do texto do documento e converte para DataFrame."""
    lines = [l for l in doc_text.split("\n") if l.strip().startswith("|")]
    if not lines:
        return pd.DataFrame()

    # Remove a linha separadora  (|---|---|)
    lines = [l for l in lines if not re.match(r"^\|\s*[-:]+[\s|:-]*\|?\s*$", l)]

    if len(lines) < 2:
        return pd.DataFrame()

    header = [c.strip() for c in lines[0].split("|")[1:-1]]
    rows = []
    for line in lines[1:]:
        values = [v.strip() for v in line.split("|")[1:-1]]
        if len(values) == len(header):
            rows.append(dict(zip(header, values)))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Enriquecimento de metadados via LLM
# ---------------------------------------------------------------------------

def _enrich_table_metadata(table_doc, llm) -> dict:
    """
    Chama o LLM uma vez por tabela para gerar metadados semânticos.
    O resultado é propagado para todos os chunks derivados dessa tabela.
    """
    try:
        prompt = _ENRICH_PROMPT.format(table_text=table_doc.text[:3000])
        response = llm.complete(prompt)
        enriched = json.loads(response.text)
        return {
            "table_descricao":     enriched.get("descricao", ""),
            "table_periodo":       enriched.get("periodo", ""),
            "table_indicadores":   ", ".join(enriched.get("indicadores", [])),
            "table_granularidade": enriched.get("granularidade", ""),
        }
    except Exception as e:
        source = table_doc.metadata.get("source_file", "?")
        print(f"  Aviso: enriquecimento de metadados falhou para {source} — ({e})")
        return {}


# ---------------------------------------------------------------------------
# Chunking de tabelas
# ---------------------------------------------------------------------------

def _row_to_structured_text(row: dict, source_file: str) -> str:
    """Converte uma linha de DataFrame para texto estruturado (abordagem 3)."""
    lines = [f"{col}: {val}" for col, val in row.items() if str(val).strip()]
    lines.append(f"Fonte: {source_file}")
    return "\n".join(lines)


def _chunk_table(table_doc, extra_metadata: dict | None = None) -> list:
    """
    Estratégia de chunking para tabelas:

    Abordagem 1 — tabela inteira (tabelas pequenas, <= SMALL_TABLE_MAX_ROWS linhas)
        Gera 1 chunk com todas as linhas em texto estruturado.

    Abordagem 2/3 — linha como documento (séries históricas)
        Gera 1 chunk por linha em texto estruturado.

    O formato de saída é sempre texto estruturado (abordagem 3):
        Coluna1: valor1
        Coluna2: valor2
        Fonte: arquivo.pdf
    """
    df = _markdown_to_df(table_doc.text)
    if df.empty:
        return [TextNode(text=table_doc.text, metadata=table_doc.metadata)]

    source    = table_doc.metadata.get("source_file", "")
    base_meta = {**table_doc.metadata, **(extra_metadata or {})}
    nodes     = []

    if len(df) <= SMALL_TABLE_MAX_ROWS:
        # Abordagem 1: tabela inteira como texto estruturado
        rows_text = "\n---\n".join(
            _row_to_structured_text(row.to_dict(), source)
            for _, row in df.iterrows()
        )
        nodes.append(TextNode(
            text=rows_text,
            metadata={**base_meta, "chunk_strategy": "full_table"},
        ))
    else:
        # Abordagem 2/3: uma linha por chunk
        for _, row in df.iterrows():
            text = _row_to_structured_text(row.to_dict(), source)
            if text.strip():
                nodes.append(TextNode(
                    text=text,
                    metadata={**base_meta, "chunk_strategy": "row_per_chunk"},
                ))

    return nodes


# ---------------------------------------------------------------------------
# Pipeline de texto
# ---------------------------------------------------------------------------

def _get_text_pipeline(chunk_size=1024, chunk_overlap=200):
    splitter = LangchainNodeParser(
        RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    )
    return IngestionPipeline(transformations=[
        splitter,
        TitleExtractor(nodes=5),
        KeywordExtractor(keywords=5),
    ])


# ---------------------------------------------------------------------------
# Entrada principal
# ---------------------------------------------------------------------------

def process_documents(documents):
    """
    Processa documentos em dois caminhos:
    - Texto  → RecursiveCharacterTextSplitter + TitleExtractor + KeywordExtractor
    - Tabela → chunking determinístico em texto estruturado (abordagem 1 ou 2/3)
    """
    text_docs  = [d for d in documents if d.metadata.get("type") != "table"]
    table_docs = [d for d in documents if d.metadata.get("type") == "table"]

    print(f"  {len(text_docs)} documento(s) de texto | {len(table_docs)} tabela(s)")

    # --- Texto ---
    text_nodes = []
    if text_docs:
        print("  Processando documentos de texto...")
        text_nodes = _get_text_pipeline().run(documents=text_docs)

    # --- Tabelas ---
    table_nodes = []
    if table_docs:
        print(f"  Enriquecendo metadados e aplicando chunking nas tabelas ({interp_model()})...")
        llm = make_llm(interp=True, temperature=0.0, timeout=30.0)

        # Paraleliza as chamadas LLM de enriquecimento (I/O-bound → ThreadPoolExecutor)
        max_workers = min(8, len(table_docs))
        enriched: dict[int, dict] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_enrich_table_metadata, doc, llm): i
                       for i, doc in enumerate(table_docs)}
            for fut in as_completed(futures):
                enriched[futures[fut]] = fut.result()

        for i, doc in enumerate(table_docs):
            extra_meta = enriched[i]
            nodes      = _chunk_table(doc, extra_metadata=extra_meta)
            table_nodes.extend(nodes)
            strategy  = nodes[0].metadata.get("chunk_strategy", "?") if nodes else "?"
            source    = doc.metadata.get("source_file", "?")
            page      = doc.metadata.get("page", "?")
            descricao = extra_meta.get("table_descricao", "")
            print(f"    {source} p.{page} → {len(nodes)} chunk(s) [{strategy}] | {descricao[:60]}")

    all_nodes = text_nodes + table_nodes
    print(
        f"Normalização concluída. "
        f"{len(text_nodes)} nós de texto + {len(table_nodes)} nós de tabela = {len(all_nodes)} total."
    )
    return all_nodes
