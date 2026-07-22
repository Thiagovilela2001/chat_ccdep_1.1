import json
import os
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser
from llama_index.core.extractors import TitleExtractor, KeywordExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import TextNode
from rag_core.llm import interp_model, llm_concurrency, make_llm, provider_name


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


def llm_ingest_enrichment_enabled() -> bool:
    """Define se a indexação deve chamar o LLM para gerar metadados.

    Modelos locais são desativados por padrão: centenas de chamadas seriais
    tornam a primeira indexação muito lenta e um único timeout pode inutilizar
    todo o lote. O enriquecimento continua disponível por configuração.
    """
    raw = os.getenv("RAG_INGEST_LLM_ENRICHMENT")
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return provider_name() != "ollama"


def _infer_table_metadata(table_doc) -> dict:
    """Gera metadados úteis sem rede, usados como padrão/fallback."""
    df = _markdown_to_df(table_doc.text)
    columns = [str(column).strip() for column in df.columns if str(column).strip()][:5]
    source = str(table_doc.metadata.get("source_file") or "tabela")
    description = f"Tabela de {source}"
    if columns:
        description += f" com {', '.join(columns)}"

    text = table_doc.text.lower()
    years = sorted({int(year) for year in re.findall(r"\b(?:19|20)\d{2}\b", text)})
    if re.search(r"\b(?:jan(?:eiro)?|fev(?:ereiro)?|mar(?:ço)?|abr(?:il)?|mai(?:o)?|jun(?:ho)?|jul(?:ho)?|ago(?:sto)?|set(?:embro)?|out(?:ubro)?|nov(?:embro)?|dez(?:embro)?)\b", text):
        granularity = "mensal"
    elif re.search(r"\b[1-4](?:º|°|o)?\s*(?:tri|trim|trimestre)\b", text):
        granularity = "trimestral"
    elif len(years) >= 2:
        granularity = "anual"
    else:
        granularity = ""

    if len(years) >= 2:
        period = f"{years[0]}-{years[-1]}"
    elif years:
        period = str(years[0])
    else:
        period = ""

    return {
        "table_descricao": description,
        "table_periodo": period,
        "table_indicadores": ", ".join(columns),
        "table_granularidade": granularity,
    }


# ---------------------------------------------------------------------------
# Enriquecimento de metadados via LLM
# ---------------------------------------------------------------------------

def _enrich_table_metadata(table_doc, llm) -> dict:
    """
    Chama o LLM uma vez por tabela para gerar metadados semânticos.
    O resultado é propagado para todos os chunks derivados dessa tabela.
    """
    fallback = _infer_table_metadata(table_doc)
    if llm is None:
        return fallback

    try:
        prompt = _ENRICH_PROMPT.format(table_text=table_doc.text[:3000])
        response = llm.complete(prompt)
        enriched = json.loads(response.text)
        return {
            "table_descricao":     enriched.get("descricao") or fallback["table_descricao"],
            "table_periodo":       enriched.get("periodo") or fallback["table_periodo"],
            "table_indicadores":   ", ".join(enriched.get("indicadores", [])) or fallback["table_indicadores"],
            "table_granularidade": enriched.get("granularidade") or fallback["table_granularidade"],
        }
    except Exception as e:
        source = table_doc.metadata.get("source_file", "?")
        print(f"  Aviso: enriquecimento de metadados falhou para {source} — ({e})")
        return fallback


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

def _get_text_pipeline(
    llm=None,
    chunk_size=1024,
    chunk_overlap=200,
    enrich_metadata: bool | None = None,
):
    splitter = LangchainNodeParser(
        RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    )
    transformations = [splitter]
    if enrich_metadata is None:
        enrich_metadata = llm_ingest_enrichment_enabled()
    if enrich_metadata:
        if llm is None:
            raise ValueError("llm é obrigatório quando o enriquecimento está ativado")
        workers = llm_concurrency()
        transformations.extend([
            # Passe o provedor configurado explicitamente. Sem isso, os extractors
            # recorrem ao LLM global do LlamaIndex, cujo default é a OpenAI.
            # Falhas de metadados não devem descartar os nós já processados.
            TitleExtractor(
                llm=llm,
                nodes=5,
                num_workers=workers,
                raise_on_error=False,
            ),
            KeywordExtractor(
                llm=llm,
                keywords=5,
                num_workers=workers,
                raise_on_error=False,
            ),
        ])
    return IngestionPipeline(transformations=transformations)


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
    use_llm_enrichment = llm_ingest_enrichment_enabled()
    llm = (
        make_llm(interp=True, temperature=0.0, timeout=30.0)
        if use_llm_enrichment and (text_docs or table_docs)
        else None
    )

    print(f"  {len(text_docs)} documento(s) de texto | {len(table_docs)} tabela(s)")

    # --- Texto ---
    text_nodes = []
    if text_docs:
        mode = "com metadados via LLM" if use_llm_enrichment else "sem chamadas ao LLM"
        print(f"  Processando documentos de texto ({mode})...")
        text_nodes = _get_text_pipeline(
            llm,
            enrich_metadata=use_llm_enrichment,
        ).run(documents=text_docs)

    # --- Tabelas ---
    table_nodes = []
    if table_docs:
        if use_llm_enrichment:
            print(f"  Enriquecendo metadados e aplicando chunking nas tabelas ({interp_model()})...")
        else:
            print("  Inferindo metadados e aplicando chunking nas tabelas (modo local)...")

        enriched: dict[int, dict] = {}
        if llm is None:
            enriched = {
                i: _enrich_table_metadata(doc, None)
                for i, doc in enumerate(table_docs)
            }
        else:
            # Paraleliza as chamadas LLM de enriquecimento (I/O-bound).
            max_workers = min(llm_concurrency(default=8), len(table_docs))
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
            print(f"    {source} p.{page} -> {len(nodes)} chunk(s) [{strategy}] | {descricao[:60]}")

    all_nodes = text_nodes + table_nodes
    print(
        f"Normalização concluída. "
        f"{len(text_nodes)} nós de texto + {len(table_nodes)} nós de tabela = {len(all_nodes)} total."
    )
    return all_nodes
