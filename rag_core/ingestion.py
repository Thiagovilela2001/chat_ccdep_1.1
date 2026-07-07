import os
import json
import uuid
from pathlib import Path
from datetime import datetime

# Adiciona o Ghostscript ao PATH para o Camelot funcionar no modo lattice
_GS_BIN = r"C:\Program Files\gs\gs10.06.0\bin"
if _GS_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _GS_BIN + os.pathsep + os.environ.get("PATH", "")

import fitz  # pymupdf
import camelot
import pandas as pd
from llama_index.core import Document

SUPPORTED_EXTS = {'.pdf', '.csv', '.xlsx', '.xls', '.txt'}


def extract_pdf_text(pdf_path: Path) -> list:
    """Extrai texto por página via PyMuPDF."""
    chunks = []
    doc = fitz.open(str(pdf_path))
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "source_file": pdf_path.name,
                "page": page_num,
                "text": text,
                "type": "text",
            })
    doc.close()
    return chunks


def extract_pdf_tables(pdf_path: Path) -> list:
    """Extrai tabelas de PDF via Camelot (lattice → stream como fallback)."""
    tables = []
    try:
        result = camelot.read_pdf(str(pdf_path), pages="all", flavor="lattice")
        if result.n == 0:
            result = camelot.read_pdf(str(pdf_path), pages="all", flavor="stream")

        for i, table in enumerate(result):
            df = table.df
            if df.empty:
                continue
            tables.append({
                "table_id": str(uuid.uuid4()),
                "source_file": pdf_path.name,
                "page": table.page,
                "table_index": i,
                "markdown": df.to_markdown(index=False),
                "rows": df.shape[0],
                "cols": df.shape[1],
                "type": "table",
            })
    except Exception as e:
        print(f"  Aviso: falha ao extrair tabelas de {pdf_path.name}: {e}")
    return tables


def extract_spreadsheet(file_path: Path) -> list:
    """Extrai tabelas de CSV/XLSX/XLS via pandas. Cada aba vira uma tabela."""
    tables = []
    try:
        if file_path.suffix.lower() == ".csv":
            df = None
            for sep in [",", ";", "\t"]:
                try:
                    candidate = pd.read_csv(file_path, sep=sep)
                    if candidate.shape[1] > 1:
                        df = candidate
                        break
                except Exception:
                    continue
            if df is None:
                df = pd.read_csv(file_path)
            sheets = {"Planilha": df}
        else:
            xl = pd.ExcelFile(file_path)
            sheets = {name: xl.parse(name) for name in xl.sheet_names}

        for sheet_name, df in sheets.items():
            df = df.dropna(how="all").dropna(axis=1, how="all")
            if df.empty:
                continue
            tables.append({
                "table_id": str(uuid.uuid4()),
                "source_file": file_path.name,
                "page": sheet_name,
                "table_index": 0,
                "markdown": df.to_markdown(index=False),
                "rows": df.shape[0],
                "cols": df.shape[1],
                "type": "table",
            })
    except Exception as e:
        print(f"  Aviso: erro ao ler {file_path.name}: {e}")
    return tables


def save_intermediate(documents_dir: Path, text_chunks: list, tables: list, file_metadata: list):
    """Persiste o formato intermediário em documents/."""
    documents_dir.mkdir(exist_ok=True)

    with open(documents_dir / "text_chunks.json", "w", encoding="utf-8") as f:
        json.dump(text_chunks, f, ensure_ascii=False, indent=2)

    if tables:
        df_tables = pd.DataFrame(tables)
        df_tables["page"] = df_tables["page"].astype(str)
        df_tables.to_parquet(documents_dir / "tables.parquet", index=False)

    with open(documents_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {"files": file_metadata, "generated_at": datetime.now().isoformat()},
            f,
            ensure_ascii=False,
            indent=2,
        )


def to_llama_documents(text_chunks: list, tables: list) -> list:
    """Converte chunks e tabelas para LlamaIndex Documents."""
    docs = []
    for chunk in text_chunks:
        docs.append(Document(
            text=chunk["text"],
            metadata={
                "source_file": chunk["source_file"],
                "page": chunk["page"],
                "type": "text",
                "chunk_id": chunk["chunk_id"],
            },
        ))
    for table in tables:
        docs.append(Document(
            text=f"Tabela extraída de {table['source_file']} (página/aba: {table['page']}):\n\n{table['markdown']}",
            metadata={
                "source_file": table["source_file"],
                "page": str(table["page"]),
                "type": "table",
                "table_id": table["table_id"],
                "rows": table["rows"],
                "cols": table["cols"],
            },
        ))
    return docs


def load_documents(data_dir: str) -> list:
    """
    Pipeline de ingestão:
      1. Extrai texto (PyMuPDF) e tabelas (Camelot) de PDFs
      2. Extrai tabelas de planilhas (pandas)
      3. Salva formato intermediário em documents/
      4. Retorna LlamaIndex Documents prontos para indexação
    """
    path = Path(data_dir)
    if not path.exists():
        os.makedirs(data_dir, exist_ok=True)
        print(f"Aviso: Diretório '{data_dir}' criado. Adicione documentos e rode novamente.")
        return []

    arquivos = [f for f in path.rglob("*") if f.suffix.lower() in SUPPORTED_EXTS]
    if not arquivos:
        print("Aviso: Nenhum arquivo suportado encontrado em 'data/'.")
        return []

    print(f"Encontrados {len(arquivos)} arquivo(s) em {data_dir}...")

    all_text_chunks, all_tables, file_metadata = [], [], []

    for file_path in arquivos:
        print(f"  Processando: {file_path.name}")
        text_chunks, tables = [], []

        if file_path.suffix.lower() == ".pdf":
            text_chunks = extract_pdf_text(file_path)
            tables = extract_pdf_tables(file_path)
        elif file_path.suffix.lower() in {".csv", ".xlsx", ".xls"}:
            tables = extract_spreadsheet(file_path)
        elif file_path.suffix.lower() == ".txt":
            text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                text_chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "source_file": file_path.name,
                    "page": 1,
                    "text": text,
                    "type": "text",
                })

        all_text_chunks.extend(text_chunks)
        all_tables.extend(tables)
        file_metadata.append({
            "file_name": file_path.name,
            "file_type": file_path.suffix.lower().lstrip("."),
            "text_chunks": len(text_chunks),
            "tables": len(tables),
            "processed_at": datetime.now().isoformat(),
        })
        print(f"    → {len(text_chunks)} chunk(s) de texto, {len(tables)} tabela(s)")

    documents_dir = path.parent / "documents"
    save_intermediate(documents_dir, all_text_chunks, all_tables, file_metadata)

    print(f"\nFormato intermediário salvo em: {documents_dir}")
    print(f"  text_chunks.json : {len(all_text_chunks)} chunks")
    print(f"  tables.parquet   : {len(all_tables)} tabelas")

    docs = to_llama_documents(all_text_chunks, all_tables)
    print(f"  Total de documentos para indexação: {len(docs)}")
    return docs


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    docs = load_documents(DATA_DIR)
    if docs:
        print("\nAmostra do primeiro documento:")
        print(docs[0].text[:300])
        print("Metadados:", docs[0].metadata)
