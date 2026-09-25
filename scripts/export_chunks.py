"""Passo 1 de qualquer migração de banco vetorial: extrair os trechos.

Le chroma.sqlite3 em modo SOMENTE LEITURA (nunca abre o ChromaDB, nunca
escreve no banco) e escreve um JSONL portátil com texto + metadados de cada
trecho. E esse arquivo que alimenta a re-vetorizacao, seja qual for o provedor.

Uso:
    python export_chunks.py <chroma.sqlite3> <saida.jsonl>
"""
import json
import sqlite3
import sys
from collections import Counter

ESPERADO = 57112  # contagem medida do indice; o script avisa se divergir


def main() -> int:
    db_path, out_path = sys.argv[1], sys.argv[2]
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = con.cursor()

    ids = [r[0] for r in cur.execute("SELECT id FROM embeddings ORDER BY id")]
    print(f"trechos no banco : {len(ids)}")
    if len(ids) != ESPERADO:
        print(f"AVISO: esperava {ESPERADO} trechos (contagem medida antes)")

    campos = Counter()
    registros = {i: {"id": i} for i in ids}

    cur.execute(
        "SELECT id, key, string_value, int_value, float_value, bool_value "
        "FROM embedding_metadata"
    )
    for eid, key, sval, ival, fval, bval in cur:
        reg = registros.get(eid)
        if reg is None:
            continue
        if sval is not None:
            reg[key] = sval
        elif ival is not None:
            reg[key] = ival
        elif fval is not None:
            reg[key] = fval
        elif bval is not None:
            reg[key] = bool(bval)
        campos[key] += 1
    con.close()

    escritos = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for i in ids:
            reg = registros[i]
            texto = reg.pop(
                "chroma:document", reg.get("_node_content", "")
            )
            fh.write(json.dumps({"text": texto, "metadata": reg}, ensure_ascii=False))
            fh.write("\n")
            escritos += 1

    print(f"registros escritos: {escritos}")
    print(f"saida             : {out_path}")
    print("\n=== campos por trecho (contagem de presenca) ===")
    for k, n in campos.most_common(20):
        print(f"  {n:>6}  {k}")

    print("\n=== amostra ===")
    with open(out_path, encoding="utf-8") as fh:
        amostra = json.loads(fh.readline())
    meta = amostra["metadata"]
    print(f"  text: {amostra['text'][:70]!r}...")
    for k in ("source_file", "page", "type", "chunk_strategy"):
        if k in meta:
            print(f"  {k}: {meta[k]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
