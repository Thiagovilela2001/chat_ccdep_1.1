"""Inspeciona chroma.sqlite3 em modo SOMENTE LEITURA para dimensionar uma
migração de banco vetorial: quantos trechos, quanto texto, quais metadados.
Não abre o ChromaDB e não escreve nada.
"""
import sqlite3
import sys
from collections import Counter

path = sys.argv[1]
con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
cur = con.cursor()

print("=== tabelas ===")
tabs = [r[0] for r in cur.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
print(", ".join(tabs))

print("\n=== contagens ===")
for t in ("collections", "embeddings", "embedding_metadata", "segments"):
    if t in tabs:
        print(f"{t}: {cur.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]}")

print("\n=== colecoes ===")
for row in cur.execute("SELECT id, name FROM collections"):
    print("  ", row)

print("\n=== chaves de metadado (o que cada trecho carrega) ===")
keys = Counter(r[0] for r in cur.execute(
    "SELECT DISTINCT key FROM embedding_metadata"))
for k, v in keys.most_common(20):
    print(f"  {k}: {v}")

print("\n=== volume de texto armazenado (campo chroma:document) ===")
row = cur.execute(
    "SELECT COUNT(*), SUM(LENGTH(string_value)), AVG(LENGTH(string_value)), "
    "MAX(LENGTH(string_value)) FROM embedding_metadata "
    "WHERE key='chroma:document' AND string_value IS NOT NULL").fetchone()
n, total, avg, mx = row
print(f"  trechos com texto : {n}")
print(f"  caracteres totais : {total}")
if n:
    print(f"  media por trecho  : {avg:.0f} caracteres")
    print(f"  maior trecho      : {mx} caracteres")
    print(f"  ~tokens estimados : {total/4:.0f} (regra de 4 caracteres/token)")

print("\n=== exemplos de metadado de um trecho ===")
sample = cur.execute(
    "SELECT id FROM embedding_metadata LIMIT 1").fetchone()
if sample:
    for r in cur.execute(
        "SELECT key, substr(string_value,1,90) FROM embedding_metadata "
        "WHERE id=(SELECT id FROM embedding_metadata LIMIT 1)"):
        print(f"  {r[0]} = {r[1]!r}")
con.close()
