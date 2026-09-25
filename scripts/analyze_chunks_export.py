"""Analisa o JSONL exportado por export_chunks.py.

Reporta composicao por tipo, corpus coberto, volume de texto a re-vetorizar e o
tamanho de uma versao enxuta (sem o campo _node_content, que duplica o texto).

Uso:
    python analyze_chunks_export.py chunks.jsonl
"""
import json
import sys
from collections import Counter, defaultdict

if len(sys.argv) < 2:
    raise SystemExit("uso: python analyze_chunks_export.py <chunks.jsonl>")
caminho = sys.argv[1]

tipos = Counter()
arquivos = set()
estrategias = Counter()
chars_por_tipo = defaultdict(int)
n_por_tipo = defaultdict(int)
chars_slim = 0
n = 0

with open(caminho, encoding="utf-8") as fh:
    for linha in fh:
        reg = json.loads(linha)
        meta = reg["metadata"]
        tipo = meta.get("type")
        n += 1
        n_por_tipo[tipo] += 1
        chars_por_tipo[tipo] += len(reg["text"])
        tipos[tipo] += 1
        arquivos.add(meta.get("source_file"))
        estrategias[meta.get("chunk_strategy")] += 1
        slim = {
            "text": reg["text"],
            "metadata": {k: v for k, v in meta.items() if k != "_node_content"},
        }
        chars_slim += len(json.dumps(slim, ensure_ascii=False))

texto_total = sum(chars_por_tipo.values())

print(f"registros        : {n}")
print(f"arquivos-fonte   : {len(arquivos)}")
print(f"\n=== por type ===")
for tipo, qtd in tipos.most_common():
    media = chars_por_tipo[tipo] / qtd if qtd else 0
    print(f"  {qtd:>6}  {str(tipo):<6} {chars_por_tipo[tipo]:>12,} chars  media {media:>6.0f}")
print(f"\n=== por chunk_strategy ===")
for k, v in estrategias.most_common():
    print(f"  {v:>6}  {k}")
print(f"\ncaracteres de texto (a re-vetorizar) : {texto_total:,}")
print(f"  ~tokens (4 char/token)             : {texto_total/4:,.0f}")
print(f"  JSONL enxuto (sem _node_content)   : {chars_slim/1e6:.1f} MB")
