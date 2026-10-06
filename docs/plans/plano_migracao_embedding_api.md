# Plano de migração — embedding por API + banco vetorial gerenciado

> **Estado: investigação concluída, nada implementado.**
> Nenhum arquivo de código do projeto foi alterado. Nada foi commitado. Nada foi
> publicado. Este documento é uma proposta para decisão, com os números medidos.

## 1. Por que esta migração existe

Objetivo: rodar o RAG numa máquina pequena (ou sem servidor dedicado). O gargalo
não é o dado — é o **modelo de embedding carregado em cada processo**.

Medido numa engine real (`rag_principal`, esta máquina, `.venv` do projeto):

| Etapa | RSS |
|---|---|
| Python vazio | 14,4 MB |
| + bibliotecas (torch, chromadb, pandas, llama-index, rank_bm25) | 314,0 MB |
| + cache BM25 (57.112 nós) | 586,1 MB |
| + índice e retriever ligados | **644,1 MB** |
| **com `bge-m3` (medido na engine rodando)** | **1.490,0 MB** |

`bge-m3` custa **~846 MB por processo**, e cada engine abre o seu. Tirá-lo do
processo é a economia que vale a pena; o resto (644 MB) é o piso.

### Decomposição do piso (medida, importando em ordem)

| Passo | RSS | Custo do passo |
|---|---|---|
| Python vazio | 16,9 MB | — |
| `import torch` | 17,0 MB | **+0,1 MB** |
| `import chromadb` | 69,3 MB | +52,3 |
| `import pandas` | 110,9 MB | +41,6 |
| llama-index core + vector store Chroma | 166,7 MB | +55,7 |
| `import rank_bm25` | 166,7 MB | +0,0 |
| carregar os 57.112 nós BM25 | 439,7 MB | **+273,0** |
| abrir índice + retriever | **631,5 MB** | **+191,8** |

Duas leituras importam:

- **`torch` custa 0,1 MB para importar** — ele carrega de forma preguiçosa. Tirar o
  caminho local de embedding **não rende RAM pelas bibliotecas**, só pelo modelo.
  (Eu esperava o contrário; a medição corrigiu.)
- O piso não é dominado por bibliotecas: é dominado pelos **nós BM25 em memória**
  (273 MB) e pela **abertura do índice/retriever** (192 MB).

### O que o Supabase muda, e o que ele não muda

Mover o banco para o Supabase **não reduz o piso acima por si só**. O `bge-m3` só sai
se o **embedding** for para uma API — e isso obriga reindexar os 57.112 trechos, porque
vetores de modelos diferentes não são comparáveis. Duas consequências práticas:

1. **O armazenamento pode ficar no Chroma local.** 746 MB de disco é barato; o que
   custa é RAM. Mantendo o Chroma, a garantia de integridade do artefato/release
   (`scripts/index_artifact.py`) continua valendo — e some um risco da migração.
2. **O ganho do Supabase não está medido.** Com o banco remoto a estrutura HNSW local
   deixa de ser gravada, mas o retriever do LlamaIndex e os nós continuam no processo.
   Quanto dos 192 MB de abertura do índice voltaria, eu não sei — precisa de um projeto
   real para medir. Não prometa essa economia.

Conclusão: são **duas migrações independentes**, e a que resolve o gargalo é a do
embedding. O Supabase é a segunda, opcional, e a de ganho não medido.

### O que a migração NÃO resolve

O piso de 644 MB continua, e dele **272 MB são só os nós BM25 em memória** — a
busca lexical é local e não migra para o Supabase. Somado às bibliotecas, um
processo sem modelo de embedding fica em ~650 MB. Ou seja: **cabe com folga em
2 GB, cabe apertado em 1 GB** — mas não é "serverless".

## 2. O que foi medido (reproduzível)

```bash
# 1. consumo com o modelo (engine nativa, health 200, uma consulta)
tasklist /FI "PID eq <pid>"          # 1.559.456 K em repouso

# 2. consumo sem o modelo (mesmas libs + BM25, MockEmbedding no lugar do bge-m3)
.venv/Scripts/python.exe scripts/measure_engine_rss.py    # 644,1 MB
```

Scripts deixados no repositório (aditivos, nada existente foi alterado):

| Script | Para que serve |
|---|---|
| `scripts/inspect_chroma_sqlite.py` | inventário do banco em modo somente-leitura, sem abrir o ChromaDB |
| `scripts/export_chunks.py` | extrai os trechos para JSONL — **passo 1 da migração** |
| `scripts/analyze_chunks_export.py` | composição por tipo, corpus coberto, tokens a re-vetorizar |
| `scripts/measure_engine_rss.py` | mede o consumo do pipeline com e sem o modelo local |

Inventário do banco atual, lido em **modo somente-leitura** direto do sqlite
(`file:chroma.sqlite3?mode=ro` — sem abrir o ChromaDB):

| Medida | Valor |
|---|---|
| Vetores na coleção `estatisticas` | **57.112** |
| Nós BM25 (`bm25_nodes.pkl`) | 57.112 (batem) |
| Composição do índice | **44.401 trechos de tabela** / **12.711 de texto** |
| Estratégias de chunk | 42.878 `row_per_chunk`, 1.523 `full_table`, 12.711 texto |
| Arquivos-fonte cobertos | **513** |
| Texto armazenado (`chroma:document`) | 17.373.379 caracteres (~**4,34 M tokens**) |
| Texto — trechos de prosa | 10.834.676 chars, média 852 |
| Texto — trechos de tabela | 6.538.703 chars, média 147 |
| Disco: `chroma.sqlite3` | 428 MB |
| Disco: índice HNSW (segmentos) | 271 MB + 8,5 MB + 4,6 MB |
| Disco: cache BM25 | 34 MB |
| **Total do banco (Principal)** | **746 MB** |

Exportação dos trechos (passo 1 de qualquer migração), já executada:

```bash
.venv/Scripts/python.exe export_chunks.py rag_principal/chroma_db/chroma.sqlite3 saida.jsonl
# 57.112 trechos em 4,8 s → 108 MB (50 MB sem o campo _node_content)
```

**Achado que destrava a migração:** o texto dos 57.112 trechos, o `source_file`,
o `page`, o `type` e o nó serializado (`_node_content`) **estão dentro do banco**.
Não é preciso ter os 513 PDFs (0,86 GB) para re-vetorizar — o corpus sai do
próprio sqlite. Isso importa porque os PDFs não estão nesta máquina.

Dependências novas resolvem no ambiente pinado, sem mexer no core:

```
pip install --dry-run llama-index-vector-stores-postgres llama-index-embeddings-openai
Would install asyncpg-0.31.0 llama-index-vector-stores-postgres-0.9.0 pgvector-0.5.0 psycopg2-binary-2.9.13
```

`llama-index-core` 0.14.23 já instalado permanece intocado.

## 3. Superfície de mudança

A migração é **concentrada**: mexe na camada de armazenamento e de embedding, não
no resto do RAG.

| Arquivo:linha | O que muda |
|---|---|
| `rag_core/indexing.py:6,71-74` | `HuggingFaceEmbedding("BAAI/bge-m3")` → provedor por API, escolhido por variável de ambiente |
| `rag_core/indexing.py:83-86,117` | `chromadb.PersistentClient` + `ChromaVectorStore` → `PostgresVectorStore` |
| `rag_core/index_sync.py` | sincronização incremental usa `chromadb` e `collection.count()` → SQL |
| `scripts/index_artifact.py:30` | constante `EMBEDDING_MODEL = "BAAI/bge-m3"` |
| `scripts/index_artifact.py:140,243` | manifesto e `_verify_versions` (a garantia de integridade do artefato) |
| `rag_orchestrator/src/registry.py:55` | `embed_model: str = "BAAI/bge-m3"` declarado por engine |
| `requirements.txt` | +3 pinos (`vector-stores-postgres`, `psycopg2-binary`, driver) |
| `.env` / `docker-compose.yml` | variáveis novas (provedor, chave, URL do banco) |

**Não muda:** `text_retriever.py`, `tables_retriever.py`, `timeseries_retriever.py`
(os de tabela/série leem `documents/tables.parquet`, não o banco vetorial), os
validadores, a camada de LLM, a API e o frontend.

Já compatíveis, sem alteração: `graph_indexing.py:145`, `graph_retriever.py:50`
(usam `Settings.embed_model`) e `raptor_indexing.py:49,134` (chamam
`embed_model.get_text_embedding`).

## 4. Esquema alvo (Supabase / pgvector)

Fixar **1024 dimensões** é a escolha que mantém tudo do mesmo tamanho de hoje:

| Modelo | Dimensões | Observação |
|---|---|---|
| `bge-m3` (atual) | 1024 | baseline de qualidade a bater |
| Cohere `embed-multilingual-v3.0` | **1024** | multilíngue, mesma largura do atual (doc Cohere) |
| Cohere `embed-v4.0` | 256/512/1024/1536 | selecionável |
| OpenAI `text-embedding-3-large` | até 3072 | parâmetro `dimensions` encurta (doc OpenAI) — usar 1024 |

⚠️ **Teto do pgvector:** índice HNSW aceita no máximo **2000 dimensões** para o
tipo `vector` (4000 para `halfvec`) — confirmado na doc da Supabase. Em 1024 não
há problema; **3072 exigiria `halfvec`** e a consulta precisaria repetir o cast
idêntico, senão o planner cai em varredura completa em silêncio.

⚠️ **Limite do plano gratuito:** a Supabase entra em **modo somente-leitura acima
de 500 MB de banco** (1 GB de disco, mas o corte é o tamanho do banco) e **pausa
o projeto após 1 semana sem uso**.

A conta de armazenamento, em 1024 dimensões:

| Componente | Precisão cheia | `halfvec` |
|---|---|---|
| Vetores na tabela (57.112 × 1024) | ~234 MB | ~117 MB |
| Índice HNSW (est.) | ~240 MB | ~125 MB |
| Texto + metadados + JSONB do nó | ~60–100 MB | ~60–100 MB |
| **Total estimado** | **~530–570 MB** ❌ | **~300–340 MB** ✅ |

Os dois primeiros são aritmética (4 bytes por dimensão; o pgvector guarda uma
cópia do vetor no índice HNSW) — o total com texto é **estimativa**, não medida:
não construí nenhum projeto Supabase. A conclusão prática: **em precisão cheia
isso provavelmente estoura o plano gratuito**; com `halfvec` cabe com folga.
Decisão sua: `halfvec` (grátis) ou Supabase Pro ($25/mês).

## 5. Riscos

1. **Qualidade em português — o risco decisivo.** `bge-m3` é um modelo
   multilíngue forte, treinado com português. Trocar por um modelo de API pode
   **piorar o recall** e nenhum número garante o contrário. Não é medível sem
   chave de API. Por isso existe o teste de aceitação da seção 7.
2. **O índice sai do controle de versão.** Hoje o artefato é publicado como
   release com SHA e validação de contagem de vetores
   (`scripts/index_artifact.py:243`). No Supabase, essa garantia precisa ser
   reproduzida (tabela de manifesto com hash e contagem), senão você perde a
   capacidade de provar que o índice é o esperado.
3. **Custo recorrente novo:** uma chamada de embedding por consulta, além do LLM.
   Somado ao link aberto, pede teto de gasto no provedor.
4. **Metade da recuperação não migra:** BM25 (272 MB, local) e os retrievers de
   tabela/série continuam no processo. A fusão dos resultados precisa ser testada
   de novo, não presumida.
5. **Dependência de rede por consulta** — pequena frente aos 38–84 s medidos por
   consulta, mas é um terceiro de quem você passa a depender.
6. **Contraria a regra 2 do `AGENTS.md`** ("reutilize o que já existe... antes de
   introduzir novas dependências"): a migração adiciona 3 pinos. Decisão
   consciente, não acidente.

## 6. Rollback

Nada é destruído. O banco atual (746 MB locais + release de 400 MB) e o JSONL
exportado (108 MB) continuam válidos. Enquanto o caminho local for o padrão e a
troca for por variável de ambiente, voltar é trocar a variável.

## 7. Teste de aceitação (o que decide se valeu)

Linhas de base **já medidas** com `bge-m3`, para comparar depois:

- 30/30 arquivos ground-truth presentes no índice;
- nas 7 perguntas dev que completaram: fonte correta no top-k, **rank 1 em 5 de 7,
  pior rank 4**.

Depois da migração, rodar as 15 perguntas de `evaluation/golden_dataset_dev.json`
e comparar rank e presença no top-k. Critério de aceite: **nenhuma fonte sai do
top-k e o pior rank não piora**. Se piorar, a migração está reprovada — e a
resposta é voltar para `bge-m3` numa máquina com RAM, não insistir.

## 8. Passos

| # | Passo | Quem |
|---|---|---|
| 1 | Exportar os trechos para JSONL | ✅ feito (4,8 s, 57.112 registros) |
| 2 | Verificar que as dependências novas resolvem | ✅ feito (sem tocar no core) |
| 3 | Créditos + chave de embedding | **você** |
| 4 | Projeto Supabase + schema com `halfvec` | **você** cria; credencial fica no `.env` do servidor |
| 5 | Re-vetorizar os 57.112 trechos (4,34 M tokens, uma vez) | eu, quando houver chave |
| 6 | Carga no pgvector + manifesto de integridade | eu |
| 7 | Troca atrás de variável, caminho local como padrão | eu |
| 8 | Medir com o golden dataset (seção 7) | eu |
| 9 | Rodar a suíte completa (baseline: 197 passed, 1 skipped) | eu |
| 10 | Só então virar o padrão e publicar | juntos |

**Limite explícito deste agente:** eu não peço nem aceito senha, connection
string ou chave em conversa. Você configura o `.env` no servidor; eu nunca
preciso ver o valor.

## 9. O que eu NÃO verifiquei

- **Qualidade de recuperação em português** com qualquer modelo de API (exige chave).
- **Latência real** de embedding por consulta.
- **Tamanho real do banco no Supabase** — a tabela da seção 4 é aritmética para
  os vetores e **estimativa** para o resto; nenhum projeto foi criado.
- O comportamento do `halfvec` com a consulta do LlamaIndex (o cast precisa ser
  idêntico nos dois lados; não testei a integração).
- Se o servidor Oracle é ARM (`aarch64`) — a pergunta segue sem resposta, e ela
  decide se o build precisa ser feito no próprio servidor.
