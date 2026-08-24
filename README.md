# RAG CCDEP — análise econômica de São Paulo

Sistema RAG distribuído para consultar boletins e bases tabulares. Quatro engines
especializadas compartilham ingestão, indexação e contratos em `rag_core/`; o
Meta RAG analisa a pergunta, escolhe uma estratégia e aplica failover quando o
backend preferido não está saudável.

```text
React SPA :8501
      │
      ▼
Orquestrador :8010 ── health + circuit breaker + failover
      │
      ├── Principal :8000  (Vector + BM25, tabelas, séries, grafo opcional)
      ├── Agentic  :8001  (function calling limitado)
      ├── RAPTOR   :8002  (índice hierárquico)
      └── Self-RAG :8003  (relevância e verificação de suporte)
                          │
                          ▼
                 ChromaDB + cache BM25 local
```

## Início rápido

1. Crie `.env` com a chave do provedor:

```dotenv
RAG_LLM_PROVIDER=maritaca
MARITACA_API_KEY=...
RAG_POPUP_MODEL=sabiazinho-4

# Recomendado em ambientes compartilhados
RAG_API_KEY=chave-da-interface
RAG_BACKEND_API_KEY=chave-interna-entre-servicos
```

As explicações curtas dos popups numéricos usam `sabiazinho-4` por padrão,
em uma chamada estruturada e única por resposta. `RAG_POPUP_EXPLANATIONS=0`
desativa o recurso; `RAG_POPUP_TIMEOUT` controla seu timeout independente
(20 segundos por padrão). Saída ausente, inválida ou com números novos mantém
automaticamente a explicação determinística da interface.

Para executar também o LLM localmente com Ollama, use:

```dotenv
RAG_LLM_PROVIDER=ollama
RAG_LLM_MODEL=qwen3:4b-instruct
RAG_INTERP_MODEL=qwen3:4b-instruct
```

O endpoint padrão é `http://127.0.0.1:11434/v1` e não exige chave.
Para preservar memória, chamadas ao Ollama são serializadas por padrão;
`RAG_LLM_CONCURRENCY` permite ajustar esse limite explicitamente.
Na indexação, o Ollama usa metadados determinísticos por padrão para evitar
milhares de chamadas locais e timeouts. Para optar pelo enriquecimento por LLM,
defina `RAG_INGEST_LLM_ENRICHMENT=1` (a primeira indexação será bem mais lenta).
O reranking das consultas também preserva diretamente o score híbrido Vector+BM25
por padrão. `RAG_LLM_RERANK=1` reativa o reranker por LLM, caso o Ollama esteja
configurado com janela de contexto e timeout suficientes.

2. Coloque PDFs, CSVs, XLSX/XLS ou TXT em `data/`. Após a primeira carga,
   Principal, Agentic e Self-RAG processam somente arquivos novos, modificados
   ou removidos. RAPTOR e o grafo opcional da Principal reconstroem suas
   estruturas globais quando o corpus muda.
3. Suba o ambiente:

```bash
docker compose up --build
```

O frontend React fica disponível em `http://127.0.0.1:8501`. Para desenvolver
somente a interface com hot reload:

```bash
cd frontend
npm install
npm run dev
```

Por padrão, a interface usa o Meta RAG em `:8010` e permite alternar para as
engines diretas nas portas `8000`–`8003`. Endpoints e API key podem ser ajustados
na própria tela de configurações.

## Índice portátil via GitHub Releases

O ChromaDB Principal pode ser publicado como artefato versionado sem entrar no
histórico Git. O pacote inclui banco vetorial, segmentos HNSW, cache BM25,
manifesto, checksums por arquivo, versões das dependências e contagem de vetores.
Pare todos os serviços antes de exportar:

```powershell
docker compose down
python scripts/index_artifact.py export `
  --output index_artifacts/rag-principal-index-v3.tar.gz `
  --confirm-stopped
```

Valide e publique usando [GitHub CLI](https://cli.github.com/):

```powershell
python scripts/index_artifact.py verify `
  --archive index_artifacts/rag-principal-index-v3.tar.gz

python scripts/index_artifact.py publish `
  --archive index_artifacts/rag-principal-index-v3.tar.gz `
  --repo Thiagovilela2001/chat_ccdep_1.1 `
  --tag vector-index-v3
```

Em outra máquina, basta clonar e iniciar a engine Principal. No primeiro start,
se `rag_principal/chroma_db` estiver ausente ou vazio, o sistema baixa
automaticamente a Release `vector-index-v3`, confere o SHA-256, valida as versões
e a contagem de 35.438 vetores e instala o banco. Nos starts seguintes, o banco
local válido é reutilizado sem download e sem reindexação.

O comando manual equivalente é:

```powershell
python scripts/index_artifact.py download `
  --repo Thiagovilela2001/chat_ccdep_1.1 `
  --tag vector-index-v3 `
  --asset rag-principal-index-v3.tar.gz
```

O instalador valida SHA-256, versões e contagem vetorial antes da troca. Banco
anterior é movido para `chroma_backups_<data>/`; nenhuma exclusão definitiva é
feita. Banco parcial ou corrompido nunca é sobrescrito automaticamente: o start
para com erro claro. O bootstrap usa HTTPS e não exige GitHub CLI. Para
repositório privado, defina `GITHUB_TOKEN`.

Download e modo somente leitura ficam ativos por padrão na engine Principal.
Para desativar a automação e permitir a sincronização local do corpus:

```dotenv
RAG_INDEX_AUTO_DOWNLOAD=0
RAG_INDEX_READ_ONLY=0
```

O banco contém textos e metadados dos documentos, não apenas embeddings. Use
repositório privado quando o corpus não puder ser redistribuído. Para atualizar
o índice, desative o modo somente leitura, faça sincronização local, exporte novo
artefato e publique nova tag.

Interfaces locais: React em `http://127.0.0.1:8501`, API principal em
`:8000` e orquestrador em `:8010`. As portas ficam vinculadas ao loopback por
padrão. Para execução sem Docker, use `python main.py` dentro de cada engine.

`POST /query` aceita `conversation_id` e até 12 mensagens em `history`, além de
`question`. O frontend cria um identificador por conversa, envia o histórico
recente e o troca ao iniciar novo chat. Sem `history`, a API recupera os turnos
anteriores do cache local pelo mesmo `conversation_id`. Respostas informam
`conversation_id` e `memory_turns`.

O turno fica na memória para que a resposta curta do usuário complete a consulta.

## Componentes compartilhados

| Módulo | Responsabilidade |
|---|---|
| `rag_core/ingestion.py`, `processing.py`, `indexing.py` | ingestão, normalização, ChromaDB e BM25 |
| `rag_core/index_manifest.py` | detecção recursiva de mudanças e exclusões |
| `scripts/index_artifact.py` | exportação/instalação verificável via GitHub Releases |
| `rag_core/*_retriever.py` | recuperação narrativa, tabular e temporal |
| `rag_core/domain_skills.py` | descoberta e roteamento das skills econômicas locais |
| `rag_core/api_models.py`, `query_service.py` | contrato e fluxo HTTP comum das engines |
| `rag_core/conversation_memory.py` | cache TTL/LRU da memória conversacional por sessão |
| `rag_core/api_security.py` | autenticação, CORS, CSP, headers e rate limit |
| `rag_core/runtime.py` | deadline, limites e orçamento de contexto |
| `rag_core/provenance.py` | arquivo, página e score normalizados |
| `rag_orchestrator/` | classificação, roteamento, health, circuit breaker e failover |

As respostas tabulares do LLM usam JSON validado; nenhum código gerado pelo
modelo é executado no fluxo de produção. `safe_exec.py` permanece apenas como
helper legado restrito e não constitui uma fronteira de isolamento.

## Configuração operacional

| Variável | Padrão | Finalidade |
|---|---:|---|
| `RAG_REQUEST_TIMEOUT` | `180` s | deadline total por consulta |
| `RAG_MAX_CONTEXT_TOKENS` | `12000` | orçamento aproximado do contexto |
| `RAG_MEMORY_TTL_SECONDS` | `3600` s | expiração de conversa inativa no cache |
| `RAG_MEMORY_MAX_CONVERSATIONS` | `1000` | sessões mantidas no cache LRU por processo |
| `RAG_MEMORY_MAX_TURNS` | `6` | turnos recentes usados para resolver perguntas subsequentes |
| `RAG_MEMORY_MAX_CONTEXT_CHARS` | `12000` | limite de caracteres do histórico injetado no RAG |
| `RAG_AGENTIC_MAX_ITERATIONS` | `8` | iterações Agentic, limitado a 12 |
| `RAG_AGENTIC_MAX_TOOL_CALLS` | `12` | chamadas de ferramenta, limitado a 32 |
| `RAG_AGENTIC_CRITIC_ROUNDS` | `1` | revisões críticas, limitado a 3 |
| `RAG_SELFRAG_MAX_RETRIES` | `1` | retries Self-RAG, limitado a 2 |
| `RAG_CIRCUIT_FAILURE_THRESHOLD` | `3` | falhas antes de abrir o circuito |
| `RAG_CIRCUIT_RECOVERY_SECONDS` | `30` s | espera antes de nova tentativa |
| `RAG_CORS_ORIGINS` | interfaces locais | allowlist CORS separada por vírgula |
| `RAG_RATE_LIMIT` / `RAG_RATE_WINDOW` | `30` / `60` s | limite em memória por IP |
| `RAG_USE_GRAPH` | `0` | habilita grafo na engine principal |
| `RAG_DATA_DIR` | `<engine>/data` ou `../data` | seleciona o corpus documental |
| `RAG_DB_DIR` | `<engine>/chroma_db` | usa um índice ChromaDB separado |
| `RAG_INDEX_AUTO_DOWNLOAD` | `1` na Principal | baixa Release somente quando banco está ausente ou vazio |
| `RAG_INDEX_REPO` | `Thiagovilela2001/chat_ccdep_1.1` | repositório da Release do índice |
| `RAG_INDEX_TAG` | `vector-index-v3` | tag imutável da Release |
| `RAG_INDEX_ASSET` | `rag-principal-index-v3.tar.gz` | asset do banco pré-indexado |
| `RAG_INDEX_DOWNLOAD_TIMEOUT` | `600` | timeout HTTPS em segundos |
| `RAG_INDEX_READ_ONLY` | automático com bootstrap | usa somente leitura sem diferenças; habilita sincronização quando o corpus local contém fontes novas |
| `RAG_RETRIEVAL_TOP_K` | `80` | candidatos por pool textual ou tabular |
| `RAG_QUERY_FUSION_QUERIES` | `2` | consulta original mais expansões semânticas |
| `RAG_RERANK_CANDIDATE_LIMIT` | `40` | candidatos enviados ao reranker |
| `RAG_RERANK_TOP_N` | `24` | resultados preservados pelo reranker |
| `RAG_TEXT_TOP_N` | `20` | trechos narrativos enviados à síntese |
| `RAG_STRUCTURED_TOP_N` | `10` | trechos tabulares/temporais por retriever |
| `RAG_MAX_CHUNKS_PER_DOCUMENT` | `3` | prioridade inicial por documento antes do preenchimento |
| `RAG_MAX_DOMAIN_SKILLS` | `2` | skills especializadas combinadas por consulta |
| `RAG_INPUT_COST_PER_MILLION_USD` | `0` | preço para estimativa agregada de custo |
| `RAG_OUTPUT_COST_PER_MILLION_USD` | `0` | preço para estimativa agregada de custo |

Os containers executam como usuário não-root. Em hosts Linux, ajuste
`APP_UID`/`APP_GID` no build se os diretórios persistentes tiverem outro dono.

## Verificação

```bash
python -m pytest -p no:cacheprovider rag_principal/tests rag_orchestrator/tests -q
python rag_orchestrator/tests/test_router.py
python rag_orchestrator/tests/test_pipeline.py
cd frontend && npm ci && npm run lint && npm test && npm run build
docker compose config --quiet
python benchmarks/benchmark_core.py --iterations 10000
```

A avaliação reproduzível usa somente o `rag_principal` e o judge Maritaca
`sabia-4` por padrão:

```bash
python evaluate.py --split dev --seed 42
python evaluate.py --split test --seed 42
python evaluate.py --split adversarial --seed 42
python evaluate.py --split all --seed 42
```

O judge usa `MARITACA_API_KEY`; `RAGAS_JUDGE_API_KEY`,
`RAGAS_JUDGE_BASE_URL` e `RAGAS_JUDGE_MODEL` permitem overrides explícitos.
Cada execução registra hash dos splits, commit, versões, precisão numérica e
latências p50/p95. Cada API também expõe `/metrics` em formato Prometheus, sem
incluir perguntas ou conteúdo documental.

Quando `RAG_DATA_DIR` aponta para um corpus local mais recente que o índice da
Release, a Principal sincroniza somente as fontes adicionadas, modificadas ou
removidas. Defina `RAG_INDEX_READ_ONLY=1` explicitamente apenas quando quiser
ignorar essas diferenças e manter o artefato portátil imutável.
A análise técnica detalhada está em [ANALISE_PROJETO.md](ANALISE_PROJETO.md).
