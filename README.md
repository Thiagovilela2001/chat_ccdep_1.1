# RAG CCDEP — análise econômica de São Paulo

Sistema RAG distribuído para consultar boletins e bases tabulares. Quatro engines
especializadas compartilham ingestão, indexação e contratos em `src/rag_ccdep/core/`; o
Meta RAG analisa a pergunta, escolhe uma estratégia e aplica failover quando o
backend preferido não está saudável.

```text
React SPA :5173 (desenvolvimento)
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

Código Python usa layout instalável `src/`: infraestrutura em
`src/rag_ccdep/core/`, engines em `src/rag_ccdep/engines/` e Meta RAG em
`src/rag_ccdep/orchestrator/`. Testes ficam em `tests/`; índices e demais
artefatos mutáveis ficam em `var/`.

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

Para execução local:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
python -m rag_ccdep.cli principal --port 8000
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

Para testar modelos open-weight grandes sem hospedar os pesos localmente, use o
endpoint OpenAI-compatible do OpenRouter:

```dotenv
RAG_LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=...
RAG_LLM_MODEL=openai/gpt-oss-120b
RAG_INTERP_MODEL=openai/gpt-oss-120b
RAG_POPUP_MODEL=openai/gpt-oss-120b
RAG_LLM_RERANK=0
```

Para comparar com o Nemotron, mantenha o provedor e troque os três modelos:

```dotenv
RAG_LLM_MODEL=nvidia/nemotron-3-super-120b-a12b
RAG_INTERP_MODEL=nvidia/nemotron-3-super-120b-a12b
RAG_POPUP_MODEL=nvidia/nemotron-3-super-120b-a12b
```

O endpoint hospedado da NVIDIA NIM também é suportado diretamente:

```dotenv
RAG_LLM_PROVIDER=nvidia
NVIDIA_API_KEY=...
RAG_LLM_MODEL=openai/gpt-oss-120b
RAG_INTERP_MODEL=openai/gpt-oss-120b
RAG_POPUP_MODEL=openai/gpt-oss-120b
```

Nos provedores reasoning, o padrão usa esforço `low`, oculta o raciocínio da
resposta e reserva 4.096 tokens de saída. Ajustes opcionais:
`RAG_REASONING_EFFORT`, `RAG_REASONING_EXCLUDE` e `RAG_LLM_MAX_TOKENS`.
Para preservar resultados separados durante a comparação:

```bash
python evaluate.py --split dev --limit 1 --output-label gpt-oss-120b
python evaluate.py --split dev --limit 1 --output-label nemotron-3-super
```

2. Coloque PDFs, CSVs, XLSX/XLS ou TXT em `data/`. Após a primeira carga,
   Principal, Agentic e Self-RAG processam somente arquivos novos, modificados
   ou removidos. RAPTOR e o grafo opcional da Principal reconstroem suas
   estruturas globais quando o corpus muda.
3. Em um terminal, suba a engine Principal:

```powershell
python -m rag_ccdep.cli principal --port 8000
```

4. Em outro terminal, suba o frontend:

```powershell
cd frontend
npm ci
npm run dev
```

O frontend fica disponível em `http://127.0.0.1:5173`.

Por padrão, a interface usa o Meta RAG em `:8010` e permite alternar para as
engines diretas nas portas `8000`–`8003`. Endpoints e API key podem ser ajustados
na própria tela de configurações.

## Índice portátil via GitHub Releases

O ChromaDB Principal pode ser publicado como artefato versionado sem entrar no
histórico Git. O pacote inclui banco vetorial, segmentos HNSW, cache BM25,
manifesto, checksums por arquivo, versões das dependências e contagem de vetores.
Pare todas as engines antes de exportar:

```powershell
python scripts/index_artifact.py export `
  --output index_artifacts/rag-principal-index-v1.tar.gz `
  --confirm-stopped
```

Valide e publique usando [GitHub CLI](https://cli.github.com/):

```powershell
python scripts/index_artifact.py verify `
  --archive index_artifacts/rag-principal-index-v1.tar.gz

python scripts/index_artifact.py publish `
  --archive index_artifacts/rag-principal-index-v1.tar.gz `
  --repo Thiagovilela2001/chat_ccdep_1.1 `
  --tag vector-index-v1
```

Em outra máquina, basta clonar e iniciar a engine Principal. No primeiro start,
se `var/principal/chroma_db` estiver ausente ou vazio, o sistema baixa
automaticamente a Release `vector-index-v1`, confere o SHA-256, valida as versões
e a contagem de 16.237 vetores e instala o banco. Nos starts seguintes, o banco
local válido é reutilizado sem download e sem reindexação.

O comando manual equivalente é:

```powershell
python scripts/index_artifact.py download `
  --repo Thiagovilela2001/chat_ccdep_1.1 `
  --tag vector-index-v1 `
  --asset rag-principal-index-v1.tar.gz
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

Interfaces locais: React dev em `http://127.0.0.1:5173`, API principal em
`:8000` e orquestrador em `:8010`. As portas ficam vinculadas ao loopback por
padrão. Use `python -m rag_ccdep.cli <componente>`; componentes:
`principal`, `agentic`, `raptor`, `selfrag` e `orchestrator`.

## Componentes compartilhados

| Módulo | Responsabilidade |
|---|---|
| `src/rag_ccdep/core/ingestion.py`, `processing.py`, `indexing.py` | ingestão, normalização, ChromaDB e BM25 |
| `src/rag_ccdep/core/index_manifest.py` | detecção recursiva de mudanças e exclusões |
| `scripts/index_artifact.py` | exportação/instalação verificável via GitHub Releases |
| `src/rag_ccdep/core/*_retriever.py` | recuperação narrativa, tabular e temporal |
| `src/rag_ccdep/core/domain_skills.py` | descoberta e roteamento das skills econômicas locais |
| `src/rag_ccdep/core/api_models.py`, `query_service.py` | contrato e fluxo HTTP comum das engines |
| `src/rag_ccdep/core/api_security.py` | autenticação, CORS, CSP, headers e rate limit |
| `src/rag_ccdep/core/runtime.py` | deadline, limites e orçamento de contexto |
| `src/rag_ccdep/core/provenance.py` | arquivo, página e score normalizados |
| `src/rag_ccdep/orchestrator/` | classificação, roteamento, health, circuit breaker e failover |

As respostas tabulares do LLM usam JSON validado; nenhum código gerado pelo
modelo é executado no fluxo de produção. `safe_exec.py` permanece apenas como
helper legado restrito e não constitui uma fronteira de isolamento.

## Configuração operacional

| Variável | Padrão | Finalidade |
|---|---:|---|
| `RAG_REQUEST_TIMEOUT` | `180` s | deadline total por consulta |
| `RAG_MAX_CONTEXT_TOKENS` | `12000` | orçamento aproximado do contexto |
| `RAG_AGENTIC_MAX_ITERATIONS` | `8` | iterações Agentic, limitado a 12 |
| `RAG_AGENTIC_MAX_TOOL_CALLS` | `12` | chamadas de ferramenta, limitado a 32 |
| `RAG_AGENTIC_CRITIC_ROUNDS` | `1` | revisões críticas, limitado a 3 |
| `RAG_SELFRAG_MAX_RETRIES` | `1` | retries Self-RAG, limitado a 2 |
| `RAG_CIRCUIT_FAILURE_THRESHOLD` | `3` | falhas antes de abrir o circuito |
| `RAG_CIRCUIT_RECOVERY_SECONDS` | `30` s | espera antes de nova tentativa |
| `RAG_CORS_ORIGINS` | interfaces locais | allowlist CORS separada por vírgula |
| `RAG_RATE_LIMIT` / `RAG_RATE_WINDOW` | `30` / `60` s | limite em memória por IP |
| `RAG_USE_GRAPH` | `0` | habilita grafo na engine principal |
| `RAG_DATA_DIR` | `data/` | seleciona o corpus documental compartilhado |
| `RAG_RUNTIME_DIR` | `var/` | raiz dos artefatos mutáveis de todas as engines |
| `RAG_DB_DIR` | `var/<engine>/chroma_db` | usa um índice ChromaDB separado |
| `RAG_INDEX_AUTO_DOWNLOAD` | `1` na Principal | baixa Release somente quando banco está ausente ou vazio |
| `RAG_INDEX_REPO` | `Thiagovilela2001/chat_ccdep_1.1` | repositório da Release do índice |
| `RAG_INDEX_TAG` | `vector-index-v1` | tag imutável da Release |
| `RAG_INDEX_ASSET` | `rag-principal-index-v1.tar.gz` | asset do banco pré-indexado |
| `RAG_INDEX_DOWNLOAD_TIMEOUT` | `600` | timeout HTTPS em segundos |
| `RAG_INDEX_READ_ONLY` | `1` com bootstrap | impede alteração ou reindexação do banco portátil |
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

## Verificação

```bash
python -m pytest -q
python -m pytest tests/unit -q
python -m pytest tests/integration -q
cd frontend && npm ci && npm run lint && npm test && npm run build
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
A análise técnica detalhada está em [ANALISE_PROJETO.md](ANALISE_PROJETO.md).
