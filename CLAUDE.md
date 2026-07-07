# RAG Estatístico SP — Instruções para Claude Code

## Skill automática: Mercado de Trabalho

Sempre que o usuário fizer uma pergunta ou pedir uma tarefa relacionada a
**mercado de trabalho**, leia e aplique a skill antes de responder:

    .agents/skills/labor_market_analysis/SKILL.md

Tópicos que ativam a skill (exemplos):
- Emprego, desemprego, desocupação, informalidade
- CAGED, RAIS, PNAD Contínua (PNADC), PED
- Salários, remuneração, rendimento do trabalho
- Taxa de desocupação, saldo de empregos, rotatividade
- Mercado de trabalho formal/informal

## Arquitetura do projeto

Quatro engines RAG (`rag_principal/`, `rag_agentic/`, `rag_raptor/`,
`rag_selfrag/`) + orquestrador (`rag_orchestrator/`, Meta RAG). A
infraestrutura comum vive em **`rag_core/`** (raiz do repo); cada engine mantém
em `<engine>/src/` apenas o que a diferencia. O `src/__init__.py` de cada
engine adiciona o diretório-pai ao `sys.path` para tornar `rag_core` importável.

| Módulo | Função |
|---|---|
| `rag_core/ingestion.py` / `processing.py` / `indexing.py` | Pipeline PDF → nós → ChromaDB (+ cache BM25 ancorado ao db_path) |
| `rag_core/text_retriever.py` | Retrieval híbrido (Vector + BM25) para texto narrativo |
| `rag_core/tables_retriever.py` | Extração de dados de tabelas estáticas via pandas |
| `rag_core/timeseries_retriever.py` | Extração e análise de séries temporais via pandas |
| `rag_core/safe_exec.py` | Sandbox p/ código pandas gerado por LLM (AST + builtins + timeout) |
| `rag_core/numerical_validator.py` | Confere números da resposta contra as fontes |
| `rag_core/labor_market_skill.py` | Carrega a skill e detecta queries de mercado de trabalho |
| `rag_core/api_security.py` | API key opcional (`RAG_API_KEY`) + rate limiting (`RAG_RATE_LIMIT`) |
| `<engine>/src/startup.py` | Inicialização da engine: indexação, LLMs, retrievers |
| `rag_core/llm.py` | Fábrica única de LLM: escolhe o provedor (Maritaca/OpenAI) por env |
| `<engine>/src/query_interpreter.py` | Roteia query para fontes + detecta `is_labor_market` |
| `<engine>/src/api.py` | FastAPI: endpoint POST /query |
| `<engine>/main.py` | Entrypoint: servidor HTTP ou CLI interativo |

### Provedor de LLM (`rag_core/llm.py`)

Todo o projeto usa a API no formato OpenAI; o provedor é configurável por
ambiente (embeddings continuam locais — `BAAI/bge-m3`). Todos os sites de LLM
passam por `make_llm(...)` (LlamaIndex) ou `openai_client_kwargs()` (cliente
cru), então trocar de LLM é só configuração:

| Env | Padrão | Função |
|---|---|---|
| `RAG_LLM_PROVIDER` | `maritaca` | `maritaca` \| `openai` |
| `RAG_LLM_MODEL` | `sabia-4` (maritaca) | Modelo de síntese |
| `RAG_INTERP_MODEL` | `sabia-4` (maritaca) | Interpretação/crítica/enriquecimento |
| `RAG_LLM_BASE_URL` / `RAG_LLM_API_KEY` | — | Sobrescrevem base/chave (provedor custom) |

Chaves: Maritaca usa `MARITACA_API_KEY`; OpenAI usa `OPENAI_API_KEY`.
Nomes de modelo fora do catálogo OpenAI (ex.: `sabia-4`) exigem `OpenAILike`,
por isso `llama-index-llms-openai-like` está nas dependências.

## Dados indexados

Boletins de Conjuntura Paulista (PDFs, 2022–2025) em `/data/`.
Banco vetorial: ChromaDB em `<engine>/chroma_db/`.
Testes: `rag_principal/tests/` (pytest) e `rag_orchestrator/tests/` (scripts).

## Avaliação

`python evaluate.py --split dev|test|adversarial|all`
Métricas: Faithfulness, ContextPrecision, ContextRecall (RAGAS) + refusal_accuracy.
