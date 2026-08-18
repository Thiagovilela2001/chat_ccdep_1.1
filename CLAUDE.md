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

## Skill automática: Proteção Social

Sempre que o usuário fizer pergunta ou pedir tarefa sobre **proteção social**,
leia e aplique:

    .agents/skills/social-protection-analysis/SKILL.md

Tópicos de ativação:
- Cadastro Único ou CadÚnico
- Programa Bolsa Família ou PBF
- Benefício de Prestação Continuada ou BPC
- Transferência de renda, Regra de Proteção e famílias beneficiárias

## Arquitetura do projeto

Pacote instalável em `src/rag_ccdep/`. Quatro engines ficam em
`src/rag_ccdep/engines/`; Meta RAG fica em `src/rag_ccdep/orchestrator/`; código
compartilhado fica em `src/rag_ccdep/core/`. Não adicionar hacks de `sys.path`.
Dados de entrada ficam em `data/`; índices, grafo e caches mutáveis ficam em
`var/<engine>/`.

| Módulo | Função |
|---|---|
| `src/rag_ccdep/core/ingestion.py` / `processing.py` / `indexing.py` | Pipeline PDF → nós → ChromaDB (+ cache BM25 ancorado ao db_path) |
| `src/rag_ccdep/core/text_retriever.py` | Retrieval híbrido (Vector + BM25) para texto narrativo |
| `src/rag_ccdep/core/tables_retriever.py` | Extração de dados de tabelas estáticas via pandas |
| `src/rag_ccdep/core/timeseries_retriever.py` | Extração e análise de séries temporais via pandas |
| `src/rag_ccdep/core/structured_output.py` | Validação de JSON tabular produzido pelo LLM; sem execução de código |
| `src/rag_ccdep/core/safe_exec.py` | Helper legado restrito; não usado pelo fluxo RAG e não é isolamento de segurança |
| `src/rag_ccdep/core/numerical_validator.py` | Confere números da resposta contra as fontes |
| `src/rag_ccdep/core/labor_market_skill.py` | Carrega a skill e detecta queries de mercado de trabalho |
| `src/rag_ccdep/core/api_security.py` | API keys, CORS/CSP, headers defensivos e rate limiting |
| `src/rag_ccdep/core/api_models.py` / `query_service.py` | Schemas e fluxo HTTP compartilhado pelas engines |
| `src/rag_ccdep/core/runtime.py` | Deadline, limites de iteração e orçamento de contexto |
| `src/rag_ccdep/engines/<engine>/startup.py` | Inicialização da engine: indexação, LLMs, retrievers |
| `src/rag_ccdep/core/llm.py` | Fábrica única de LLM: escolhe o provedor (Maritaca/OpenAI) por env |
| `scripts/index_artifact.py` | Publica e instala índice portátil via GitHub Releases |
| `src/rag_ccdep/engines/<engine>/query_interpreter.py` | Roteia query para fontes + detecta `is_labor_market` |
| `src/rag_ccdep/engines/<engine>/api.py` | FastAPI: endpoint POST /query |
| `rag-ccdep <engine>` | Entrypoint: servidor HTTP ou CLI interativo |

### Provedor de LLM (`src/rag_ccdep/core/llm.py`)

Todo o projeto usa a API no formato OpenAI; o provedor é configurável por
ambiente (embeddings continuam locais — `BAAI/bge-m3`). Todos os sites de LLM
passam por `make_llm(...)` (LlamaIndex) ou `openai_client_kwargs()` (cliente
cru), então trocar de LLM é só configuração:

| Env | Padrão | Função |
|---|---|---|
| `RAG_LLM_PROVIDER` | `maritaca` | `maritaca` \| `openai` |
| `RAG_LLM_MODEL` | `sabia-4` (maritaca) | Modelo de síntese |
| `RAG_INTERP_MODEL` | `sabia-4` (maritaca) | Interpretação/crítica/enriquecimento |
| `RAG_POPUP_MODEL` | `sabiazinho-4` (maritaca) | Explicações curtas dos popups numéricos |
| `RAG_LLM_BASE_URL` / `RAG_LLM_API_KEY` | — | Sobrescrevem base/chave (provedor custom) |

Chaves: Maritaca usa `MARITACA_API_KEY`; OpenAI usa `OPENAI_API_KEY`.
Nomes de modelo fora do catálogo OpenAI (ex.: `sabia-4`) são registrados em
tempo de execução no catálogo do LlamaIndex (`_register_model` em `llm.py`),
usando a classe `OpenAI` base — sem depender do pacote `openai-like`, cuja
versão colide com o `llama-index-llms-openai` fixado pelo core.

## Dados indexados

Boletins de Conjuntura Paulista e Seade Social em `/data/`, incluindo
`/data/seade_social/painel/` e `/data/seade_social/trabalho/`.
Banco vetorial: ChromaDB em `var/<engine>/chroma_db/`.
Testes: `tests/unit/` e `tests/integration/` (pytest).

## Avaliação

`python evaluate.py --split dev|test|adversarial|all`
Métricas: Faithfulness, ContextPrecision, ContextRecall (RAGAS) + refusal_accuracy.
O benchmark avalia somente `rag_principal`; o judge padrão é o Maritaca `sabia-4`
e usa `MARITACA_API_KEY` (ou `RAGAS_JUDGE_API_KEY`).
