# Linha do Tempo do Desenvolvimento: RAG Estatístico SP (`rag_ccdep`)

---

## Tabela Cronológica de Eventos

| Data ou período aproximado | Funcionalidade implementada | Problema ou desafio encontrado | Solução adotada | Impacto no projeto | Evidência documental |
|---|---|---|---|---|---|
| **Antes de mar/2026** (fase inicial) | Concepção e planejamento do sistema RAG. Definição da stack: LlamaIndex, HuggingFace BGE, GPT-4o, ChromaDB. Módulos planejados: `ingestion.py`, `processing.py`, `indexing.py`, `retriever.py`, `qa_chain.py`. | Escolha do framework de orquestração; necessidade de embeddings locais para não expor dados sensíveis a APIs externas. | Adoção de embeddings open-source locais (`BAAI/bge-small-en-v1.5`) via HuggingFace; ChromaDB local como banco vetorial; LlamaIndex como orquestrador inicial. | Estabeleceu a arquitetura base de todo o projeto e a decisão de soberania dos dados (sem transmissão a APIs de terceiros para embeddings). | `implementation_plan.md.resolved` |
| **Antes de mar/2026** (primeira implementação) | Pipeline RAG completo (`rag_principal`): ingestão de PDFs, segmentação com `SentenceSplitter`, extração de metadados (`TitleExtractor`, `KeywordExtractor`), indexação no ChromaDB, cadeia de QA com GPT-4o (temperatura 0). CLI interativo via `main.py`. | Necessidade de preservar contexto numérico das tabelas dos Boletins de Conjuntura; risco de alucinações numéricas. | Armazenamento separado de texto narrativo e dados tabulares; prompt do LLM com regra explícita de citação de fonte e temperatura zero. | Sistema capaz de responder perguntas sobre os Boletins de Conjuntura Paulista com referência direta às páginas-fonte. | `walkthrough.md.resolved`; código em `rag_principal/src/` |
| **25/03/2026** | Primeira avaliação quantitativa sistemática do sistema. Execução de pipeline de avaliação com métricas RAGAS. | Necessidade de medir a qualidade da geração de forma objetiva antes de expandir o sistema. | Implementação de `evaluate.py` com métricas RAGAS: Faithfulness, ContextPrecision, ContextRecall. | Estabeleceu baseline quantitativo do projeto: faithfulness=0,767; context_precision=0,856; context_recall=0,711. | `evaluation/results.json` (timestamp `2026-03-25T16:28:41`) |
| **08/04/2026** | Avaliação nos splits `test` e `adversarial`. Introdução de conjunto adversarial para testar recusa a perguntas fora do escopo (Rio de Janeiro, taxa Selic, déficit fiscal). | Garantir que o sistema não alucine em perguntas sobre dados ausentes nos documentos indexados. | Prompt de síntese com regra explícita: *"Se a informação não está no contexto, responda exatamente: 'A informação não consta nos documentos fornecidos.'"* | refusal_accuracy=1,0 no split adversarial; faithfulness=0,801 e context_recall=1,0 no split test — melhora sobre a avaliação de março. | `evaluation/results_test.json` (timestamp `2026-04-08T17:44:28`); `evaluation/results_adversarial.json` (timestamp `2026-04-08T17:48:29`) |
| **Antes de mai/2026** | Criação de três variantes RAG adicionais: `rag_agentic` (loop agentic com function calling OpenAI), `rag_raptor` (indexação hierárquica RAPTOR com folhas + resumos multinível), `rag_selfrag` (Self-Reflective RAG baseado em Asai et al. 2023 — sem fine-tuning, critique tokens via chamadas LLM). | Abordagem única de retrieval mostrava limitações em perguntas de diferentes tipos (amplas vs. pontuais). Não havia baseline comparativo entre estratégias. | Cada variante implementada com a mesma interface (`answer()`) para permitir avaliação comparativa direta via `evaluate.py`. | Projeto expandido para plataforma de comparação de arquiteturas RAG, com quatro abordagens distintas operando sobre o mesmo corpus. | Estrutura de diretórios `rag_agentic/`, `rag_raptor/`, `rag_selfrag/`; docstrings em `agent_engine.py`, `raptor_engine.py`, `self_rag_engine.py` |
| **Antes de 04/05/2026** | Criação da Skill de Mercado de Trabalho (`.agents/skills/labor_market_analysis/SKILL.md`). Integração do módulo `labor_market_skill.py` em todas as variantes. Trigger automático em `CLAUDE.md`. | Perguntas sobre mercado de trabalho exigem conhecimento especializado sobre conceitos metodológicos (PNAD Contínua, CAGED, RAIS, taxa de subutilização, plano amostral). O LLM genérico não interpretava corretamente esses indicadores. | Skill especializada com: documentação de fontes (PNADC, RAIS, CAGED, PME), variáveis-chave, fórmulas de cálculo, exemplos de código e referências bibliográficas. Bloco `[Conhecimento Especializado]` injetado no prompt quando detectada query de mercado de trabalho. | Melhora qualitativa nas respostas sobre emprego, desemprego e rendimento. Avaliação dev (2026-05-04) registrou faithfulness=0,744 no conjunto com queries de mercado de trabalho. | `CLAUDE.md`; `evaluation/results_dev.json` (timestamp `2026-05-04T13:28:16`); `.agents/skills/labor_market_analysis/SKILL.md` |
| **Antes de mai/2026** | Adição do método de **grafo de conhecimento** ao `rag_principal` (`graph_indexing.py`, `graph_retriever.py`), integrado ao `analysis_engine.py`. | Ausência de cobertura para perguntas que se beneficiam de relações entre entidades ao longo dos boletins. | `GraphRetriever` construído sobre `PropertyGraphIndex` do LlamaIndex, acionado pelo roteamento de query. | `rag_principal` passou a combinar retrievers de texto (Vector+BM25), tabelas, séries temporais e grafo. | `rag_principal/src/graph_indexing.py`, `rag_principal/src/graph_retriever.py` |
| **13/05/2026** | Dockerização do projeto. Criação de `Dockerfile` (Python 3.11, Ghostscript) e `docker-compose.yml` com os serviços RAG. | Porta 8000 já ocupada no ambiente Windows: *"listen tcp 0.0.0.0:8000: bind: Foi feita uma tentativa de acesso a um soquete de uma maneira que é proibida"*. | Porta alterada para 8080 no `docker-compose.yml`. | Sistema containerizado e reprodutível; API disponível em `http://localhost:8080`; deployment facilitado para outras máquinas. | `Dockerfile`, `docker-compose.yml`, `DOCKER.md` |
| **Antes de mai/2026** | Exposição de cada variante como API REST (`api.py` em cada módulo, endpoint `POST /query`) e frontend web (`frontend/`). | Necessidade de operar e comparar as variantes por uma interface HTTP única e por um cliente web. | `FastAPI` em cada módulo com `main.py` capaz de subir servidor HTTP ou CLI interativo; frontend estático em `frontend/`. | Cada variante RAG operável via API e navegador. | Commit `6d3ca74` ("APIs REST para todos os módulos RAG + frontend + Docker"); `rag_*/src/api.py`, `frontend/` |
| **~jul/2026** (pendente, não commitado) | Correção no `rag_raptor/src/raptor_indexing.py`: modelo de resumo alterado de `gpt-5-mini` para `gpt-4o-mini`; metadados `raptor_level` e `cluster_size` convertidos para `str()` antes de gravar no nó. | ChromaDB rejeita/limita metadados não-escalares; modelo de resumo desatualizado. | Conversão explícita de metadados numéricos para string e fixação do modelo de resumo. | Estabiliza a indexação hierárquica do RAPTOR. | `git diff rag_raptor/src/raptor_indexing.py` (working tree, ainda não commitado) |

---

## Síntese da Evolução do Projeto

### Fase 1 — Concepção e Protótipo Inicial (anterior a mar/2026)

O projeto nasceu com um objetivo claro: construir um sistema RAG capaz de responder perguntas sobre os **Boletins de Conjuntura Paulista** (Fundação Seade, 2020–2025) de forma precisa e rastreável. A principal decisão arquitetural desta fase foi a adoção de **embeddings locais** (`BAAI/bge-small-en-v1.5` via HuggingFace), que permitia manter os dados dentro da infraestrutura da instituição sem transmiti-los a APIs externas — decisão coerente com o caráter institucional do projeto. O LLM escolhido para geração foi a API da OpenAI (GPT-4o), operando com temperatura zero para minimizar alucinações.

A arquitetura inicial era um RAG clássico de único estágio: extração de texto dos PDFs → segmentação em chunks → indexação no ChromaDB → busca vetorial → resposta. O módulo de QA (`qa_chain.py`) tinha um prompt fortemente restritivo, obrigando citação explícita de fonte (nome do PDF e página) em cada afirmação.

### Fase 2 — Avaliação e Refinamento (mar–abr/2026)

A primeira avaliação formal (25/03/2026) revelou as limitações quantitativas do sistema: faithfulness de 0,767 indicava respostas parcialmente não suportadas pelo contexto; context_recall de 0,711 mostrava que o retriever nem sempre encontrava os trechos necessários. A introdução do conjunto **adversarial** (08/04/2026) — com perguntas sobre dados inexistentes nos documentos — confirmou que o mecanismo de recusa funcionava perfeitamente (refusal_accuracy = 1,0). A avaliação no split de teste (faithfulness = 0,801, context_recall = 1,0) demonstrou melhora após ajustes no prompt de síntese, que ganhou regras mais granulares sobre conflitos entre dados estruturados e narrativos.

### Fase 3 — Expansão para Múltiplas Variantes Arquiteturais (abr–mai/2026)

Esta foi a fase de maior expansão conceitual. Reconhecendo que um único paradigma RAG não cobre todos os tipos de perguntas igualmente bem, o projeto criou **quatro variantes arquiteturais comparáveis**:

- **`rag_principal`** — retrieval híbrido (Vector + BM25 + reranker) com múltiplos retrievers especializados (texto, tabelas, séries temporais) e um retriever de grafo de conhecimento.
- **`rag_agentic`** — loop iterativo com function calling da OpenAI, onde o LLM decide quais ferramentas chamar e quando parar.
- **`rag_raptor`** — índice hierárquico com folhas (chunks) e resumos automáticos multinível, permitindo perguntas específicas e amplas com uma única busca.
- **`rag_selfrag`** — self-reflective RAG baseado em Asai et al. (2023), com tokens de crítica implementados por chamadas LLM explícitas (RETRIEVE? → ISREL → GENERATE → ISSUP → RETRY).

Todas as variantes implementam a mesma interface (`answer()` assíncrono), o que permite avaliação comparativa direta pelo `evaluate.py`.

### Fase 4 — Skill Especializada de Mercado de Trabalho (abr–mai/2026)

A constatação de que perguntas sobre indicadores do mercado de trabalho (PNAD Contínua, CAGED, RAIS) exigiam conhecimento metodológico especializado levou à criação da **Skill de Análise de Mercado de Trabalho**. Esta skill — materializada em `.agents/skills/labor_market_analysis/SKILL.md` — documenta variáveis de microdados, fórmulas de cálculo, tratamento de quebras estruturais, análise setorial e regional, e referências bibliográficas. Quando detectada uma query de mercado de trabalho, o conteúdo da skill é injetado no prompt como bloco `[Conhecimento Especializado]`, orientando o LLM na interpretação correta dos indicadores.

### Fase 5 — Operacionalização: APIs REST, Frontend e Docker (mai/2026)

Cada variante foi exposta como **API REST** (`api.py`, endpoint `POST /query`) com um `main.py` capaz de subir servidor HTTP ou CLI interativo, acompanhada de um **frontend web** em `frontend/`. A containerização via **Docker** (`Dockerfile` + `docker-compose.yml`) marcou a transição do projeto de protótipo local para sistema operacional. A principal dificuldade foi um conflito de portas no ambiente Windows (porta 8000 ocupada), resolvido pela migração para 8080.

### Objetivos alcançados

- [x] Pipeline RAG funcional sobre os Boletins de Conjuntura Paulista (2020–2025, 23 PDFs em `data/`)
- [x] Quatro variantes arquiteturais RAG implementadas e avaliáveis (`rag_principal`, `rag_agentic`, `rag_raptor`, `rag_selfrag`)
- [x] Retriever de grafo de conhecimento no `rag_principal`
- [x] Framework de avaliação quantitativa com RAGAS (splits dev/test/adversarial)
- [x] refusal_accuracy = 1,0 no conjunto adversarial
- [x] Skill especializada de mercado de trabalho, integrada em todas as variantes
- [x] APIs REST + frontend web por variante
- [x] Containerização Docker

### Objetivos em desenvolvimento / planejados

> Itens abaixo foram **planejados** mas ainda **não estão implementados** no repositório versionado (branch `master`) até a data deste documento.

- [ ] **Camada semântica determinística em JSON** para substituir o roteamento por LLM. Atualmente `query_interpreter.py` ainda usa `llm.complete(...)` para decidir os retrievers; a migração para vocabulário semântico em JSON (`semantic-vocabulary.json`) segue pendente.
- [ ] **Método de grafo no `rag_selfrag`** — hoje o grafo existe apenas no `rag_principal`.
- [ ] **Variante de contexto longo** (`rag_longcontext`) e/ou `LongContextRetriever` no `rag_principal` — ainda não criados.
- [ ] **Gateway unificado** (`app.py`) para subir qualquer variante por um único entrypoint com UI web embutida — ainda não criado; cada variante roda por seu próprio `main.py`.
- [ ] Avaliação comparativa formal entre as variantes (análise quantitativa dos trade-offs).
- [ ] Avaliação do impacto da skill de mercado de trabalho nas métricas RAGAS.
- [ ] Possível implementação de avaliação humana complementar ao RAGAS.

---

## Eventos Potencialmente Relevantes para o Relatório Parcial

### Objetivos alcançados

1. **Pipeline RAG sobre corpus estatístico institucional** — Primeiro RAG construído sobre os Boletins de Conjuntura Paulista, corpus de natureza incomum (documentos com dados econômicos misturados a texto analítico e tabelas). Relevância: demonstra viabilidade técnica do uso de RAG para recuperação de informação estatística regional.

2. **Baseline de avaliação quantitativa** — Métricas RAGAS com splits dev/test/adversarial estabelecem referência numérica objetiva: faithfulness=0,801 (test), context_recall=1,0 (test), refusal_accuracy=1,0 (adversarial). Relevância científica: avaliação estruturada e reprodutível, comparável com a literatura de RAG.

3. **Quatro variantes RAG implementadas** — Implementação comparativa de RAG Clássico (com grafo), Agentic RAG, RAPTOR e Self-RAG sobre o mesmo corpus e com a mesma interface de avaliação. Esta plataforma experimental é o principal ativo científico do projeto.

4. **Skill especializada de domínio** — Injeção de conhecimento metodológico sobre mercado de trabalho no contexto do LLM como mecanismo explícito de orientação da geração. Abordagem relevante para RAG em domínios técnicos.

### Dificuldades encontradas e estratégias de superação

1. **Preservação de integridade numérica** — PDFs com tabelas numéricas exigiam estratégia para evitar que o LLM distorcesse ou interpolasse valores. Solução: regras de citação obrigatória no prompt e separação de retrievers por tipo de dado (texto / tabela / série temporal).

2. **Conflito de portas na dockerização** — Porta 8000 ocupada no ambiente Windows impedia subir a API em contêiner. Solução: migração para a porta 8080 no `docker-compose.yml`.

3. **Estabilidade da indexação hierárquica (RAPTOR)** — Metadados numéricos (`raptor_level`, `cluster_size`) e modelo de resumo desatualizado geravam falhas na gravação dos nós de resumo. Solução (pendente de commit): conversão de metadados para string e fixação do modelo `gpt-4o-mini`.

### Alterações realizadas sobre o projeto original

1. **Migração de orquestrador** — O plano original previa LlamaIndex como início com transição para LangChain. O projeto manteve LlamaIndex para indexação mas substituiu a cadeia de QA por chamadas diretas à API da OpenAI (`AsyncOpenAI`), eliminando a dependência de frameworks de orquestração de alto nível para geração.

2. **Expansão de módulo único para plataforma multi-variante** — O projeto original previa uma única pipeline RAG; a expansão para quatro variantes arquiteturais transformou o projeto em uma plataforma experimental comparativa.

3. **Adição de retriever de grafo de conhecimento** — Método de retrieval não previsto no plano original, adicionado ao `rag_principal`.

### Metodologia

1. **Corpus**: 23 Boletins de Conjuntura Paulista (Fundação Seade, 2020–2025), cobertura trimestral, em `data/`.
2. **Embeddings locais**: `BAAI/bge-small-en-v1.5` — escolha motivada por soberania de dados institucionais.
3. **Avaliação automatizada com RAGAS**: três splits (dev, test, adversarial) com métricas de Faithfulness, ContextPrecision, ContextRecall e refusal_accuracy.
4. **Estratégia de prompting anti-alucinação**: temperatura zero, citação obrigatória de fonte (nome do PDF + página), regras de conflito entre dados estruturados e narrativos.
5. **Isolamento de variáveis entre arquiteturas**: interface única `answer()` em todos os engines permite comparação direta por `evaluate.py` sem variação de corpus ou métricas.

### Resultados preliminares

| Métrica | Avaliação inicial (mar/2026) | Split test (abr/2026) | Split dev (mai/2026) | Split adversarial (abr/2026) |
|---|---|---|---|---|
| Faithfulness | 0,767 | 0,801 | 0,744 | — |
| ContextPrecision | 0,856 | 0,953 | 0,843 | — |
| ContextRecall | 0,711 | 1,000 | 0,733 | — |
| refusal_accuracy | — | — | — | 1,000 |

A melhora de faithfulness entre mar/2026 e abr/2026 (0,767 → 0,801) e a context_precision mais alta no split test (0,953) indicam que os ajustes no prompt de síntese e no sistema de citação tiveram efeito positivo. A ligeira queda no split dev (mai/2026) pode refletir a introdução de queries de mercado de trabalho — mais complexas conceitualmente — nesse conjunto de avaliação, antes de os ajustes da skill estarem plenamente integrados.

---

*Documento gerado em 03/06/2026 e revisado em 02/07/2026 para refletir o estado efetivamente versionado do repositório (branch `master`). Baseado em: código-fonte dos módulos (`rag_principal/`, `rag_agentic/`, `rag_raptor/`, `rag_selfrag/`), histórico `git log`, arquivos de avaliação (`evaluation/`), plano de implementação (`implementation_plan.md.resolved`) e walkthrough inicial (`walkthrough.md.resolved`). Itens planejados mas ainda não implementados (camada semântica JSON, grafo no SelfRAG, variante de contexto longo, gateway `app.py`) estão listados em "Objetivos em desenvolvimento / planejados".*
