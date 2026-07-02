# Plano de Implementação — Meta RAG (Orquestrador Inteligente)

*Documento de design. Objetivo: consolidar as quatro variantes RAG existentes
(`rag_principal`, `rag_agentic`, `rag_raptor`, `rag_selfrag`) sob uma única
camada de orquestração — um **Meta RAG** — que analisa cada pergunta e
**seleciona dinamicamente a melhor estratégia de recuperação**, encaminhando a
consulta para a engine escolhida. **As quatro engines permanecem independentes
e inalteradas**; toda a inteligência nova vive na camada de orquestração.*

> **Modo primário: single-best.** O orquestrador decide **qual engine executar**
> e encaminha a pergunta para ela (requisitos 4 e 6). A execução de **múltiplas
> engines é opcional** (requisito 8), acionada apenas em empate/ambiguidade,
> com seleção da melhor resposta por critérios definidos.

> **Roteamento só pela consulta.** A decisão considera **exclusivamente as
> características da pergunta** — intenção, tipo de informação, precisão ×
> abrangência, termos técnicos, complexidade, e necessidade de busca lexical ×
> semântica × híbrida — **nunca o conteúdo do corpus** (requisito 5), pois todas
> as engines operam sobre os mesmos 23 documentos.

---

## 1. Premissas e diagnóstico da base atual

Antes da arquitetura, três fatos do código atual que **determinam** o desenho:

1. **Interface uniforme entre as engines.** As quatro variantes já expõem a
   mesma assinatura assíncrona:
   ```python
   async def answer(question: str, sources: list[str],
                    rewritten_query: str, is_labor_market: bool = False
                    ) -> tuple[str, list]   # (texto_resposta, source_nodes)
   ```
   e a mesma fábrica `initialize(base_dir, data_dir=None) -> (engine, interp_llm)`.
   → **O orquestrador pode tratar cada engine como um backend intercambiável,
   sem tocar em nenhuma delas.**

2. **Mesmo corpus, estratégias distintas.** As quatro variantes indexam os
   **mesmos 23 Boletins de Conjuntura Paulista** (`/data`), cada uma em seu
   próprio `chroma_db`. Portanto **não há "múltiplos domínios de conhecimento"**
   a rotear — há **múltiplas estratégias de recuperação sobre a mesma base**.
   A única especialização de *domínio* é a **skill de mercado de trabalho**, já
   detectada por `is_labor_market`.

   > Consequência prática: o roteamento não escolhe "qual base", e sim
   > "qual estratégia tem maior probabilidade de recuperar a evidência certa
   > para *o formato* desta pergunta" (pontual, ampla, relacional, multi-hop,
   > verificação factual).

3. **Já existe um roteador de baixo nível.** Dentro de cada variante,
   `interpret_query()` decide *sub-fontes* (`text`, `tables`, `timeseries`,
   `graph`) via LLM. O orquestrador adiciona um **roteador de alto nível** por
   cima (qual engine), reaproveitando o de baixo nível intacto.

### Perfil funcional de cada variante (o que cada estratégia faz melhor)

| Variante | Estratégia | Forte em | Fraca em | Custo/latência |
|---|---|---|---|---|
| `rag_principal` | Híbrido Vector+BM25 + retrievers de tabela/série + grafo | Fatos pontuais, números, tabelas, séries temporais, relações entre entidades | Perguntas amplas/temáticas que exigem síntese de muitos trechos | Baixo (1 passada) |
| `rag_raptor` | Índice hierárquico (folhas + resumos multinível) | Perguntas amplas, comparações de período, "panorama/resumo de X" | Detalhe numérico fino | Baixo–médio |
| `rag_agentic` | Loop iterativo com function calling | Multi-hop, decomposição, perguntas que exigem várias buscas encadeadas | Latência/custo; overkill p/ lookup simples | Alto (N rounds) |
| `rag_selfrag` | Self-reflective (RETRIEVE?→ISREL→GENERATE→ISSUP→RETRY) | Alta fidelidade, redução de alucinação, verificação de suporte | Latência (críticas via LLM) | Alto |

---

## 2. Arquitetura proposta

Nova camada **`rag_orchestrator/`** — um módulo irmão dos existentes, que os
**consome como bibliotecas**. Nenhuma variante é modificada.

```
                        ┌──────────────────────────────────────────┐
   Pergunta  ─────────► │            RAG ORCHESTRATOR               │
   do usuário           │                                          │
                        │  1. QueryAnalyzer  (classificação        │
                        │     semântica de intenção + escopo)      │
                        │            │                             │
                        │            ▼                             │
                        │  2. Router  (política de seleção         │
                        │     de estratégia + confiança)           │
                        │       │            │                     │
                        │   single-best   fan-out (top-k)          │
                        │       │            │                     │
                        │       ▼            ▼                     │
                        │  3. EngineRegistry (lazy load, embed     │
                        │     model compartilhado)                 │
                        │   ┌──────┬───────┬───────┬──────────┐    │
                        │   │princ.│raptor │agentic│ selfrag  │    │
                        │   └──┬───┴───┬───┴───┬───┴────┬─────┘    │
                        │      └───────┴───────┴────────┘          │
                        │            │ (source_nodes + respostas)  │
                        │            ▼                             │
                        │  4. Fusion/Aggregation                   │
                        │     (síntese sobre união de evidências   │
                        │      OU seleção do melhor candidato)     │
                        │            │                             │
                        │            ▼                             │
                        │  5. QualityGate                          │
                        │     (numerical_validator + grounding     │
                        │      check + regra de recusa)            │
                        └────────────┬─────────────────────────────┘
                                     ▼
                        Resposta + fontes rastreáveis + rota usada
```

### Estrutura de arquivos (novo módulo, ~reuso máximo)

```
rag_orchestrator/
├── main.py                 # entrypoint: servidor FastAPI (:8010) ou --cli
└── src/
    ├── registry.py         # carrega/gerencia as 4 engines (lazy + embed compartilhado)
    ├── query_analyzer.py   # classificação semântica da intenção (LLM + embeddings)
    ├── router.py           # política de seleção de engine(s) + confiança
    ├── fusion.py           # agregação multi-engine e síntese final
    ├── quality_gate.py     # validação numérica + grounding + recusa
    ├── orchestrator.py     # pipeline: analyze → route → run → fuse → gate
    └── api.py              # POST /query (mesmo contrato QueryResponse já existente)
```

**Reuso direto (sem alteração):** as 4 engines, `interpret_query` de cada
variante, `numerical_validator.py`, `labor_market_skill.py`, o contrato
`QueryResponse` da API e o **chatbox Streamlit já pronto** (basta adicionar a
opção "Unificado" apontando para :8010).

---

## 3. Mecanismo de roteamento das consultas

Roteamento em **duas etapas**, priorizando precisão e custo (requisitos 5 e 6):

### Etapa A — Análise semântica da intenção (`query_analyzer.py`)

Uma **única chamada LLM barata** (ex.: `gpt-5-mini`, temperatura 0, saída JSON
estrita) que classifica a pergunta em dimensões — **não** por palavras-chave,
mas por interpretação semântica (requisitos 1 e 2):

```jsonc
{
  "intent": "consulta_dado | comparar | resumir | explicar | verificar",  // intenção
  "query_type": "pontual | tabular | temporal | ampla | relacional | multi_hop | verificacao",
  "semantic_domain": "emprego | pib | industria | precos | comercio | geral | ...",  // domínio
  "specificity": "especifica | intermediaria | ampla",   // nível de especificidade
  "expected_answer": "numerico | tabela | serie | narrativo | comparativo",  // tipo esperado
  "priority": "precisao | abrangencia",     // precisão × abrangência
  "retrieval_need": "lexical | semantica | hibrida",  // preferência de busca
  "technical_terms": true/false,             // terminologia técnica
  "complexity": "baixa | media | alta",
  "linguistic_patterns": ["comparativo (\"entre X e Y\")", "quantitativo (\"qual foi\")"],  // padrões da formulação
  "needs_multi_hop": true/false,
  "is_labor_market": true/false,
  "in_scope": true/false,          // fora do escopo dos Boletins → recusa
  "entities": ["PIB", "indústria de transformação", "2023"],
  "period": "1º trim 2023 | 2020–2024 | null",
  "confidence": 0.0–1.0,
  "reasoning": "1 frase"
}
```

> As dimensões acima cobrem **integralmente** a lista de critérios do princípio 2
> (intenção, domínio semântico, nível de especificidade, precisão × abrangência,
> terminologia técnica, complexidade, tipo esperado de resposta, preferência
> lexical/semântica/híbrida, e padrões linguísticos da formulação). Todas são
> derivadas **exclusivamente da pergunta**, antes de qualquer recuperação, nunca
> do corpus (princípio 2 / requisito 5).

Para reduzir custo e latência, uma **rota rápida por embeddings** (opcional,
ativável): mantém-se um "perfil-embedding" por estratégia (frases-protótipo do
que cada uma responde melhor); a similaridade de cosseno da pergunta com os
perfis dá um roteamento determinístico e barato quando a confiança é alta,
caindo para a classificação LLM apenas em casos ambíguos.

### Etapa B — Política de seleção (`router.py`)

Mapa **determinístico e auditável** de classificação → estratégia(s):

| `query_type` / sinal | Estratégia primária | Justificativa |
|---|---|---|
| `pontual`, `tabular`, `temporal` | **principal** (retrievers especializados) | Precisão numérica/tabela/série |
| `relacional` (entidade↔entidade) | **principal + grafo** (`use_graph`) | GraphRetriever cobre relações |
| `ampla` / `comparativo` | **raptor** | Resumos hierárquicos multinível |
| `multi_hop` / `complexity=alta` | **agentic** | Decomposição iterativa |
| `verificacao` / risco de alucinação | **selfrag** | Crítica de suporte embutida |
| `in_scope=false` | **nenhuma** (recusa imediata) | Evita custo desnecessário (req. 6) |
| `is_labor_market=true` | injeta skill (já automático) + preferir principal/raptor | Domínio especializado |

Dimensões secundárias que **desempatam** dentro da política acima:

| Sinal | Efeito no roteamento |
|---|---|
| `priority=precisao` | favorece **principal**/**selfrag** (fidelidade/números) |
| `priority=abrangencia` | favorece **raptor** (síntese ampla) |
| `retrieval_need=lexical` | favorece **principal** com peso BM25 (termos exatos/siglas) |
| `retrieval_need=semantica` | favorece **principal** (vetorial) / **raptor** |
| `retrieval_need=hibrida` | **principal** (fusão Vector+BM25 já é o forte dele) |
| `technical_terms=true` + `is_labor_market` | reforça injeção da skill |

**Regra de decisão (modo primário = single-best; multi-engine opcional):**
- `confidence ≥ τ_alta (≈0.75)` → **single-best**: encaminha para **uma** engine
  e retorna a resposta dela (fluxo padrão — requisitos 4, 6 e 9).
- `confidence < τ_alta` **ou empate de tipos** → **desempate** pelas dimensões
  secundárias acima; persistindo o empate, aplica-se a política de ambiguidade
  (§5), que pode **opcionalmente** executar 2 engines e escolher a melhor
  resposta (requisito 8).

Os limiares `τ` e a ativação do multi-engine são **configuráveis** (flag) e
calibrados na avaliação (§7). Por padrão, o sistema privilegia single-best.

---

## 4. Seleção e combinação entre RAGs (`fusion.py`)

- **Single-best (caminho padrão):** resposta e `source_nodes` da engine
  escolhida passam direto ao QualityGate. Máxima precisão de fonte, custo mínimo.

- **Fan-out (opcional — requisito 8, desligado por padrão):** executa 2 engines
  **em paralelo** (`asyncio.gather`) e combina. Só é acionado quando a flag de
  multi-engine está ligada e o desempate de §3B falha. Duas políticas de
  combinação, selecionáveis:

  1. **Síntese sobre união de evidências (recomendada):** unem-se os
     `source_nodes` das engines, deduplicam-se por `(source_file, página)`,
     re-ranqueiam-se por relevância, e uma **única** chamada LLM sintetiza a
     resposta final citando apenas esses trechos. Preserva rastreabilidade
     (requisito 8) e evita "colar" duas respostas concorrentes.
  2. **Seleção por juiz (LLM-as-judge):** as duas respostas + suas evidências
     são avaliadas por um verificador que escolhe a mais fundamentada. Mais
     barato que sintetizar, porém descarta evidência da perdedora.

  Padrão: **política 1** quando há sobreposição de evidências; política 2
  quando as engines divergem em fonte (requisito 7: avaliar relevância,
  qualidade e especialização antes de responder).

- **Deduplicação/priorização (requisito 7):** ao sobrepor bases, ordena-se por
  (a) score de relevância normalizado, (b) especialização da fonte para o tipo
  de resposta esperado (ex.: nó de tabela ganha peso em pergunta `tabular`),
  (c) recência do boletim quando a pergunta tem período aberto.

---

## 5. Tratamento de consultas ambíguas

Sinais de ambiguidade detectados no `query_analyzer`: baixa `confidence`,
empate entre `query_type`, termos vagos ("como está a economia?"), ausência de
período/entidade, ou pergunta multi-parte.

Estratégia em cascata:
1. **Fan-out + fusão** (política 1) entre as 2 estratégias mais prováveis —
   default automático, mantém o fluxo sem intervenção do usuário.
2. **Pergunta de esclarecimento** (modo interativo/chat): quando faltam
   parâmetros essenciais (período, região), o orquestrador pode devolver uma
   pergunta curta em vez de adivinhar. Configurável (ligado no Streamlit,
   desligado na API batch/avaliação).
3. **Fallback seguro:** se ainda assim não houver evidência suficiente, aplica
   a regra de recusa padrão ("A informação não consta nos documentos
   fornecidos") em vez de alucinar (requisito 8).

---

## 6. Critérios de avaliação da qualidade das respostas (`quality_gate.py`)

Reaproveita mecanismos que já existem, encadeados após a fusão:

1. **Validação numérica** (`numerical_validator.validate_numbers`): confere
   cada número da resposta contra os `source_nodes`. Números não verificados
   são sinalizados (já exposto no `QueryResponse.validation`).
2. **Grounding check** (reuso do padrão *ISSUP* do SelfRAG): uma verificação
   leve de que cada afirmação-chave tem suporte nos trechos citados. Se abaixo
   do limiar → nova tentativa (fan-out ampliado) ou recusa.
3. **Regra de escopo/recusa:** herdada do prompt de síntese atual.
4. **Rastreabilidade (requisito 8):** a resposta sempre carrega a lista de
   fontes (`sources[]`) e a rota usada (`route`), permitindo auditar qual(is)
   estratégia(s) e quais documentos fundamentaram cada resposta.

---

## 7. Métricas de desempenho

Reaproveita `evaluate.py` e os splits `dev/test/adversarial` já existentes,
acrescentando métricas específicas de orquestração.

**Qualidade da resposta (RAGAS + próprias):**
- Faithfulness, ContextPrecision, ContextRecall (por split).
- `refusal_accuracy` (split adversarial).
- % de respostas com **todos** os números verificados.

**Qualidade do roteamento (novo):**
- **Acurácia de roteamento:** rota escolhida × rota "ouro" (rotular os itens do
  `golden_dataset` com a estratégia ideal — novo campo `route`).
- **Taxa de single-engine:** % de consultas resolvidas com uma só engine
  (mede se o roteador evita consultas desnecessárias — requisito 6).
- **Taxa de fan-out** e ganho de qualidade quando há fan-out (o custo extra se
  paga?).
- **Custo:** tokens/consulta e **latência p50/p95** — comparando orquestrador
  × cada variante isolada.

**Baseline comparativo (fecha item pendente do projeto):** rodar o orquestrador
e as 4 variantes isoladas sobre os mesmos splits e tabular qualidade × custo ×
latência. É a "avaliação comparativa formal" que o relatório lista como
pendente.

**Meta inicial sugerida:** faithfulness do orquestrador ≥ melhor variante
isolada em cada split, com latência p50 ≤ a da variante `agentic`.

---

## 8. Exemplos de fluxo de execução

**Ex. 1 — Pontual/numérico.** *"Qual foi a taxa de desocupação no 3º trim de
2023 em SP?"*
→ analyzer: `pontual/temporal`, `is_labor_market=true`, conf 0.88 →
**single-best: principal** (retrievers de série/tabela) + skill injetada →
validação numérica confere o valor → resposta com fonte (boletim 3T2023, pág).

**Ex. 2 — Ampla/temática.** *"Faça um panorama da indústria paulista entre 2020
e 2024."*
→ analyzer: `ampla/comparativo`, conf 0.82 → **single-best: raptor** (resumos
multinível cobrem o período) → síntese com citações dos resumos.

**Ex. 3 — Multi-hop.** *"Como a variação do PIB da indústria se relacionou com
o emprego formal no setor no mesmo período?"*
→ analyzer: `multi_hop`, `complexity=alta`, conf 0.79 → **agentic** (decompõe:
PIB indústria → emprego formal → correlação) → resposta encadeada; grounding
check aprova.

**Ex. 4 — Ambígua.** *"Como está a economia?"*
→ analyzer: `ampla`, conf 0.41, sem período/entidade → **ambiguidade**:
no chat, devolve "De qual período e setor? (ex.: indústria, 2024)"; na API,
**fan-out raptor+principal** e funde um panorama recente.

**Ex. 5 — Fora de escopo.** *"Qual a taxa Selic definida pelo Copom?"*
→ analyzer: `in_scope=false` → **recusa imediata** (nenhuma engine chamada,
custo zero) → "A informação não consta nos documentos fornecidos."

**Ex. 6 — Verificação factual sensível.** *"O desemprego caiu ou subiu no último
boletim?"* com risco de inversão de sinal → **selfrag** (ISSUP evita inverter a
direção do dado) → resposta com suporte verificado.

---

## 9. Plano de execução (fases) e alterações necessárias

**Alterações nas variantes existentes: praticamente nenhuma.** A interface já é
uniforme. A única otimização recomendada é **compartilhar o embedding model**
entre as engines para não carregar 4× o BGE em memória (injetável via um
parâmetro opcional em `initialize`, com fallback ao comportamento atual).

| Fase | Entregável | Toca em |
|---|---|---|
| **0. Pré-requisito** | Consertar ambiente (`transformers<5`) p/ subir qualquer backend; pinar `requirements.txt` | infra |
| **1. Registry** | `registry.py`: carrega as 4 engines (lazy load sob demanda + embed compartilhado) | novo módulo |
| **2. Analyzer** | `query_analyzer.py`: classificação semântica LLM (JSON) + rota rápida por embeddings | novo |
| **3. Router** | `router.py`: política §3B + limiares de confiança | novo |
| **4. Orchestrator + Fusion** | `orchestrator.py`, `fusion.py`: pipeline single-best/fan-out + síntese | novo |
| **5. QualityGate** | `quality_gate.py`: validação numérica + grounding + recusa | reuso |
| **6. API + Chat** | `api.py` (contrato existente + campo `route`), `main.py` (:8010); opção "Unificado" no Streamlit | reuso |
| **7. Avaliação** | Rotular `golden_dataset` com `route`; estender `evaluate.py` c/ métricas de roteamento; baseline comparativo | reuso |

### Contrato da API (retrocompatível)

O `QueryResponse` atual é mantido, acrescido de metadados de orquestração:

```jsonc
{
  "answer": "...",
  "sources": [ {"file": "...", "score": 0.9} ],
  "validation": { "verified": 3, "total": 3, "unverified": [] },
  "route": {                       // NOVO — rastreabilidade da decisão
    "engines_used": ["principal"],
    "query_type": "pontual",
    "confidence": 0.88,
    "mode": "single_best",
    "reasoning": "pergunta numérica pontual com período definido"
  }
}
```

---

## 10. Riscos e decisões em aberto

- **Memória:** 4 engines no mesmo processo (ChromaDB + BM25 + grafo + embeddings).
  Mitigação: lazy load + embed compartilhado; ou manter cada engine em seu
  processo/porta e o orquestrador chamar via HTTP (mais isolado, custo de rede).
  → **Decisão a tomar:** in-process (mais rápido) × multi-processo (mais isolado).
- **Custo do analyzer:** +1 chamada LLM por consulta. Mitigável pela rota rápida
  por embeddings nos casos de alta confiança.
- **Rótulos de rota "ouro":** exigem curadoria manual do `golden_dataset` para
  medir acurácia de roteamento — esforço pontual, mas necessário para §7.
- **Fan-out vs. custo:** política de combinação por síntese dobra a recuperação;
  calibrar `τ` para acionar fan-out só quando compensa.

---

## 11. Extensibilidade — adicionar novas estratégias (requisito 7)

O orquestrador trata cada engine como um **plugin** descrito por um *perfil*.
Adicionar uma 5ª estratégia (ex.: `rag_longcontext`) no futuro exige **apenas**:

1. Que o novo módulo exponha a interface já padrão: `initialize(...)` e
   `async answer(question, sources, rewritten_query, is_labor_market)`.
2. **Registrar** a engine no `registry.py` com um *perfil* declarativo — sem
   tocar em analyzer, fusion, API ou nas demais engines:

O perfil descreve **características, pontos fortes, limitações e tipos de
consulta** para os quais a estratégia rende melhor (princípio 3):

```python
# registry.py — registro declarativo (adicionar estratégia = 1 entrada)
@dataclass
class StrategyProfile:
    loader: Callable            # -> initialize(...) da variante (lazy)
    description: str            # característica geral da estratégia
    strengths: list[str]        # pontos fortes
    limitations: list[str]      # limitações (princípio 3)
    good_for: list[str]         # query_type(s) em que rende melhor
    priority: list[str]         # "precisao" | "abrangencia"
    retrieval: list[str]        # "lexical" | "semantica" | "hibrida"
    complexity: list[str] = ()  # níveis de complexidade que suporta bem
    prototypes: list[str] = ()  # frases-protótipo p/ a rota rápida por embeddings

STRATEGIES = {
    "principal": StrategyProfile(
        loader=lambda: import_initialize("rag_principal"),
        description="Híbrido Vector+BM25 + retrievers de tabela/série + grafo",
        strengths=["fatos pontuais", "números", "tabelas", "séries", "relações"],
        limitations=["síntese ampla de muitos trechos"],
        good_for=["pontual", "tabular", "temporal", "relacional"],
        priority=["precisao"], retrieval=["lexical", "semantica", "hibrida"],
    ),
    "raptor":  StrategyProfile(
        loader=lambda: import_initialize("rag_raptor"),
        description="Índice hierárquico (folhas + resumos multinível)",
        strengths=["perguntas amplas", "comparações de período", "panoramas"],
        limitations=["detalhe numérico fino"],
        good_for=["ampla", "comparativo"], priority=["abrangencia"],
        retrieval=["semantica"]),
    "agentic": StrategyProfile(
        loader=lambda: import_initialize("rag_agentic"),
        description="Loop iterativo com function calling",
        strengths=["multi-hop", "decomposição de perguntas encadeadas"],
        limitations=["latência/custo altos", "overkill p/ lookup simples"],
        good_for=["multi_hop"], complexity=["alta"], priority=["precisao"],
        retrieval=["hibrida"]),
    "selfrag":  StrategyProfile(
        loader=lambda: import_initialize("rag_selfrag"),
        description="Self-reflective (RETRIEVE?→ISREL→GENERATE→ISSUP→RETRY)",
        strengths=["alta fidelidade", "redução de alucinação", "verificação"],
        limitations=["latência (críticas via LLM)"],
        good_for=["verificacao"], priority=["precisao"], retrieval=["semantica"]),
    # nova estratégia entra aqui, e só aqui.
}
```

O `router.py` consulta os perfis por *matching* com a classificação da consulta
— então **o mapa de roteamento se estende sozinho** ao registrar um perfil novo.
O `perfil` também alimenta a "rota rápida por embeddings" (frases-protótipo por
estratégia). Nenhuma alteração estrutural no restante do sistema (requisito 7).

---

## 12. Rastreabilidade dos requisitos

| Req. | Onde é atendido |
|---|---|
| 1. Engines independentes e inalteradas | §2 (módulo irmão), §9 ("alterações: nenhuma") |
| 2. Orquestrador recebe a pergunta antes da recuperação | §2 (pipeline), §3A |
| 3. Interpretação semântica (6 dimensões) | §3A (schema do analyzer) |
| 4. Seleção automática da engine | §3B (política), §11 (perfis) |
| 5. Roteia só por características da consulta, não pelo corpus | §1, notas do topo, §3A |
| 6. Não consultar todas as bases sem necessidade | §3B (single-best padrão), §8 ex. 5 |
| 7. Incluir novas estratégias sem alterar o resto | §11 (registry de plugins) |
| 8. Desempate ou execução múltipla opcional | §3B (desempate), §4 (fan-out opcional), §5 |
| 9. Reutilizar `answer()` e `interpret_query` | §1, §2 (reuso), §9 |

**Princípios da arquitetura → onde vivem:**

| Princípio | Seção |
|---|---|
| 1. Single Best Routing (multi-engine só opcional) | §3B, §4 |
| 2. Roteamento exclusivamente pela consulta (9 critérios) | §3A |
| 3. Engines especializadas isoladas, com perfil declarativo | §1 (perfis), §11 |
| 4. Extensibilidade (registrar + perfil + `answer()`) | §11 |
| 5. Reutilização máxima (`answer`, `interpret_query`, índices Chroma) | §1, §2, §9 |
| 6. Meta RAG = camada de decisão, não um novo RAG | §2 (arquitetura), topo |

---

*Foco do plano, conforme solicitado: toda a inteligência nova vive na camada
`rag_orchestrator/` (análise semântica, roteamento, seleção e — opcionalmente —
fusão). As quatro variantes e seus índices permanecem intactos, reutilizados
como backends intercambiáveis por sua interface `answer()` já uniforme. Ponto de
entrada único (`POST /query` em :8010) → analisa → escolhe estratégia → retorna
a resposta da engine selecionada, com a rota registrada para auditoria.*
