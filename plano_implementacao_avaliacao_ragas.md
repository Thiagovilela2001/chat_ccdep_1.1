# Plano de implementação — raciocínio por papel e avaliação RAG confiável

**Status:** proposto — item 1 ainda não está pronto para commit  
**Decisão de fechamento atual:** `REPLAN`  
**Escopo:** `rag_principal`, fábrica compartilhada de LLM, validação numérica e `evaluate.py`  
**Data de referência:** 01/10/2026

## 1. Contexto

O item em avaliação tenta reduzir a latência do pipeline por três mudanças principais:

1. separar o modelo usado na extração estruturada por meio de `RAG_EXTRACT_MODEL`;
2. permitir o envio de `reasoning_effort=none` aos modelos compatíveis com a API OpenAI;
3. reforçar a extração de séries temporais para preservar todos os pares período/valor.

O candidato atual também foi medido com `RAG_QUERY_FUSION_QUERIES=1` e
`RAG_LLM_RERANK=0`. Essas variáveis precisam permanecer constantes nos próximos
experimentos para que o efeito do raciocínio possa ser atribuído corretamente.

### 1.1 Evidência quantitativa disponível

| Indicador | Baseline | Candidato atual | Leitura |
|---|---:|---:|---|
| Perguntas concluídas | 15/15 | 15/15 | sem erro de execução |
| Precisão numérica média | 0,838 | 0,871 | melhora agregada de 0,034 |
| Latência p50 | 24,35 s | 9,45 s | redução aproximada de 61% |
| Latência p95 | 35,68 s | 12,25 s | redução aproximada de 66% |
| Faithfulness RAGAS | 0,952 | N/A | 15 jobs falharam |
| Context precision RAGAS | 0,821 | N/A | 15 jobs falharam |
| Context recall RAGAS | 0,809 | N/A | 15 jobs falharam |

O run Maritaca terminou com código zero, mas os 45 jobs do judge receberam
`403 insufficient_funds`. Assim, `EXIT 0` prova apenas que o processo chegou ao
fim; não representa aprovação do portão RAGAS.

O teste via OpenRouter demonstrou compatibilidade funcional com
`google/gemini-2.5-flash`: 9 jobs referentes a 3 registros e 5 contextos foram
concluídos em 127,5 s. Ele não é diretamente comparável ao baseline, cujo judge
foi Maritaca `sabia-4`, e não estima com segurança o custo do dev completo, no
qual cada pergunta contém aproximadamente 30–32 contextos.

### 1.2 Regressões observadas

Das 15 perguntas do dev, 9 melhoraram na precisão numérica, 4 permaneceram
estáveis e 2 pioraram:

- `dev_demography_03`: `0,914 -> 0,750`;
- `dev_economy_01`: `0,850 -> 0,133`.

O caso do PIB é uma regressão funcional, não apenas uma limitação da métrica. A
pergunta solicita o primeiro trimestre de 2026 e o gabarito informa:

- `1,8%` contra o trimestre anterior;
- `1,3%` contra o mesmo trimestre de 2025;
- `0,6%` no acumulado de quatro trimestres.

A resposta candidata usou os dados mensais de abril de 2026 (`-1,1%`, `1,6%`,
`0,6%` e a decomposição setorial mensal) e os apresentou como se fossem do
primeiro trimestre. O contexto trimestral correto estava entre os trechos
recuperados. Isso aponta para falha de seleção/síntese.

Em `dev_demography_03`, a fonte-alvo de 2024 não foi recuperada e o modelo
recusou responder. A recusa foi segura, mas representa regressão de recuperação
em relação ao baseline.

### 1.3 Limitações descobertas na avaliação

- `RAG_LLM_REASONING_EFFORT` é aplicado hoje a todos os papéis criados por
  `make_llm()`, inclusive à síntese final, embora a documentação descreva a
  otimização como destinada aos passos mecânicos.
- O validador compara representações numéricas como strings. Por isso, por
  exemplo, `1,60%` não é equivalente a `1,6%`.
- Valores de células sob um cabeçalho `em %` não herdam essa unidade; a resposta
  com `%` pode ser rejeitada mesmo quando o valor está na tabela.
- Um número único pode ser aprovado sem compatibilidade entre período e
  indicador. Corrigir somente a formatação poderia validar um valor de abril
  usado incorretamente numa afirmação sobre o trimestre.
- `evaluate.py` chama `engine.answer()` diretamente. Ele não passa pelo fluxo de
  recuperação e pelos alertas de `query_service.py`; portanto, a ausência do
  alerta de sobreposição lexical não comprova aprovação pelo guardrail de
  produção.
- Os registros de geração só são salvos depois do judge. Uma interrupção durante
  o RAGAS perde respostas e contextos já produzidos.
- O resultado não falha diante de métricas `NaN` e o JSON é gravado como se o
  run tivesse concluído normalmente.
- A proveniência registra o commit, mas não o estado sujo do worktree nem as
  configurações efetivas do RAG. Além disso, o provider do judge é gravado como
  `maritaca` mesmo quando a base URL aponta para OpenRouter.

## 2. Objetivo

Entregar uma configuração rápida que preserve a correção factual e um processo
de avaliação que:

1. isole o esforço de raciocínio por papel;
2. valide números equivalentes sem perder período, indicador, unidade e sinal;
3. separe geração de respostas da etapa de judge;
4. produza artefatos reproduzíveis e falhe explicitamente quando o portão RAGAS
   não for executado com sucesso;
5. permita comparar baseline e candidato com o mesmo judge.

## 3. Decisões de arquitetura

### 3.1 Não commitar o pacote atual

Os testes unitários passam, mas a resposta errada do PIB invalida a premissa de
que a otimização preserva qualidade. O trabalho deve ser replanejado antes do
commit do item 1.

### 3.2 Separar raciocínio por papel

Substituir o controle global por configurações independentes:

- `RAG_LLM_REASONING_EFFORT`: síntese principal;
- `RAG_INTERP_REASONING_EFFORT`: interpretação, roteamento e consultas
  alternativas;
- `RAG_EXTRACT_REASONING_EFFORT`: extração de tabelas e séries temporais.

Variável ausente, vazia, `auto` ou `default` significa preservar o padrão do
provedor. Valores aceitos continuam sendo `none`, `minimal`, `low`, `medium` e
`high`.

O experimento deve escolher a configuração mais rápida que passe nos critérios
de correção; não deve assumir antecipadamente que `none` é seguro para nenhum
dos papéis.

### 3.3 Separar geração e julgamento

O resultado da geração deve existir antes da primeira chamada RAGAS. O judge
passa a consumir um artefato já salvo, permitindo:

- repetir somente o judge após recarga ou troca de provedor;
- julgar baseline e candidato sem regenerar respostas;
- preservar o trabalho quando um provider falhar ou o processo for interrompido;
- distinguir falha do RAG de falha do judge.

### 3.4 Judge final comparável

O portão final deve usar Maritaca `sabia-4`, o mesmo judge do baseline. OpenRouter
só pode substituir esse protocolo se baseline e candidato forem julgados
novamente com o mesmo endpoint, modelo, prompts e parâmetros.

## 4. Frentes de implementação

### Frente A — raciocínio por papel

**Owner:** `rag_core/llm.py` e os módulos `startup.py` das quatro engines.

#### Alterações

1. Tornar a resolução de `reasoning_effort` consciente do papel do LLM.
2. Garantir que síntese, interpretação e extração recebam apenas a configuração
   correspondente.
3. Rejeitar combinações ambíguas como `interp=True` e `extract=True` ao mesmo
   tempo, ou substituir os dois booleanos por um papel explícito.
4. Preservar o comportamento padrão quando nenhuma variável estiver definida.
5. Atualizar os testes de fábrica para inspecionar o corpo HTTP de cada papel.
6. Atualizar o README sem afirmar ganho de qualidade antes do novo benchmark.

#### Matriz experimental

Manter modelo, temperatura, corpus, seed, query fusion, reranker e limites de
retrieval constantes. Executar pelo menos duas repetições sem judge para cada
configuração:

| Configuração | Síntese | Interpretação | Extração |
|---|---|---|---|
| A — controle | padrão do provider | padrão do provider | padrão do provider |
| B — mecânica | padrão do provider | `none` | `none` |
| C — extração | padrão do provider | padrão do provider | `none` |
| D — atual | `none` | `none` | `none` |

Começar por `dev_economy_01` e `dev_demography_03`. Só ampliar para as 15
perguntas depois que uma configuração responder corretamente às duas.

#### Critérios de aceite

- `dev_economy_01` usa os números do primeiro trimestre, sem misturar abril;
- `dev_demography_03` volta a recuperar a fonte-alvo de 2024;
- nenhuma pergunta apresenta regressão factual nova;
- p50 e p95 permanecem abaixo do baseline de 24,35 s e 35,68 s;
- a suíte da fábrica prova que o valor de um papel não vaza para outro.

### Frente B — validação numérica semântica

**Owner:** `rag_core/numerical_validator.py` e seus testes.

#### Alterações

1. Canonicalizar valores com `Decimal`, de forma que `1,60` e `1,6` sejam
   equivalentes sem perder sinal.
2. Reconhecer unidade declarada no cabeçalho ou bloco de uma tabela, evitando
   exigir `%` em cada célula.
3. Tratar linguagem de magnitude negativa: `recuou 1,1%` pode corresponder a
   `-1,1%`, mas apenas quando o verbo e o contexto expressarem queda.
4. Exigir compatibilidade mínima de período e indicador. Remover a regra que
   aceita automaticamente um valor único com sobreposição contextual zero.
5. Não penalizar números apenas repetidos da pergunta em recusas seguras; marcar
   esses casos como não aplicáveis em vez de corretos ou incorretos.
6. Preservar proveniência: índice da fonte, trecho, período, indicador, unidade e
   motivo de aprovação/rejeição.

#### Casos obrigatórios de teste

- `1,60%` versus `1,6%`;
- `recuou 1,10%` versus `-1,1%`;
- célula `-2,8` sob cabeçalho `em %`;
- `1,6% em abril` não sustenta `1,6% no primeiro trimestre`;
- o ano `2024` numa recusa sobre dados ausentes não reduz a precisão;
- valores repetidos em períodos ou indicadores diferentes não são consumidos
  pela ocorrência errada.

#### Regra de comparação

Como essa frente altera o scorer, a nova precisão não pode ser comparada com os
números antigos. Recalcular baseline e candidato a partir dos artefatos salvos,
usando exatamente a mesma versão do validador.

### Frente C — endurecimento de `evaluate.py`

**Owner:** `evaluate.py` e testes novos do harness de avaliação.

#### Alterações

1. Dividir o fluxo em duas fases explícitas:
   - geração: respostas, referências, contextos, diagnósticos e latência;
   - julgamento: métricas RAGAS sobre um artefato de geração.
2. Salvar o checkpoint da geração de forma atômica antes de iniciar o judge.
3. Adicionar modos equivalentes a:
   - `--generation-only` ou `--skip-ragas`;
   - `--judge-only <artefato>`;
   - `--ids dev_economy_01,dev_demography_03` para testes focados.
4. Considerar o run reprovado quando uma métrica obrigatória não for finita.
5. Retornar código diferente de zero para falha do judge.
6. Gravar `null` e um estado explícito de falha, nunca `NaN` num JSON de sucesso.
7. Preservar erro HTTP, métrica, registro e tentativa no relatório de falhas sem
   expor chaves ou conteúdo sensível.
8. Registrar metadados suficientes para reprodução:
   - commit, dirty state e hash do diff;
   - hash do dataset;
   - modelos de síntese, interpretação e extração;
   - reasoning effort efetivo por papel;
   - query fusion, reranker e limites de retrieval;
   - provider, base URL sanitizada e modelo real do judge;
   - versões de Python, RAGAS, LlamaIndex e cliente OpenAI.
9. Derivar o provider real ou exigir `RAGAS_JUDGE_PROVIDER`, removendo o valor
   `maritaca` fixo.

#### Critérios de aceite

- uma falha 403 preserva o checkpoint de geração e termina com código não zero;
- um processo interrompido durante o judge pode ser retomado sem gerar respostas
  novamente;
- nenhum resultado com as três métricas RAGAS ausentes recebe status de sucesso;
- um run OpenRouter não é rotulado como Maritaca;
- baseline e candidato podem ser julgados a partir de arquivos independentes.

## 5. Ordem de execução

1. **Congelar evidência atual:** copiar os artefatos atuais para nomes imutáveis,
   registrando configuração e hash do worktree.
2. **Implementar a Frente A:** separar raciocínio por papel e adicionar testes.
3. **Executar a matriz focada:** PIB e mortalidade, sem RAGAS.
4. **Selecionar a configuração:** escolher a mais rápida que preserve correção.
5. **Executar o dev completo sem judge:** confirmar 15/15 e latências.
6. **Implementar a Frente B:** corrigir o scorer e recalcular baseline/candidato.
7. **Implementar a Frente C:** checkpoint, judge-only, códigos de saída e
   proveniência.
8. **Executar o portão final:** recarregar a Maritaca e julgar o par válido com
   `sabia-4`.
9. **Revisar o diff completo:** código, testes, documentação e artefatos antes do
   commit.

As três frentes devem ser commits separados para permitir bisect e atribuição de
efeito. A Frente A decide a viabilidade do item 1; as Frentes B e C tornam a
medição confiável.

## 6. Plano de validação

### 6.1 Testes locais

```bash
python -m pytest -p no:cacheprovider rag_principal/tests rag_orchestrator/tests -q
git diff --check
```

Adicionar testes focados para:

- roteamento de modelo e raciocínio por papel;
- serialização estrita do resultado;
- saída não zero quando o judge falhar;
- retomada `judge-only` a partir de checkpoint;
- provider correto nos metadados;
- todos os casos semânticos da Frente B.

### 6.2 Avaliação focada após os novos modos

```bash
python evaluate.py --split dev --ids dev_economy_01,dev_demography_03 --generation-only --seed 42
```

Executar esse comando para cada linha da matriz experimental, mantendo todas as
demais variáveis fixas.

### 6.3 Avaliação completa sem judge

```bash
python evaluate.py --split dev --generation-only --seed 42
```

O artefato resultante deve ser imutável e identificado como candidato ao portão.

### 6.4 Portão RAGAS final

```bash
python evaluate.py --judge-only evaluation/runs/<candidato>/generation.json
```

Executar com Maritaca `sabia-4`. Se OpenRouter for adotado, repetir o mesmo
comando para o artefato do baseline e para o candidato usando exatamente o mesmo
judge.

## 7. Matriz de aceite final

| Dimensão | Critério |
|---|---|
| Execução | 15/15 perguntas concluídas, sem erro do pipeline |
| PIB | períodos e valores do 1º trimestre corretos; nenhum dado mensal rotulado como trimestral |
| Mortalidade | fonte e números de 2024 recuperados ou regressão explicitamente resolvida |
| Numérica | candidato não inferior ao baseline recalculado com o mesmo scorer |
| Regressão individual | nenhuma queda material escondida pela média agregada |
| Latência | p50 < 24,35 s e p95 < 35,68 s; registrar também a variação entre repetições |
| RAGAS | 45/45 resultados finitos com o mesmo judge do baseline |
| Faithfulness | não regredir mais que 0,02 absoluto frente a 0,952 |
| Context precision | não regredir mais que 0,02 absoluto frente a 0,821 |
| Context recall | não regredir mais que 0,02 absoluto frente a 0,809 |
| Proveniência | commit/diff, dataset, configuração RAG e judge completamente identificáveis |
| Falha externa | status de falha e código não zero; checkpoint de geração preservado |

As tolerâncias RAGAS de `0,02` são uma proposta operacional. Devem ser mantidas
iguais para baseline e candidato e registradas antes da rodada paga.

## 8. Fora de escopo

- trocar permanentemente o provider de geração;
- reindexar o corpus sem evidência de defeito no índice;
- alterar prompts de resposta além do necessário para corrigir período e
  indicador;
- usar resultados de judges diferentes numa comparação direta;
- implantar, publicar ou alterar credenciais e saldo de provedores;
- tratar o teste OpenRouter de três registros como substituto do gate completo.

## 9. Resultado esperado

Ao final, o repositório terá:

1. controle de raciocínio isolado por papel;
2. uma configuração de baixa latência sem a regressão do PIB;
3. validação numérica capaz de distinguir equivalência de formato de identidade
   semântica;
4. geração reaproveitável e julgamento retomável;
5. códigos de saída e artefatos que não confundem falha do judge com sucesso;
6. comparação final válida contra o baseline Maritaca `sabia-4`.
