# TypeSafe Jev (System One) — referência do projeto

Documento de apoio para qualquer trabalho que envolva decisões semânticas neste
repositório. Condensa o "Project Knowledge Resource" de TypeSafe Jev (gist
`pjburnhill/adf8d28efcad9df037bfdece178ef965`, conhecimento de 16/09/2026) e
descreve como o paradigma está aplicado aqui. Preços, limites e versões de modelo
mudam: confirme em <https://docs.typesafe.ai> antes de assumir valores.

## O modelo mental

Jev não é um LLM. A interface é:

> **estado (texto ou JSON) → decisões tipadas com probabilidade.**

Não há geração de texto: as perguntas e o espaço de respostas são definidos antes;
o modelo devolve a opção vencedora, a distribuição de probabilidade por opção e
`confidence`. O software continua dono do fluxo, das regras e dos efeitos.

    Código calcula. Jev julga. LLMs raciocinam/criam. Humanos definem objetivos.

O teste rápido de adequação: *"dado este estado, me diga X"*, com X sendo um
`Choice`, um `Score` ou uma probabilidade. Se um especialista com o contexto em
mãos responderia em cinco segundos, sem pesquisar, sem encadear raciocínio e sem
criar conteúdo, a tarefa é "formato Jev". Julgamentos complexos devem ser
decompostos em várias perguntas estreitas e recompostos em código.

## As três primitivas

| Primitiva | Pergunta | Retorno |
|---|---|---|
| **Noul** | Isto é verdade? | `noul` (0–1); sem `confidence` separada |
| **Choice** | Qual destas opções? | `choice`, `probabilities`, `confidence` |
| **Score** | Onde isto está nesta escala? | `score`, `legend`, `probabilities`, `confidence` |

- `Choice` aceita até 255 opções; inclua `other`/`none of the above` quando a
  taxonomia pode não cobrir toda entrada.
- `Score` aceita de 2 a 10 níveis, descritos por situações concretas ("serviço
  indisponível sem alternativa"), nunca por adjetivos vagos ("alto").
- Noul `0.5` é empate, não "média": para medir grau, use `Score`.
- Todas as perguntas de um pedido olham o **mesmo estado**, são avaliadas **em
  paralelo** e de forma **independente** (a resposta de uma não vira contexto da
  outra). Peça tudo junto; dependência real só existe quando o código precisa da
  primeira resposta para montar a segunda requisição.

## Probabilidade e confiança são saída de primeira classe

Os modelos são treinados por RLCD (calibração): grupos de previsões com
probabilidade mais alta acertam mais. Mas **calibração não é correção** — uma
previsão confiante pode estar errada. Por isso:

- `WHAT` o modelo pensa e `HOW SURE` ele está são eixos separados;
- limiares devem ser validados com dados do domínio;
- ação de risco exige evidência mais forte, e o caso incerto escala para revisão.

"Zero alucinação" significa apenas que a resposta é **impossível fora do schema**
(a distribuição só cobre as opções enviadas). Erro de julgamento semântico
continua possível e precisa ser medido — schema hallucination ≠ zero erros.

## Configuração (variáveis de ambiente)

| Variável | Padrão | Efeito |
|---|---|---|
| `TYPESAFE_API_KEY` | — | chave oficial (o SDK da TypeSafe usa o mesmo nome); `JEV_API_KEY` é aceito como alias e o oficial prevalece |
| `RAG_JEV_ENABLED` | `1` | `0`/`false`/`no`/`off` desliga a Jev e devolve o caminho LLM em todos os pontos |
| `RAG_JEV_THRESHOLD` | `0.5` | probabilidade mínima do Noul para a fonte entrar (é também o que converte Noul em decisão booleana) |
| `RAG_JEV_TIMEOUT` | `20` s | prazo por requisição (faixa aceita: 2–120) |
| `RAG_JEV_MAX_ATTEMPTS` | `2` | tentativas por chamada (faixa: 1–4); só erros repetíveis (408/409/425/429/5xx) são retentados |
| `TYPESAFE_BASE_URL` | `https://api.typesafe.ai` | endpoint System One |
| `TYPESAFE_MODEL` | `jev-latest` | modelo System One |
| `RAG_USE_GRAPH` | `0` | o grafo é a única fonte que a Jev pode escolher e o engine ignora quando está desligado |

Sem chave — ou com `RAG_JEV_ENABLED=0` — `jev_enabled()` é falso, nenhum ponto cria cliente
e tudo segue no LLM, sem chamada de rede.

## Onde isso está aplicado neste repositório

| Componente | Primitiva | Arquivo |
|---|---|---|
| Classificação da consulta (10 perguntas, 1 requisição) | 8 `Choice` + 2 `Noul` | `rag_orchestrator/src/jev_analyzer.py` |
| Seleção de fontes do rag_principal (tabelas/séries/grafo) | 3 `Noul` | `rag_principal/src/jev_interpreter.py` |
| RETRIEVE? do Self-RAG | `Noul` | `rag_selfrag/src/self_rag_engine.py` |
| ISREL do Self-RAG (1 `Noul` por trecho, fan-out) | `Noul` | `rag_selfrag/src/self_rag_engine.py` |
| ISSUP do Self-RAG | `Choice` `full\|partial\|none` | `rag_selfrag/src/self_rag_engine.py` |
| Fronteira HTTP, tipos e limiar | — | `rag_core/jev.py` |

Decisões de projeto que valem preservar ao estender:

1. **Fan-out especulativo:** todas as perguntas que compartilham o estado vão em
   uma única requisição; perguntar algo que talvez não importe custa quase nada
   em latência (só tokens da pergunta).
2. **Composição no código:** o limiar vive em `RAG_JEV_THRESHOLD`; pesos e
   políticas ficam no repositório, não em prompt.
3. **Confiança conservadora no roteamento:** a confiança agregada é o menor grau
   de decisão entre as dimensões **que o roteador consome** (`query_type`,
   `priority`, `retrieval_need`, `complexity`, `in_scope`) — dimensão que não
   muda rota nenhuma não pode derrubar o portão de recusa.
4. **Fallback obrigatório:** toda chamada pode falhar sem derrubar a consulta —
   o caminho LLM original continua no lugar.
5. **Sem texto livre:** campos como `entities`/`period` não vêm da Jev; se algum
   dia forem necessários para o roteamento, precisam de extração determinística
   (regex/dicionário) ou de um LLM, não de um Noul disfarçado.

## Lições medidas (o que já foi testado com dados reais)

Medições feitas com as 10 perguntas do `evaluation/golden_dataset.json` + 5 casos
difíceis, sempre pela API real:

1. **`Score` espalha a massa e sai menos decisivo que `Choice`.** `retrieval_need`
   e `complexity` foram convertidas para escala e a decisão **piorou**: para a
   mesma pergunta, `complexity` veio `{0: 0.66, 1: 0.17, 2: 0.17}` como `Score`
   contra `{baixa: 0.90, media: 0.07, alta: 0.05}` como `Choice`; margem média
   0.24 contra 0.46. Escala serve para expressar grau, não para tornar uma
   crença categórica mais firme. As duas seguem como `Choice`.
2. **Não pergunte duas vezes o mesmo fato.** `needs_multi_hop` (grau de decisão
   médio 0.40) duplicava a opção `multi_hop` de `query_type` e as duas
   discordavam. Agora é derivada de `query_type`.
3. **Dimensão sem consumidor não entra na conta de confiança.** `technical_terms`
   não é lido por ninguém e prendia o mínimo da confiança em 3 das 15 perguntas.
4. **A hesitação é intrínseca em `retrieval_need`/`complexity`** (grau de decisão
   médio 0.48/0.54 nas duas formulações): o modelo não sabe se a pergunta precisa
   de busca lexical ou híbrida, porque isso depende do índice, não da pergunta.
   São hoje as dimensões que prendem a confiança e o alvo de uma próxima medição.

## Referências

- Documentação: <https://docs.typesafe.ai> (`/llms.txt` lista todas as páginas)
- Padrões úteis: `patterns/intent-routing`, `patterns/fan-out`,
  `patterns/confidence-routing`, `patterns/composite-scoring`
- Cookbooks próximos deste domínio: `cookbooks/classifying_rag_passages`,
  `cookbooks/citation_check`, `cookbooks/llm_guardrails`
- Verificação local: `python scripts/jev_smoke.py`; testes:
  `rag_principal/tests/test_jev_client.py`, `rag_principal/tests/test_jev_interpreter.py`,
  `rag_principal/tests/test_selfrag_jev.py`, `rag_orchestrator/tests/test_jev_analyzer.py`
