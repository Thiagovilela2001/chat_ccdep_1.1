# Nadia — Consulta Inteligente aos Dados de São Paulo
## Apresentação de conceitos para plateia leiga

**Estrutura:** 7 blocos · 17 slides · ~15 minutos
**Regra de ouro da apresentação:** a plateia não precisa entender de tecnologia. Ela precisa entender **um problema, uma ideia e um limite honesto**.

**Como usar este roteiro:** cada slide traz os *bullets que vão na tela* e uma *fala sugerida* (o que você diz em voz alta). Os blocos estão marcados com ⏱ para você controlar o tempo.

| Bloco | Slides | Tema |
|---|---|---|
| 1 | 1–2 | A ideia do projeto |
| 2 | 3–5 | O que é um RAG |
| 3 | 6–8 | Banco de dados vetorial |
| 4 | 9–10 | Como a pergunta se alinha ao acervo |
| 5 | 11–12 | Tipos de RAG |
| 6 | 13–14 | Limitações |
| 7 | 15–17 | O projeto do chatbot |

---

# BLOCO 1 — A IDEIA DO PROJETO ⏱ 2 min

## Slide 1 — A informação existe. Achar é que é o problema.

**Na tela:**
- Os indicadores de São Paulo estão publicados em **boletins oficiais**, edições trimestrais, 2020–2025
- Para achar **um número**, é preciso: saber em qual boletim está → abrir o PDF → achar a tabela certa → anotar a página
- Conferir manualmente 22 edições, página por página, é lento e repetitivo

**🗣 Fala sugerida:**
> "Toda vez que alguém precisa de um dado de São Paulo — quanto variou o PIB, qual a taxa de desocupação — esse número existe, é público e é oficial. O problema nunca foi a informação existir. O problema é **achar**. Imagine que você queira um número específico e ele esteja em um de 22 boletins, cada um com dezenas de páginas. Você precisa descobrir qual boletim, abrir, achar a tabela e ainda anotar de onde tirou. Isso é fácil de fazer uma vez e insuportável de fazer todo dia. E, na estatística, o número sozinho não serve: um número sem o período e sem a fonte é um número que ninguém pode usar."

**💡 Analogia guardada:** *um cartório com 22 livros de registro — a informação está lá dentro, mas você não sabe em qual livro nem em qual página.*

---

## Slide 2 — A proposta: uma ferramenta de busca para dados não estruturados

**Na tela:**
- **Dado não estruturado** = informação que não está em planilha organizada. É texto corrido, tabela dentro de PDF, página de boletim
- A proposta: uma ferramenta que recebe **pergunta em português** e responde a partir dos documentos oficiais
- **Regra de ouro:** a ferramenta **não substitui o dado original** — ela encontra a evidência e aponta a fonte
- Sempre com três coisas juntas: **valor + período + fonte**

**🗣 Fala sugerida:**
> "Quando a gente fala em dado não estruturado, é simples: é a informação que não está numa planilha organizada. É o texto de análise, é a tabela impressa dentro de um PDF, é a página de um boletim. Computador adora planilha e sofre com PDF — e é justamente aí que mora quase todo o conhecimento público. A nossa proposta é uma ferramenta de busca para esse tipo de material: você pergunta em português e ela vai atrás da resposta nos documentos oficiais. Com uma regra que a gente não abre mão: **a ferramenta não inventa e não substitui o dado original**. Ela encontra a evidência e mostra a fonte. E ela sempre devolve três coisas juntas — o valor, o período e a página de onde veio."

**💡 Analogia guardada:** *não é um tradutor que reescreve o livro; é um bibliotecário que sabe exatamente onde está o trecho e te leva até a página.*

---

# BLOCO 2 — O QUE É UM RAG ⏱ 3 min

## Slide 3 — O que acontece se você só perguntar para a inteligência artificial

**Na tela:**
- Um modelo de linguagem responde **de memória** — aquilo que viu no treinamento
- Três problemas:
  - o dado pode estar **desatualizado** ou simplesmente **não existir** na memória dele
  - ele **não sabe dizer a página** — porque não tirou de página nenhuma
  - quando não sabe, ele **inventa**. E a invenção sai com a mesma cara de confiança da verdade. Isso se chama **alucinação**
- **Num texto comum, um erro de escrita se percebe. Num número estatístico, um número inventado parece tão correto quanto o verdadeiro.**

**🗣 Fala sugerida:**
> "Todo mundo já usou um chat de inteligência artificial. Ele responde com muita segurança. Só que ele responde de memória — do que ele viu durante o treinamento. E aí vêm três problemas. Primeiro: o dado pode estar velho, ou simplesmente não existir na memória dele. Segundo: ele não tem como dizer de qual página tirou, porque ele não tirou de página nenhuma. Terceiro, e o mais perigoso: quando ele não sabe, ele inventa. E a invenção vem escrita com a mesma confiança da verdade. Isso tem nome: **alucinação**. Agora pensa no nosso caso. Se o modelo 'lembrar' que a desocupação foi 7,4% e o boletim diz 7,1%, **ninguém percebe**. Num texto, um erro de português salta aos olhos. Num número estatístico, o erro passa batido."

---

## Slide 4 — RAG, em uma frase: procurar antes de responder

**Na tela:**
- **RAG** = *Retrieval-Augmented Generation* = **geração aumentada por recuperação**
- **Primeiro o sistema procura a informação. Depois ele escreve a resposta.**
- Sem RAG: o modelo responde **de memória** — e não mostra onde consultou
- Com RAG: o modelo responde **a partir de trechos do acervo** — e mostra a fonte
- Atualizar o conteúdo **não exige retreinar o modelo**: basta reindexar o documento novo

**🗣 Fala sugerida:**
> "A solução se chama RAG — geração aumentada por recuperação. Mas o nome bonito esconde uma ideia simples: **procurar antes de responder**. Sem RAG é uma prova sem consulta: o aluno responde do que lembra e, se não lembra, chuta. Com RAG é uma prova **com consulta e com citação obrigatória**: ele pode abrir o material, mas tem que escrever a resposta **e indicar a página**. Se não achar, o certo é dizer 'não encontrei' — não chutar. E tem um ganho prático enorme: quando chega um boletim novo, a gente não treina nada de novo. A gente simplesmente coloca o documento no acervo. O modelo é o mesmo."

**💡 Analogia:** *prova sem consulta × prova com consulta e citação obrigatória.*

---

## Slide 5 — Como funciona: quatro etapas

**Na tela:**

| | Etapa | O que acontece |
|---|---|---|
| 1 | **Preparação** | Ler os documentos, separar texto de tabela, dividir em trechos com identificação |
| 2 | **Indexação** | Cada trecho ganha um "endereço de significado" e entra num índice |
| 3 | **Recuperação** | Diante da pergunta, o sistema busca os trechos mais relevantes |
| 4 | **Geração** | O modelo escreve a resposta **usando apenas** os trechos recuperados, citando a fonte |

- **A etapa 4 é onde tudo se decide:** o modelo não recebe a pergunta sozinha — recebe **a pergunta + os trechos selecionados**
- **A busca é feita em trechos pequenos, não em documentos inteiros**

**🗣 Fala sugerida:**
> "Todo RAG do mundo, por mais sofisticado que seja, tem essas quatro etapas. Prepara o documento, indexa, recupera e gera. Eu quero que vocês guardem duas observações. A primeira: a etapa quatro é onde tudo se decide, porque o modelo **não recebe a pergunta sozinha**. Ele recebe a pergunta **mais os trechos** que o sistema encontrou. É por isso que dá para exigir que ele cite a fonte e conferir o número depois. Se a etapa três trouxe o trecho errado, nenhum modelo bom salva a resposta. A segunda: a busca é feita em **pedaços pequenos** de texto, nunca no documento inteiro. O sistema não entrega 'o boletim de 2024' para o modelo — ele entrega os pedaços que importam."

---

# BLOCO 3 — BANCO DE DADOS VETORIAL ⏱ 3 min

## Slide 6 — Como a máquina compara significados

**Na tela:**
- Um computador não compara **palavras**, compara **números**
- **Embedding** = uma lista de números que representa o **significado** de um texto
- O truque: textos com **significado parecido** recebem listas de números **parecidas**
- Por isso "desemprego" e "desocupação" ficam **perto**, mesmo sem nenhuma letra em comum

**🗣 Fala sugerida:**
> "Aqui está o problema técnico central. A sua pergunta usa a palavra 'desemprego'. O boletim oficial escreve 'taxa de desocupação'. Nenhuma palavra é igual. Uma busca comum, por palavra, devolveria **zero resultados**. Então a gente precisa ensinar o computador a comparar **significado**, e não letra. Para isso existe o **embedding**: uma lista de números que representa o significado de um texto. O detalhe genial é este: textos que falam da mesma coisa recebem listas de números parecidas. Então 'desemprego' e 'desocupação' acabam ficando **numericamente próximos**, mesmo sem compartilhar uma única letra."

**💡 Analogia:** *uma cidade dos assuntos. Cada trecho recebe um endereço nessa cidade. Textos do mesmo tema moram na mesma rua — têm CEP parecido.*

---

## Slide 7 — O banco de dados vetorial: um armário feito para isso

**Na tela:**
- **Banco de dados vetorial** = armário que guarda esses "endereços de significado" e responde rápido: *quais são os mais próximos?*
- No projeto: **ChromaDB**, rodando **localmente**
- **Fica salvo em disco** → o acervo não é reprocessado a cada pergunta
- O índice do projeto tem cerca de **57 mil trechos** guardados
- Cada trecho guardado é uma **ficha com 4 partes:**

| Parte | O que é |
|---|---|
| **Texto** | o conteúdo do trecho |
| **Vetor** | o endereço de significado |
| **Etiquetas** | arquivo de origem, página e tipo |
| **Identificador** | um número único da ficha |

**🗣 Fala sugerida:**
> "Um banco de dados vetorial é um armário construído para guardar esses endereços de significado e responder rapidinho à pergunta: 'quais são os mais próximos do que eu quero?'. No nosso projeto ele é o ChromaDB, rodando na própria máquina — ou seja, o acervo não sai da nossa infraestrutura. Ele fica salvo em disco, então o material não é reprocessado a cada pergunta. São cerca de 57 mil trechos guardados. E cada trecho não é só o texto: é uma **ficha** com quatro partes — o texto, o endereço de significado, as etiquetas de origem e um identificador."

**💡 Analogia:** *um fichário com 57 mil fichas, ordenadas **por assunto parecido** em vez de ordem alfabética. Cada ficha tem uma etiqueta colada dizendo de qual documento e de qual página ela veio.*

---

## Slide 8 — Como os dados ficam organizados

**Na tela:**
- Cada ficha carrega etiquetas: **arquivo**, **página**, **tipo** (`texto` ou `tabela`)
- **Tabela não é texto corrido.** Se a tabela for cortada como parágrafo, o valor se descola do cabeçalho e do período
- Regra do projeto: tabela **pequena** entra inteira; tabela **longa** (série histórica) entra **linha por linha**
- Cada linha é guardada **com o nome da coluna junto**: `Ano: 2023 · Indústria: 2,1 · Fonte: boletim.pdf`
- **Série temporal é uma tabela que sabe que é uma série** (tem a etiqueta de granularidade)

**🗣 Fala sugerida:**
> "Aqui está uma decisão que parece detalhe e mudou a qualidade do sistema. Tabela não é texto corrido. Se você cortar uma tabela como se fosse parágrafo, o valor **se descola do cabeçalho e do período** — e um '3,4' solto não significa nada. Então o projeto trata diferente: tabela pequena entra inteira, para não perder o cabeçalho; tabela longa, que é uma série histórica, entra linha por linha, para dar para recuperar ano a ano. E cada linha é guardada **com o nome da coluna junto**. Uma série temporal, aliás, não é um terceiro tipo de dado: **é uma tabela que sabe que é uma série**, porque carrega uma etiqueta dizendo a granularidade — mensal, trimestral, anual."

**⚠️ Se perguntarem "por que a tabela é guardada como texto?":** *porque o modelo de linguagem lê texto. O texto estruturado com o nome da coluna preserva o significado do dado sem depender de o modelo entender a geometria da tabela.*

---

# BLOCO 4 — COMO A PERGUNTA SE ALINHA AO ACERVO ⏱ 3 min

## Slide 9 — O caminho de uma pergunta

**Na tela:**
1. **Interpretar** — o sistema identifica indicador, local e período, e reescreve a pergunta no vocabulário do acervo
2. **Transformar a pergunta em vetor** — pelo **mesmo** modelo que leu os documentos
3. **Filtrar** — *"procure apenas entre as fichas do tipo texto"* (ou tabela)
4. **Buscar** — traz os ~80 candidatos mais próximos
5. **Busca por palavra-chave em paralelo** — pega siglas e termos exatos que o significado sozinho não pega
6. **Combinar as duas listas** — pelos **maiores colocados nas duas**, não por nota
7. **Reordenar e enxugar** — sobra o essencial
8. **Diversificar** — no máximo 3 trechos por documento, para a resposta enxergar vários boletins

**🗣 Fala sugerida:**
> "Vamos seguir o caminho de uma pergunta de verdade. Ela não entra no banco direto. Primeiro o sistema **interpreta**: identifica o indicador, o local, o período, e reescreve a pergunta no vocabulário do boletim. Depois essa pergunta vira vetor, pelo mesmo modelo que leu os documentos — isso é essencial, os dois precisam falar a mesma língua. Aí ele **filtra** por tipo e busca os candidatos mais próximos. Em paralelo, roda uma busca por **palavra-chave**, que é ótima para siglas e termos exatos. No fim, as duas listas são combinadas — e o critério não é a nota, é **quem apareceu bem nas duas**. É o que a gente chama de dois bibliotecários: um busca por assunto, outro busca pela palavra exata. Em vez de discutir qual régua é melhor, o sistema promove quem os dois colocaram no topo."

**💡 A frase que você deve saber dizer:** *"A pergunta não vai direto ao banco: ela é interpretada, filtrada, buscada por dois critérios diferentes e só então combinada."*

---

## Slide 10 — O encontro: semelhança × igualdade, e o carimbo da fonte

**Na tela:**
- **Embedding é buscado por semelhança.** "Parecido com isto"
- **Etiqueta é buscada por igualdade.** "Que tenha exatamente esta marca"
- A etiqueta **não participa** do cálculo de semelhança — ela **restringe** a busca e **rotula** o resultado
- **O modelo nunca vê o PDF.** Ele recebe o trecho com a fonte carimbada na frente:

```
[Fonte: boletim_1tri2024.pdf, p. 37]
A taxa de desocupação no 1º trimestre de 2024 foi de 7,1%...
```

- **É por isso que a resposta consegue citar a página** — o carimbo chegou junto com o recorte

**🗣 Fala sugerida:**
> "Este é o slide mais importante da apresentação. Existem **duas buscas diferentes** acontecendo. A busca por **semelhança** responde 'o que é parecido com isso?' — e é o vetor que faz. A busca por **igualdade** responde 'o que tem exatamente esta marca?' — e é a etiqueta que faz. As etiquetas não entram no cálculo de semelhança: elas **restringem** o universo e depois **rotulam** o resultado. E aí vem o segredo da citação. O modelo de linguagem **nunca viu o PDF**. Ele recebe o trecho **dentro de um envelope com a fonte carimbada do lado de fora**. Quando ele escreve 'segundo o boletim tal, página 37', ele está lendo o carimbo — porque o carimbo chegou junto com o recorte. É isso que transforma uma resposta de chat em uma resposta auditável."

**💡 Analogia:** *o redator recebe recortes de jornal dentro de envelopes. Cada envelope tem a fonte carimbada. Ele escreve citando o carimbo — porque o carimbo chegou junto.*

---

# BLOCO 5 — TIPOS DE RAG ⏱ 2 min

## Slide 11 — Quatro tipos, e onde eles diferem

**Na tela:**

| Variante | Como recupera | Indicada para | Limite |
|---|---|---|---|
| **Simples** | Busca por semelhança, uma vez | Pergunta direta, resposta em um trecho só | Não distingue texto de tabela nem de série |
| **Avançado** | Semelhança **+** palavra-chave, com reordenação | Termos técnicos, siglas, vocabulário que não bate | Exige manter dois índices e uma etapa a mais |
| **Modular** | **Roteia** a pergunta: escolhe qual recuperador consultar | Pergunta que precisa de tabela ou de série temporal | Cada rota exige tratamento e manutenção próprios |
| **Agêntico** | O **próprio modelo decide**, em passos, quais ferramentas acionar | Investigação em várias etapas, com hipóteses intermediárias | Mais chamadas, maior tempo de resposta, menos previsibilidade |

- **A fronteira entre os tipos está em duas decisões: como recuperar e quem decide quando parar de buscar**

**🗣 Fala sugerida:**
> "Existem vários tipos de RAG, e a diferença entre eles cabe em duas perguntas. Primeira: **como** eu recupero? Por significado só, ou por significado **mais** palavra-chave? Segunda: **quem decide quando parar de buscar?** Se é o sistema, numa passada só, é o RAG simples ou avançado. Se é o próprio modelo, que decide buscar de novo, é o agêntico. E tem o modular, que é o mais interessante para nós: em vez de ter uma busca só, ele **escolhe** qual recuperador usar conforme o tipo de pergunta. Se a pergunta é sobre número exato, ele chama o caminho de tabela. Se é sobre evolução no tempo, chama o caminho de série. Repare que o modular não é melhor por ser mais complexo — é melhor porque o nosso acervo tem **formatos diferentes**, e formatos diferentes pedem caminhos diferentes."

---

## Slide 12 — Onde o nosso projeto está (e o status honesto de cada estratégia)

**Na tela:**
- **Em uso:** RAG **modular** com recuperação **avançada** dentro de cada rota (texto, tabela, série)
- **Em uso:** uma camada de decisão que escolhe o caminho — o **Meta-RAG**
- **Status real das estratégias:**

| Estratégia | Status |
|---|---|
| Principal (híbrida + tabelas + séries) | **Implementada e em uso** |
| Modo agêntico | **Implementado, sem acesso pelo serviço** |
| RAPTOR (índice hierárquico com resumos) | **Estudado, não avaliado** |
| Self-RAG (verifica se a resposta tem suporte) | **Estudado, não avaliado** |

**🗣 Fala sugerida:**
> "Então, respondendo direto: o nosso sistema é um **RAG modular com recuperação avançada** em cada rota. E existe uma camada por cima que escolhe o caminho, que a gente chama de Meta-RAG. Agora, eu quero ser preciso sobre o que está pronto e o que não está, porque isso é uma escolha de honestidade do trabalho. A estratégia principal está implementada e em uso. O modo agêntico está implementado, mas só acessível por linha de comando — ainda não pelo serviço. E RAPTOR e Self-RAG, neste estágio, são **estudo**: a ideia está descrita e há protótipo em laboratório, mas não foram avaliados nem entraram em produção. Eu prefiro dizer isso aqui do que deixar vocês imaginando que temos quatro sistemas rodando."

**⚠️ Ponto crítico:** os seus slides atuais dizem "estudada, não implementada" para RAPTOR e Self-RAG, mas o repositório tem as pastas `rag_raptor/` e `rag_selfrag/` com código, e o relatório parcial lista as quatro como implementadas. **Decida a frase antes da apresentação e não mude no meio dela.** A versão acima ("estudado, com protótipo, não avaliado nem exposto no serviço") é defensável nas duas leituras.

---

# BLOCO 6 — LIMITAÇÕES ⏱ 2 min

## Slide 13 — Onde o sistema sofre

**Na tela:**
- **A tabela não sobrevive à extração automática.** PDFs estatísticos misturam texto e tabela na mesma página
- **O número sem período e sem unidade é um número errado** do ponto de vista estatístico
- **O vocabulário da pergunta não é o do boletim.** O usuário escreve "desemprego", o documento escreve "desocupação"
- **Inconsistências de nomenclatura no acervo.** Há arquivos com o período no nome e outros sem — um deles sem ano
- **Tempo de resposta alto.** As respostas medidas na demonstração levaram de **40 a 46 segundos**
- **Nenhum desses gargalos é de instalação: todos vêm da natureza do acervo estatístico**

**🗣 Fala sugerida:**
> "Agora a parte que eu considero mais importante de qualquer apresentação: o que não funciona bem. Primeiro: **a tabela não sobrevive sozinha à extração automática**. Boletim estatístico mistura texto e tabela na mesma página, e nem sempre a extração acerta. Segundo: um número sem período, sem unidade e sem recorte geográfico é um número **errado** do ponto de vista estatístico — e a gente tem que garantir que isso não se perca no caminho. Terceiro: o vocabulário do usuário não é o vocabulário do boletim. Quarto: o próprio acervo tem inconsistências — tem arquivo com o período no nome e arquivo sem, um deles sem o ano. E quinto: **o tempo de resposta é alto, de 40 a 46 segundos**. Nenhum desses problemas vem de instalação ou de configuração. Todos vêm da natureza do material — e é isso que os torna interessantes como objeto de pesquisa."

---

## Slide 14 — O risco central e os limites honestos do sistema

**Na tela:**
- **O risco central: número plausível e errado.** Em estatística, o número inventado parece tão correto quanto o verdadeiro
- Quatro defesas construídas:
  1. o prompt **proíbe inventar** e exige citar arquivo e página
  2. cada trecho carrega a **origem desde a preparação**
  3. os números da resposta são **conferidos** contra os números do trecho recuperado
  4. perguntas fora do escopo **podem ser recusadas**
- **Limite honesto:** a conferência automática **não é uma auditoria**. A revisão humana da fonte continua necessária
- **Limite de alcance:** a ferramenta responde **apenas** sobre o acervo indexado. Fora dele, o comportamento correto é dizer que não sabe

**🗣 Fala sugerida:**
> "O risco central do projeto tem nome: **número plausível e errado**. Num texto, um erro de redação você percebe. Num número estatístico, você não percebe. Então a gente construiu quatro defesas: o modelo é instruído a não inventar e a citar a fonte; cada trecho carrega a origem desde a preparação; os números da resposta são conferidos contra os números do trecho recuperado; e perguntas fora do escopo podem ser recusadas. Mas eu quero deixar dois limites muito claros. **Primeiro:** a conferência automática **não é uma auditoria** — ela compara números, ela não julga se a interpretação está certa. A revisão humana da fonte continua necessária. **Segundo:** o alcance da ferramenta é o acervo indexado. Fora dele, o comportamento correto é dizer 'não consta nos documentos fornecidos'. Um sistema que sabe dizer que não sabe vale mais do que um que sempre responde."

---

# BLOCO 7 — O PROJETO DO CHATBOT ⏱ 3 min

## Slide 15 — O que é a Nadia, e para quem

**Na tela:**
- **Nadia** é a assistente de consulta documental: recebe perguntas em português e responde a partir dos documentos oficiais, indicando **fonte e página**
- **Para quem:** analistas e gestores públicos · pesquisadores e estudantes · imprensa e cidadão
- **Sobre o quê:** boletins de conjuntura do Seade, ~22 edições, 2020–2025 · indicadores econômicos e de trabalho
- **Sempre com fonte:** a resposta vem acompanhada do arquivo e da página
- **Os três tipos de pergunta que ela atende:**
  - **Consulta pontual** — "qual foi a variação do PIB no 1º tri de 2024?"
  - **Evolução no tempo** — "como evoluiu a desocupação ao longo de 2024?"
  - **Análise comparada** — "como São Paulo se compara ao Brasil?"

**🗣 Fala sugerida:**
> "A Nadia é a assistente de consulta documental dos dados de São Paulo. Ela recebe uma pergunta em português e responde a partir dos documentos oficiais, sempre indicando a fonte e a página. Ela serve a três públicos: quem trabalha com política pública e precisa de consulta rápida; pesquisadores e estudantes, que ganham um ponto de partida com a referência exata; e imprensa e cidadão, que passam a acessar o dado oficial sabendo de onde ele veio. E ela atende três tipos de pergunta, que é importante separar porque **cada tipo percorre um caminho diferente dentro do sistema**: uma consulta pontual, um valor específico; uma evolução no tempo, que precisa juntar uma série de vários boletins; e uma análise comparada, que precisa buscar dois conjuntos de dados e confrontá-los."

**💡 Conexão com o slide 8:** *é exatamente por existirem esses três tipos de pergunta que o projeto separa texto, tabela e série temporal.*

---

## Slide 16 — Como o aplicativo funciona

**Na tela:**
- **Interface** — aplicação web: pergunta, histórico, resposta com citação, botão "ver evidências", exportar
- **Serviço** — camada em Python que interpreta a pergunta, aciona os recuperadores e monta a resposta
- **Índice e modelo** — acervo indexado na máquina local · **modelo de linguagem configurável**, não fixo no código
- **Fluxo de uma pergunta:**

```
Pergunta → Interpretação → Texto ┐
                          Tabela ├→ Resposta citada
                          Séries ┘
```

- **Funcionalidades da tela:** pergunta em linguagem natural, histórico de análises, fontes da resposta, **ver evidências**, conferência dos números, copiar e exportar

**🗣 Fala sugerida:**
> "A arquitetura tem três peças. A **interface**, onde a pessoa digita; o **serviço**, que interpreta a pergunta e vai buscar nos recuperadores; e o **acervo indexado**, junto com o modelo de linguagem. Um detalhe que vale destacar: **o modelo não está fixo no código**. Ele é escolhido por configuração. Isso significa que dá para trocar de provedor, ou até rodar um modelo local, sem reescrever o sistema — o que é importante em um órgão público, porque ninguém quer ficar preso a um fornecedor. E na tela, além da resposta, tem o que eu considero o recurso mais importante do produto: o botão **'ver evidências'**, que abre o trecho de origem que sustenta cada valor. A resposta deixa de ser uma afirmação e passa a ser uma afirmação **conferível**."

---

## Slide 17 — Demonstração, avaliação e próximos passos

**Na tela:**
- **Demonstração real:** PIB paulista — resposta em **41,9 s**, **9 fontes** · mercado de trabalho — **45,7 s**, **11 fontes**, cada valor com sua própria evidência
- **Base de avaliação:** **40 perguntas** de referência curadas, em três conjuntos — ajuste (15), teste (15) e fora do escopo (10)
- **As 4 dimensões medidas:** fidelidade às fontes · precisão numérica · tempo de resposta · **recusa adequada (100% no conjunto fora do escopo)**
- **Próximos passos:**
  1. preencher o gabarito das respostas esperadas
  2. comparação controlada entre as estratégias de recuperação
  3. teste com usuários reais
  4. ajuste do tempo de resposta (hoje 40–46 s)
  5. concluir a exposição do modo agêntico pelo serviço

**🗣 Fala sugerida:**
> "Na demonstração, uma pergunta sobre o PIB paulista voltou em cerca de 42 segundos, com 9 fontes. Uma pergunta de mercado de trabalho, em cerca de 46 segundos, com 11 fontes — e cada valor com a sua própria evidência. Esses números mostram as duas faces do sistema: ele é rastreável, e ele é lento. Eu prefiro trazer o número do que esconder. Na avaliação, a gente construiu uma base de 40 perguntas de referência em três conjuntos: um de ajuste, um de teste cego e um de perguntas **propositalmente fora do acervo**, para ver se o sistema tem a coragem de dizer que não sabe. Nesse último, a recusa foi de 100%. E os próximos passos são claros: preencher o gabarito das respostas esperadas, comparar as estratégias de forma controlada, testar com usuários reais, melhorar o tempo de resposta e concluir a exposição do modo agêntico. O primeiro passo não depende de tecnologia nova — depende de escrever o que se espera que o sistema responda."

**Frase de fechamento sugerida:**
> "Três coisas para levar: o método resolve um problema real; o trabalho difícil está na **preparação do dado e na recuperação**, não no modelo; e a validação foi estruturada e automatizada, com gabarito e proveniência de página. O sistema não substitui o dado oficial — ele encurta o caminho até ele."

---

# APÊNDICE A — Números que você deve manter consistentes

Cada item abaixo aparece de forma **diferente** em algum arquivo do projeto. Escolha um valor, use o mesmo em todos os slides, e saiba justificar:

| Item | Divergência encontrada | Recomendação |
|---|---|---|
| Edições de boletim | Slides: "cerca de 22" · Relatório e plano: "23 PDFs" | Escolha **um** número. Se não tiver certeza, diga "mais de 20 edições" |
| RAPTOR e Self-RAG | Slides: "estudada, não implementada" · Repositório: pastas com código · Relatório: "quatro variantes implementadas" | Use: **"estudado, com protótipo em laboratório, não avaliado nem exposto no serviço"** |
| Modelo de embeddings | Relatório: `bge-small-en-v1.5` (escolha inicial) · Código atual: **`BAAI/bge-m3`** | Diga: **"houve evolução do modelo; o atual é o bge-m3"** |
| Tamanho do índice | — | **~57 mil trechos** indexados |
| Trechos por resposta | — | **20** narrativos + **10** por recuperador estruturado |
| Tempo de resposta | Slides 19 e 20: 41,9 s e 45,7 s | **40 a 46 segundos** (seja o primeiro a dizer) |
| Base de avaliação | — | **40 perguntas** (15 ajuste + 15 teste + 10 fora do escopo); recusa **100%** |

---

# APÊNDICE B — Perguntas prováveis da banca (respostas em 2 linhas)

**1. "Por que não usar só um ChatGPT?"**
> Porque ele responde de memória, não cita página e inventa quando não sabe. Em dado estatístico, invenção é indistinguível de verdade. O RAG obriga o modelo a responder a partir do trecho recuperado.

**2. "O que é dado não estruturado?"**
> Informação que não está em planilha organizada: texto corrido, tabela dentro de PDF, página de boletim. É a maior parte do conhecimento público — e a mais difícil para o computador.

**3. "O que é embedding, em uma frase?"**
> É uma lista de números que representa o significado de um texto. Textos com significado parecido recebem números parecidos — por isso "desemprego" e "desocupação" ficam próximos.

**4. "O banco vetorial guarda o quê?"**
> Cada trecho é uma ficha com quatro partes: o texto, o vetor de significado, as etiquetas de origem (arquivo, página, tipo) e um identificador.

**5. "O metadado participa do cálculo de semelhança?"**
> Não. O metadado **filtra** ("só fichas do tipo tabela") e **rotula** ("esta veio da página 37"). Quem calcula semelhança é o vetor.

**6. "De onde vem a página que o sistema cita?"**
> Não vem do modelo. O sistema cola a etiqueta de origem na frente do trecho antes de enviá-lo. O modelo lê o carimbo que chegou junto com o recorte.

**7. "Que tipo de RAG é o seu?"**
> Um RAG **modular** com recuperação **avançada** em cada rota. Modular porque escolhe o recuperador certo para cada tipo de pergunta; avançado porque combina semelhança e palavra-chave dentro de cada rota.

**8. "Qual a maior dificuldade do projeto?"**
> A preparação do dado e a recuperação. O modelo não é o gargalo: tabela mal extraída e trecho errado recuperado estragam a resposta antes de o modelo entrar em cena.

**9. "Como você sabe que o sistema funciona?"**
> Por uma base de 40 perguntas de referência em três conjuntos, com métricas de fidelidade às fontes, precisão numérica, tempo de resposta e recusa. No conjunto de perguntas fora do acervo, a recusa foi de 100%.

**10. "O sistema pode errar?"**
> Pode, e é por isso que a conferência automática não substitui a auditoria humana. O sistema aponta a fonte exata para que a verificação seja rápida — mas ela continua necessária.

---

# APÊNDICE C — Versão de 1 minuto (se o tempo for cortado)

> "Os indicadores de São Paulo estão publicados em boletins oficiais. Achar um número exige abrir dezenas de PDFs e anotar a página — lento e repetitivo. A Nadia resolve isso com RAG: o sistema divide os boletins em trechos, transforma cada trecho em um endereço de significado e guarda tudo num índice local, junto com a etiqueta de arquivo e página. Quando você pergunta em português, ele busca os trechos mais relevantes por semelhança **e** por palavra-chave, e só então pede ao modelo que escreva a resposta usando **apenas** esses trechos — com a fonte carimbada. O trabalho difícil não está no modelo, está na preparação do dado e na recuperação: tabela não é texto corrido e série temporal precisa ser recuperada ano a ano. Temos 40 perguntas de avaliação, recusa de 100% nas perguntas fora do acervo e um limite honesto: a conferência automática não substitui a auditoria humana, e o alcance do sistema é o acervo indexado."

---

# APÊNDICE D — Como cortar se o tempo apertar

| Se você tiver | Corte |
|---|---|
| **~15 min** | Apresente tudo |
| **~10 min** | Corte o Slide 6 (embedding em detalhe) e o Slide 11 (tipos) — resuma os tipos em uma fala no Slide 12 |
| **~7 min** | Comece no Slide 2 (ideia) e vá direto para 4, 8, 10, 12, 14, 15, 16, 17. Nunca corte o **14 (limitações)** nem o **17 (próximos passos)** |
