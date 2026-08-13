# Métodos gerais de mercado de trabalho

## Seleção de fontes

| Pergunta | Fonte principal | Unidade e frequência | Cuidado central |
|---|---|---|---|
| Ocupação, desocupação e informalidade | PNAD Contínua/IBGE | Pessoa; mensal móvel, trimestral ou anual | Usar pesos e desenho amostral |
| Fluxo do emprego formal | Novo Caged/MTE | Movimentação; mensal | Quebra com Caged antigo em 2020 |
| Estoque e perfil do emprego formal | Rais/MTE | Vínculo; anual | Não cobre trabalho informal |
| Diagnóstico municipal amplo | Censo Demográfico/IBGE | Pessoa ou domicílio; decenal | Conceitos e data de referência |
| Rendimento e desigualdade | PNAD Contínua ou Rais | Pessoa ou vínculo | Populações e coberturas diferentes |

Escolher fonte pela população-alvo e medida, não apenas pela frequência mais
recente. Não usar Caged para medir desemprego nem PNAD para reproduzir saldo
administrativo de admissões e desligamentos.

## PNAD Contínua

### Conceitos usuais

- `VD4001`: condição na força de trabalho.
- `VD4002`: condição de ocupação.
- `VD4016` e `VD4019`: rendimentos habitual e efetivo do trabalho principal.
- `VD4020`: rendimento habitual de todos os trabalhos.
- `V4012`: posição na ocupação.
- `V4010` e `V4013`: atividade econômica.
- `V2007`, `V2009` e `V2010`: sexo, idade e raça/cor.
- `V1028`: peso amostral, conforme edição do dicionário.

Confirmar nomes e códigos no dicionário correspondente ao arquivo utilizado;
layouts podem mudar.

### Desenho amostral

- Aplicar pesos amostrais, estratos e unidades primárias de amostragem.
- Calcular variância sob desenho complexo; não usar erro-padrão de amostra
  aleatória simples.
- Apresentar intervalo de confiança quando precisão for material.
- Avaliar coeficiente de variação e tamanho amostral em células pequenas.
- Não somar estimativas de trimestres móveis sobrepostos.

### Indicadores

- Taxa de participação: força de trabalho / população em idade de trabalhar.
- Taxa de desocupação: desocupados / força de trabalho.
- Nível de ocupação: ocupados / população em idade de trabalhar.
- Taxa composta de subutilização: numerador e força de trabalho ampliada
  conforme definição IBGE vigente.
- Informalidade: construir com categorias documentadas de posição, carteira e
  CNPJ; não usar uma única variável sem validar definição.

## Novo Caged e Rais

### Novo Caged

- Unidade: movimentação de admissão ou desligamento.
- Saldo: admissões menos desligamentos.
- Desagregar por competência, município, CNAE, CBO e perfil somente quando os
  campos tiverem cobertura adequada.
- Verificar declarações fora do prazo, revisões e ajustes da série.
- Tratar fevereiro de 2020 como início do Novo Caged; documentar qualquer
  encadeamento com série antiga.

### Rais

- Unidade: vínculo formal; uma pessoa pode possuir mais de um vínculo.
- Usar para estoque anual, remuneração, setor, ocupação e perfil do emprego
  formal.
- Distinguir estoque em 31 de dezembro, vínculos ativos no ano e movimentações.
- Não generalizar resultados Rais para mercado informal ou população ocupada
  total.

## Rendimento

1. Escolher habitual ou efetivo conforme pergunta.
2. Distinguir rendimento do trabalho principal, todos os trabalhos, massa e
   rendimento domiciliar.
3. Informar se média considera ocupados com rendimento, todos os ocupados ou
   outra população.
4. Deflacionar com índice, período e data-base documentados.
5. Para desigualdade, informar população, conceito de renda e ponderação antes
   de calcular Gini, percentis ou razões entre grupos.

## Séries temporais

- Comparar mês contra mês anterior somente com série dessazonalizada adequada.
- Usar mesmo período do ano anterior ou acumulado em 12 meses para reduzir
  sazonalidade quando série ajustada não existir.
- Verificar pandemia, mudanças de questionário, reponderações e quebras de
  sistemas administrativos como possíveis rupturas, sem atribuir causalidade
  automaticamente.
- Não substituir série oficial dessazonalizada por ajuste próprio sem explicar
  método e diferenças.

## Análises setoriais e regionais

- Harmonizar nível CNAE antes de comparar PNAD, Caged e Rais.
- Distinguir local de residência, local de trabalho e sede do estabelecimento.
- Não confundir participação, crescimento, especialização e contribuição.
- Aplicar quociente locacional, shift-share, Gini, Theil ou Moran somente com
  denominadores, territórios e períodos compatíveis.
- Verificar mudanças de limites municipais e regiões agregadas.

## Integração multifonte

| Objetivo | Integração recomendada |
|---|---|
| Quantidade e qualidade de postos formais | Caged para fluxo; Rais para estoque e remuneração |
| Formalidade no mercado total | PNAD para população ocupada; Caged/Rais como evidência administrativa complementar |
| Diagnóstico municipal | Censo, Rais e Caged, preservando unidades distintas |
| Rendimento real | PNAD ou Rais com deflator compatível |

Não fundir bases por agregados sem documentar chaves, cobertura e dupla
contagem. Quando fontes divergirem, explicar universo e método antes de buscar
uma narrativa única.

## Reprodutibilidade

- Registrar versão, data de extração, filtros, dicionário e códigos territoriais.
- Fixar sementes em métodos estocásticos.
- Preservar dados brutos e produzir tabelas intermediárias auditáveis.
- Citar fonte, período e página ou tabela para cada resultado publicado.
