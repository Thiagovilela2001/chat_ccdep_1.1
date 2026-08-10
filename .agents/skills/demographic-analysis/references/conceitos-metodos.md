# Conceitos e métodos demográficos

## Seleção de fonte

| Necessidade | Fonte preferencial | Cuidado central |
|---|---|---|
| Estoque e estrutura municipal detalhada | Censo Demográfico, IBGE | Data de referência, cobertura e limites territoriais |
| População anual entre censos | Estimativas populacionais, IBGE | Revisões após novo censo e método de interpolação |
| Cenários futuros | Projeções da população, IBGE ou órgão documentado | Hipóteses, cenário, ano-base e horizonte |
| Nascimentos | Estatísticas do Registro Civil/IBGE ou SINASC | Registro por ocorrência ou residência e sub-registro |
| Óbitos | Registro Civil/IBGE ou SIM | Residência, ocorrência, cobertura e causas mal definidas |
| Migração | Censo e pesquisas domiciliares | Conceito de migrante, janela temporal e residência anterior |
| Características domiciliares e amostrais | PNAD Contínua | Pesos, desenho amostral e nível geográfico publicável |

Não combinar fontes apenas porque usam o mesmo nome de indicador. Conferir
universo, período, geografia, conceito, coleta e revisão.

## Medidas essenciais

### Crescimento populacional

- Variação absoluta: `ΔP = P_t - P_0`.
- Variação percentual: `100 × (P_t / P_0 - 1)`.
- Taxa média geométrica anual: `100 × ((P_t / P_0)^(1/n) - 1)`.
- Equação compensadora: `P_t = P_0 + B - D + I - E`, acrescida de ajustes
  quando a fonte os documentar.
- Crescimento vegetativo: `B - D`.
- Saldo migratório: `I - E`; quando obtido pela equação compensadora, declarar
  que se trata de saldo residual estimado.

Usar anos completos ou fração temporal documentada em `n`. Não dividir variação
percentual acumulada pelo número de anos para substituir taxa geométrica.

### Natalidade, fecundidade e reprodução

- Taxa bruta de natalidade: `nascidos vivos / população média × 1.000`.
- Taxa específica de fecundidade: `nascidos vivos de mulheres na faixa etária /
  mulheres da mesma faixa etária`.
- Taxa de fecundidade total: soma das taxas específicas por idade, multiplicada
  pela amplitude dos grupos etários quando aplicável.

Não usar taxa de natalidade como sinônimo de fecundidade. Não calcular taxa de
fecundidade total dividindo todos os nascimentos pelo total de mulheres.

### Mortalidade e sobrevivência

- Taxa bruta de mortalidade: `óbitos / população média × 1.000`.
- Taxa específica de mortalidade: `óbitos do grupo / população do grupo`.
- Mortalidade infantil: `óbitos menores de 1 ano / nascidos vivos × 1.000`.
- Esperança de vida ao nascer: medida sintética derivada de tábua de mortalidade;
  não representa idade média ao morrer no período.

Comparar taxas brutas entre populações com cautela. Diferenças podem refletir
estrutura etária; preferir taxas específicas ou padronizadas.

### Estrutura populacional

- Razão de dependência jovem: `P(0–14) / P(15–64) × 100`.
- Razão de dependência idosa: `P(65+) / P(15–64) × 100`.
- Razão de dependência total: `(P(0–14) + P(65+)) / P(15–64) × 100`.
- Índice de envelhecimento: `P(65+) / P(0–14) × 100`.
- Razão de sexo: explicitar orientação, por exemplo `homens / mulheres × 100`.
- Densidade demográfica: `população / área`; não equivale a urbanização.

Faixas etárias variam entre fontes. Recalcular componentes com faixas
compatíveis ou declarar a definição usada.

## Comparabilidade

### Tempo

- Distinguir data de referência de data de publicação.
- Verificar revisão de estimativas após censos.
- Não chamar diferença entre dois pontos de tendência persistente sem evidência
  intermediária.
- Comparar gerações por coortes de nascimento, não por faixa etária fixa em
  datas distintas.

### Território

- Usar malha e código territorial do mesmo período dos dados.
- Harmonizar municípios criados, extintos, desmembrados ou agregados.
- Não somar regiões sobrepostas nem comparar município com arranjo populacional,
  região metropolitana ou região administrativa como unidades equivalentes.
- Distinguir local de residência de local de ocorrência em registros vitais.

### Qualidade e incerteza

- Registrar cobertura, sub-registro, imputação e mudanças metodológicas citadas.
- Em pesquisas amostrais, usar pesos e desenho amostral; apresentar intervalo de
  confiança ou coeficiente de variação quando disponível.
- Em pequenas áreas ou eventos raros, evitar ranking baseado em contagens
  instáveis; agregar períodos ou usar suavização apenas com método documentado.
- Em projeções, comunicar hipótese e faixa de incerteza quando disponível; não
  apresentar valor projetado como observação realizada.

## Estrutura de resposta

1. Dar resultado principal com indicador, período e território.
2. Mostrar comparação adequada: nível, variação, taxa ou composição.
3. Explicar quais componentes sustentam a mudança, se documentados.
4. Apontar heterogeneidade etária, territorial ou por sexo quando relevante.
5. Registrar ressalva metodológica somente quando alterar leitura do resultado.
6. Citar fontes junto das afirmações que sustentam.

## Erros a impedir

- Confundir pontos percentuais com variação percentual.
- Inferir migração líquida apenas pela queda ou alta da população.
- Somar percentuais de universos diferentes.
- Comparar pirâmides etárias com escalas ou faixas incompatíveis.
- Atribuir causalidade a política, pandemia ou economia sem evidência da fonte.
- Projetar tendência por extrapolação informal quando existe projeção oficial ou
  quando hipóteses não foram definidas.
- Interpretar razão de dependência como número observado de dependentes
  econômicos; ela usa grupos etários convencionais.
