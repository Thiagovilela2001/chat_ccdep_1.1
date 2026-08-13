# Mapa do corpus Seade Social

Inventário de referência em 13 de agosto de 2026.

| Coleção | PDFs | Conteúdo principal |
|---|---:|---|
| `painel` | 11 | CadÚnico, Bolsa Família, BPC e transferência de renda |
| `trabalho` | 174 | Ocupação, rendimento, informalidade, desigualdades, regiões e emprego formal |
| Total | 185 | Boletins da Fundação Seade |

A skill `social-protection-analysis` cobre `painel`. A skill
`labor-market-analysis`, mantida na pasta legada `labor_market_analysis`, cobre
`trabalho`.

Não copiar números conjunturais dos PDFs para `SKILL.md`. Recuperar números do
corpus no momento da resposta; manter nas skills somente regras relativamente
estáveis de interpretação.

Executar `python scripts/build_corpus_manifest.py` para gerar inventário JSON
com arquivo, coleção, tema, título, páginas e tamanho.
