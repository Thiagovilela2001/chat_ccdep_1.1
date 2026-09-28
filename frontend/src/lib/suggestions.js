import { readStoredJson, writeStorage } from "./storage";

export const FEATURED_QUESTIONS_STORAGE_KEY = "nadia.featuredQuestions.v1";

export const FEATURED_QUESTION_POOLS = [
  {
    id: "labor",
    eyebrow: "Mercado de trabalho",
    questions: [
      "Quantos empregos formais São Paulo criou ou perdeu no período mais recente, e qual foi a variação frente ao período anterior?",
      "Quais cinco setores mais abriram vagas formais em São Paulo, quantos postos cada um gerou e qual foi sua participação no saldo total?",
      "Quantas admissões e demissões ocorreram no período mais recente, e qual foi o saldo de empregos paulista?",
      "Quais cinco regiões paulistas tiveram os maiores saldos de emprego, com valores e taxas de variação?",
      "Quais foram as taxas de ocupação e desocupação e o rendimento médio em São Paulo, com as respectivas variações?",
      "Compare capital, região metropolitana e interior pelo saldo e pela variação percentual do emprego formal.",
    ],
  },
  {
    id: "social",
    eyebrow: "Proteção social",
    questions: [
      "Quantas pessoas deixaram de receber o Bolsa Família em São Paulo nos períodos disponíveis, e qual foi a variação percentual?",
      "Qual foi o valor total transferido pelo Bolsa Família em São Paulo e quantos beneficiários foram atendidos no período mais recente?",
      "Quais são os tipos de benefício do Bolsa Família, com o valor médio de cada um e o total de beneficiários?",
      "Como variaram os inscritos no CadÚnico e os beneficiários do Bolsa Família no estado, em valores absolutos e percentuais?",
      "Compare São Paulo e o Brasil no número de beneficiários e no valor médio dos benefícios, com a participação percentual do estado.",
      "Quais municípios paulistas atendem o maior número de famílias pelo Bolsa Família e qual é a participação de cada um no total do estado?",
    ],
  },
  {
    id: "activity",
    eyebrow: "Atividade econômica",
    questions: [
      "Quais setores mais contribuíram para o crescimento paulista e qual foi a variação percentual de cada um?",
      "Qual foi a variação do PIB paulista nos três períodos mais recentes disponíveis?",
      "Quanto a indústria paulista cresceu ou recuou no período mais recente, em valor e percentual?",
      "Quais foram as taxas de variação de comércio e serviços, e qual deles mais contribuiu para o resultado do estado?",
      "Quais atividades mais aceleraram ou desaceleraram entre os dois períodos mais recentes, e em quantos pontos percentuais?",
      "Qual foi a participação percentual de indústria, comércio e serviços na atividade econômica paulista?",
    ],
  },
  {
    id: "comparison",
    eyebrow: "Análise comparada",
    questions: [
      "Compare indústria, serviços e comércio pelas taxas de variação e apresente o ranking do período mais recente.",
      "Compare as taxas de crescimento de São Paulo e do Brasil e informe a diferença em pontos percentuais.",
      "Compare capital, região metropolitana e interior com valores, taxas e a diferença entre o maior e o menor resultado.",
      "Quais setores tiveram as maiores e menores taxas de crescimento, e qual foi a distância entre eles?",
      "Quais indicadores melhoraram ou pioraram entre os dois períodos mais recentes, e quanto cada um variou?",
      "Compare a variação do emprego e da atividade econômica em São Paulo, com os valores de cada indicador por período.",
    ],
  },
  {
    id: "demography",
    eyebrow: "Dinâmica populacional",
    questions: [
      "Qual era a população paulista nos dois censos mais recentes e qual foi a variação absoluta e percentual?",
      "Quais cinco regiões mais ganharam população, com o aumento em habitantes e em percentual?",
      "Compare capital, região metropolitana e interior pela população e pela taxa de crescimento entre os censos.",
      "Quais municípios mais perderam população, quantos habitantes perderam e qual foi a queda percentual?",
    ],
  },
  {
    id: "demography-aging",
    eyebrow: "Envelhecimento",
    questions: [
      "Qual é a proporção de idosos em São Paulo nos períodos disponíveis e quantos pontos percentuais ela aumentou?",
      "Quais cinco regiões têm a maior proporção de idosos, com os percentuais e a população idosa de cada uma?",
      "Como mudaram as participações percentuais das principais faixas etárias de São Paulo?",
      "Quais regiões tiveram o maior aumento do índice de envelhecimento e quais foram os valores inicial e final?",
    ],
  },
  {
    id: "demography-fertility",
    eyebrow: "Fecundidade e natalidade",
    questions: [
      "Qual foi a taxa de fecundidade de São Paulo nos períodos disponíveis e quanto ela variou?",
      "Quais regiões tiveram as maiores quedas no número de nascimentos e qual foi a redução percentual?",
      "Qual era a idade média das mães nos períodos disponíveis e quantos anos ela aumentou ou diminuiu?",
      "Quais regiões têm as maiores e menores taxas de fecundidade e qual é a diferença entre elas?",
    ],
  },
  {
    id: "demography-longevity",
    eyebrow: "Mortalidade e longevidade",
    questions: [
      "Qual foi a esperança de vida em São Paulo nos períodos disponíveis e quantos anos ela aumentou?",
      "Quais regiões têm a maior e a menor esperança de vida, e qual é a diferença em anos?",
      "Qual foi a taxa de mortalidade infantil nos períodos disponíveis e qual foi a redução percentual?",
      "Quais grupos etários tiveram as maiores variações nas taxas de mortalidade, com os valores inicial e final?",
    ],
  },
  {
    id: "demography-migration",
    eyebrow: "Migração e urbanização",
    questions: [
      "Qual foi o saldo migratório de São Paulo e que parcela do crescimento populacional ele representou?",
      "Quais cinco regiões tiveram os maiores ganhos migratórios, em pessoas e por mil habitantes?",
      "Compare capital e interior pelo saldo migratório e pela taxa líquida de migração.",
      "Qual foi a taxa de urbanização de São Paulo nos períodos disponíveis e quantos pontos percentuais ela variou?",
    ],
  },
  {
    id: "demography-projections",
    eyebrow: "Projeções demográficas",
    questions: [
      "Qual é a população projetada de São Paulo para os próximos marcos disponíveis e qual é a variação percentual entre eles?",
      "Em que ano a população paulista deve atingir o pico e qual será o total projetado de habitantes?",
      "Quais são os valores projetados da razão de dependência e quantos pontos ela deve variar?",
      "Quais regiões terão o maior aumento da proporção de idosos, com os percentuais inicial e projetado?",
    ],
  },
  {
    id: "investment",
    eyebrow: "Investimentos",
    questions: [
      "Quais setores concentraram mais investimentos anunciados, em reais e como participação do total paulista?",
      "Quais cinco regiões receberam os maiores valores de investimento e quantos projetos foram anunciados em cada uma?",
      "Qual foi o valor total dos investimentos anunciados nos períodos mais recentes e qual foi a variação percentual?",
      "Quais foram os cinco maiores projetos anunciados, com valor, município e setor?",
      "Quanto dos investimentos foi destinado à indústria, aos serviços e à infraestrutura, em reais e percentual?",
      "Quais cinco municípios atraíram mais investimentos, com valor total e número de projetos?",
    ],
  },
  {
    id: "regional",
    eyebrow: "Análise regional",
    questions: [
      "Quais cinco regiões tiveram as maiores taxas de crescimento econômico e quais foram seus valores?",
      "Qual é a participação da capital, da região metropolitana e do interior na atividade econômica paulista?",
      "Quais setores lideram em cada região e qual é a participação percentual de cada um?",
      "Quais regiões apresentam as maiores diferenças de crescimento e qual é a distância em pontos percentuais?",
      "Quais regiões têm os maiores índices de especialização industrial e quais são os valores?",
      "Compare emprego, população e investimentos das principais regiões com valores, taxas e participação estadual.",
    ],
  },
];

function validPreviousSelection(value) {
  return (
    value !== null
    && typeof value === "object"
    && !Array.isArray(value)
    && Object.values(value).every((title) => typeof title === "string")
  );
}

function randomIndex(length, random) {
  const sampled = Number(random());
  if (!Number.isFinite(sampled)) return 0;
  return Math.min(Math.max(Math.floor(sampled * length), 0), length - 1);
}

export function rotateFeaturedQuestions(storage, random = Math.random) {
  const previous = readStoredJson(
    storage,
    FEATURED_QUESTIONS_STORAGE_KEY,
    {},
    validPreviousSelection,
  );
  const selected = FEATURED_QUESTION_POOLS.map((pool) => {
    const candidates = pool.questions.filter((title) => title !== previous[pool.id]);
    const available = candidates.length > 0 ? candidates : pool.questions;
    const title = available[randomIndex(available.length, random)];
    return { id: pool.id, eyebrow: pool.eyebrow, title };
  });

  writeStorage(
    storage,
    FEATURED_QUESTIONS_STORAGE_KEY,
    JSON.stringify(
      Object.fromEntries(selected.map(({ id, title }) => [id, title])),
    ),
  );
  return selected;
}
