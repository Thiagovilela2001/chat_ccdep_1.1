import { readStoredJson, writeStorage } from "./storage";

export const FEATURED_QUESTIONS_STORAGE_KEY = "nadia.featuredQuestions.v1";

export const FEATURED_QUESTION_POOLS = [
  {
    id: "labor",
    eyebrow: "Mercado de trabalho",
    questions: [
      "Como evoluiu o emprego formal paulista nos boletins mais recentes?",
      "Quais setores mais abriram vagas formais em São Paulo?",
      "Como admissões e desligamentos afetaram o saldo de empregos paulista?",
      "Quais regiões paulistas apresentaram maior dinamismo no mercado de trabalho?",
      "Como variaram ocupação, desocupação e rendimento do trabalho em São Paulo?",
      "O emprego formal cresceu mais na capital, na região metropolitana ou no interior?",
    ],
  },
  {
    id: "activity",
    eyebrow: "Atividade econômica",
    questions: [
      "Quais setores mais contribuíram para o crescimento de São Paulo?",
      "Como evoluiu o PIB paulista nos períodos mais recentes?",
      "Qual foi o desempenho recente da indústria paulista?",
      "Como comércio e serviços influenciaram a atividade econômica do estado?",
      "Quais atividades econômicas mostraram aceleração ou desaceleração?",
      "Que fatores explicaram as principais variações da economia paulista?",
    ],
  },
  {
    id: "comparison",
    eyebrow: "Análise comparada",
    questions: [
      "Compare indústria, serviços e comércio no período analisado.",
      "Como o desempenho econômico paulista se compara ao brasileiro?",
      "Quais diferenças aparecem entre capital, região metropolitana e interior?",
      "Compare o crescimento dos principais setores da economia paulista.",
      "Quais indicadores melhoraram e quais pioraram entre os períodos analisados?",
      "Como emprego e atividade econômica evoluíram em conjunto em São Paulo?",
    ],
  },
  {
    id: "demography",
    eyebrow: "Dinâmica populacional",
    questions: [
      "Como a população paulista evoluiu nos censos mais recentes?",
      "Quais regiões de São Paulo mais cresceram em população?",
      "O crescimento populacional foi maior na capital, na região metropolitana ou no interior?",
      "Quais municípios paulistas perderam população no período mais recente?",
    ],
  },
  {
    id: "demography-aging",
    eyebrow: "Envelhecimento",
    questions: [
      "Como avançou o envelhecimento da população paulista?",
      "Quais regiões paulistas apresentam maior proporção de idosos?",
      "Como mudou a estrutura etária de São Paulo nas últimas décadas?",
      "Onde o índice de envelhecimento cresceu mais rapidamente?",
    ],
  },
  {
    id: "demography-fertility",
    eyebrow: "Fecundidade e natalidade",
    questions: [
      "Como evoluiu a fecundidade no Estado de São Paulo?",
      "Quais regiões paulistas registraram maior queda da natalidade?",
      "Como mudou a idade média das mães paulistas?",
      "Quais diferenças regionais aparecem nos níveis de fecundidade?",
    ],
  },
  {
    id: "demography-longevity",
    eyebrow: "Mortalidade e longevidade",
    questions: [
      "Como evoluiu a esperança de vida da população paulista?",
      "Quais diferenças de longevidade aparecem entre as regiões de São Paulo?",
      "Como a mortalidade infantil mudou no estado?",
      "Quais grupos etários tiveram maior mudança nos níveis de mortalidade?",
    ],
  },
  {
    id: "demography-migration",
    eyebrow: "Migração e urbanização",
    questions: [
      "Como a migração influenciou o crescimento populacional paulista?",
      "Quais regiões mais ganharam população por migração?",
      "Como os fluxos migratórios se distribuem entre capital e interior?",
      "Quais mudanças ocorreram na urbanização do Estado de São Paulo?",
    ],
  },
  {
    id: "demography-projections",
    eyebrow: "Projeções demográficas",
    questions: [
      "O que as projeções indicam para a população paulista nos próximos anos?",
      "Quando a população do estado deve parar de crescer?",
      "Como deve evoluir a razão de dependência em São Paulo?",
      "Quais regiões devem envelhecer mais rapidamente nas próximas décadas?",
    ],
  },
  {
    id: "investment",
    eyebrow: "Investimentos",
    questions: [
      "Quais setores concentraram mais investimentos anunciados em São Paulo?",
      "Quais regiões paulistas receberam mais projetos de investimento?",
      "Como os investimentos anunciados evoluíram nos períodos mais recentes?",
      "Quais foram os maiores projetos de investimento anunciados no estado?",
      "Como os investimentos se distribuíram entre indústria, serviços e infraestrutura?",
      "Quais municípios se destacaram na atração de novos investimentos?",
    ],
  },
  {
    id: "regional",
    eyebrow: "Análise regional",
    questions: [
      "Quais regiões paulistas apresentaram maior dinamismo econômico?",
      "Como a atividade econômica se distribui entre capital e interior?",
      "Quais setores se destacam nas diferentes regiões de São Paulo?",
      "Onde aparecem as maiores diferenças regionais de crescimento?",
      "Quais regiões mostram maior especialização industrial?",
      "Como emprego, população e investimentos se relacionam regionalmente?",
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
