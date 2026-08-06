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
