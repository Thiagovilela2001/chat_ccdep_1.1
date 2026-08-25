// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from "vitest";

import {
  FEATURED_QUESTION_POOLS,
  FEATURED_QUESTIONS_STORAGE_KEY,
  rotateFeaturedQuestions,
} from "./suggestions";

describe("perguntas em destaque", () => {
  beforeEach(() => localStorage.clear());

  it("troca todas as perguntas após nova inicialização", () => {
    const first = rotateFeaturedQuestions(localStorage, () => 0);
    const second = rotateFeaturedQuestions(localStorage, () => 0);

    expect(first).toHaveLength(FEATURED_QUESTION_POOLS.length);
    expect(second).toHaveLength(FEATURED_QUESTION_POOLS.length);
    first.forEach((question, index) => {
      expect(second[index].id).toBe(question.id);
      expect(second[index].title).not.toBe(question.title);
    });
  });

  it("persiste somente a última pergunta de cada tema", () => {
    const selected = rotateFeaturedQuestions(localStorage, () => 0.5);
    const stored = JSON.parse(
      localStorage.getItem(FEATURED_QUESTIONS_STORAGE_KEY),
    );

    expect(stored).toEqual(
      Object.fromEntries(selected.map(({ id, title }) => [id, title])),
    );
  });

  it("ignora estado salvo inválido", () => {
    localStorage.setItem(FEATURED_QUESTIONS_STORAGE_KEY, "[]");

    const selected = rotateFeaturedQuestions(localStorage, () => Number.NaN);

    expect(selected.every(({ title }) => typeof title === "string")).toBe(true);
  });

  it("mantém vários cards dedicados à demografia", () => {
    const demographicPools = FEATURED_QUESTION_POOLS.filter(({ id }) =>
      id.startsWith("demography"),
    );

    expect(demographicPools.length).toBeGreaterThanOrEqual(6);
    expect(demographicPools.every(({ questions }) => questions.length >= 4)).toBe(true);
  });
});
