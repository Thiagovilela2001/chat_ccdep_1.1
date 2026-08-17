// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import {
  annotateNumericCitations,
  conceptualizeTabularSnippet,
  isCalendarYearCitation,
  parseTabularSnippet,
} from "./lib/numericCitations";

describe("inicialização da interface", () => {
  let container;
  let root;

  beforeEach(() => {
    globalThis.IS_REACT_ACT_ENVIRONMENT = true;
    localStorage.clear();
    sessionStorage.clear();
    globalThis.fetch = vi.fn(() => Promise.reject(new TypeError("API indisponível")));
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: vi.fn().mockResolvedValue(undefined) },
    });
    HTMLElement.prototype.scrollTo = vi.fn();
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.restoreAllMocks();
  });

  it("renderiza o estado inicial mesmo sem acesso ao backend", async () => {
    localStorage.setItem("nadia.rag.v1", "meta");
    await act(async () => root.render(<App />));

    expect(container.textContent).toContain("Nadia");
    expect(container.textContent).not.toContain("⌘ K");
    expect(container.textContent).toContain("Mercado de trabalho");
    expect(container.textContent).toContain("Análise documental");
    expect(container.textContent).not.toContain("Metodologia");
    expect(container.textContent).not.toContain("Meta RAG");
    expect(container.textContent).not.toContain("Agentic");
    expect(container.textContent).not.toContain("RAPTOR");
    expect(container.textContent).not.toContain("Self-RAG");
    expect(globalThis.fetch).toHaveBeenCalledWith(
      "http://localhost:8000/health",
      expect.any(Object),
    );
  });

  it("expande e recolhe o trecho recuperado ao clicar na fonte", async () => {
    localStorage.setItem("nadia.messages.v1", JSON.stringify([{
      id: "assistant-1",
      role: "assistant",
      content: "Resposta baseada no boletim.",
      createdAt: "2026-08-04T12:00:00.000Z",
      meta: {
        answer: "Resposta baseada no boletim.",
        sources: [{
          file: "conjuntura/boletim.pdf",
          page: 25,
          score: 0.9,
          excerpt: "Este é o trecho documental recuperado.",
        }],
        validation: { verified: 0, total: 0, unverified: [] },
      },
    }]));

    await act(async () => root.render(<App />));
    await act(async () => {
      container.querySelector('[aria-label="Alternar painel de evidências"]').click();
    });

    const sourceButton = container.querySelector(".source-card-trigger");
    expect(sourceButton.getAttribute("aria-expanded")).toBe("false");
    expect(container.textContent).not.toContain("Este é o trecho documental recuperado.");

    await act(async () => sourceButton.click());
    expect(sourceButton.getAttribute("aria-expanded")).toBe("true");
    expect(container.textContent).toContain("Este é o trecho documental recuperado.");

    await act(async () => sourceButton.click());
    expect(sourceButton.getAttribute("aria-expanded")).toBe("false");
    expect(container.textContent).not.toContain("Este é o trecho documental recuperado.");
  });

  it("torna um dado numérico verificável e fixa sua fonte ao clicar", async () => {
    const content = "O PIB paulista cresceu 12,5%.";
    const start = Array.from(content).indexOf("1");
    localStorage.setItem("nadia.messages.v1", JSON.stringify([{
      id: "assistant-numeric",
      role: "assistant",
      content,
      createdAt: "2026-08-04T12:00:00.000Z",
      meta: {
        answer: content,
        sources: [{
          file: "conjuntura/boletim.pdf",
          page: 8,
          score: 0.92,
          excerpt: "O PIB paulista cresceu 12,5% no período.",
        }],
        numeric_citations: [{
          value: "12,5%",
          start,
          end: start + Array.from("12,5%").length,
          source_index: 0,
          file: "conjuntura/boletim.pdf",
          page: 8,
          score: 0.92,
          snippet: "Indicador: PIB paulista\nValor: 12,5%\nPeríodo: 2025",
          content_type: "table",
          claim: "No período analisado, o PIB paulista cresceu 12,5%.",
          explanation: "O PIB paulista registrou crescimento de 12,5% no período analisado.",
        }],
        validation: { verified: 1, total: 1, unverified: [] },
      },
    }]));

    await act(async () => root.render(<App />));
    const citationButton = container.querySelector('[aria-label="Ver fonte do valor 12,5%"]');
    expect(citationButton).not.toBeNull();
    expect(citationButton.getAttribute("aria-expanded")).toBe("false");

    await act(async () => citationButton.click());
    expect(citationButton.getAttribute("aria-expanded")).toBe("true");
    expect(document.body.textContent).toContain("Dado verificado");
    expect(document.body.textContent).toContain("p. 8");
    expect(document.body.textContent).toContain("O que esse dado mostra");
    expect(document.body.textContent).toContain(
      "O PIB paulista registrou crescimento de 12,5% no período analisado.",
    );
    expect(document.body.textContent).toContain("PIB paulista");
    expect(document.body.textContent).toContain("Período");
    expect(document.body.querySelector(".numeric-source-table mark").textContent).toBe("12,5%");
    expect(document.body.querySelector(".numeric-table-value").textContent).toBe("2025");
    expect(document.body.querySelector(".numeric-source-raw").open).toBe(false);

    const copyEvidenceButton = [...document.body.querySelectorAll("button")]
      .find((button) => button.textContent.includes("Copiar trecho"));
    await act(async () => copyEvidenceButton.click());
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
      "Indicador: PIB paulista\nValor: 12,5%\nPeríodo: 2025",
    );
    expect(copyEvidenceButton.textContent).toContain("Copiado");

    const openSourceButton = [...document.body.querySelectorAll("button")]
      .find((button) => button.textContent.includes("Ver fonte completa"));
    await act(async () => openSourceButton.click());
    expect(container.querySelector(".inspector").classList.contains("is-open")).toBe(true);
    expect(container.querySelector(".source-card-trigger").getAttribute("aria-expanded")).toBe("true");
  });

  it("mantém anos no texto sem abrir validação argumentativa", async () => {
    const content = "O saldo ficou negativo entre 2010 e 2022.";
    const characters = Array.from(content);
    const citations = ["2010", "2022"].map((value) => {
      const start = characters.join("").indexOf(value);
      return {
        value,
        start,
        end: start + value.length,
        source_index: 0,
        file: "demografia/boletim.pdf",
        page: 5,
        score: 0.9,
        snippet: content,
        content_type: "text",
        claim: content,
      };
    });
    localStorage.setItem("nadia.messages.v1", JSON.stringify([{
      id: "assistant-years",
      role: "assistant",
      content,
      createdAt: "2026-08-04T12:00:00.000Z",
      meta: {
        answer: content,
        sources: [],
        numeric_citations: citations,
        validation: { verified: 2, total: 2, unverified: [] },
      },
    }]));

    await act(async () => root.render(<App />));

    expect(container.textContent).toContain("2010");
    expect(container.textContent).toContain("2022");
    expect(container.querySelector('[aria-label="Ver fonte do valor 2010"]')).toBeNull();
    expect(container.querySelector('[aria-label="Ver fonte do valor 2022"]')).toBeNull();
  });
});

describe("anotação de citações numéricas", () => {
  it("não trata anos-calendário como dados argumentativos", () => {
    expect(isCalendarYearCitation("2010")).toBe(true);
    expect(isCalendarYearCitation("2022")).toBe(true);
    expect(isCalendarYearCitation("2.010")).toBe(false);
    expect(isCalendarYearCitation("2010%")).toBe(false);
  });

  it("respeita offsets Unicode recebidos do backend", () => {
    const content = "📈 O índice chegou a 18,7%.";
    const characters = Array.from(content);
    const start = characters.indexOf("1");
    expect(annotateNumericCitations(content, [{
      value: "18,7%",
      start,
      end: start + 5,
    }])).toContain("[18,7%](#numeric-citation-0)");
  });

  it("converte linhas estruturadas em tabela legível", () => {
    expect(parseTabularSnippet(
      "Setor: Indústria\nEmpregos: 125.400\nVariação: 3,2%",
    )).toEqual({
      kind: "key-value",
      headers: ["Campo", "Valor"],
      rows: [
        ["Setor", "Indústria"],
        ["Empregos", "125.400"],
        ["Variação", "3,2%"],
      ],
    });
  });

  it("conceitua o valor e separa seu contexto", () => {
    const table = parseTabularSnippet(
      "Indicador: PIB paulista\nValor: 12,5%\nPeríodo: 2025",
    );
    expect(conceptualizeTabularSnippet(table, "12,5%")).toEqual({
      subject: "PIB paulista",
      qualifiers: [{ label: "Período", value: "2025" }],
      citedRow: ["Valor", "12,5%"],
    });
  });
});
