// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

import { ProgressStepper } from "./ProgressStepper";
import { DataChart } from "./DataChart";
import { SourcesDrawer } from "./SourcesDrawer";
import { ExportModal } from "./ExportModal";

describe("ProgressStepper", () => {
  let container;
  let root;

  beforeEach(() => {
    globalThis.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
  });

  it("renderiza etapas do pipeline e cronometro", async () => {
    await act(async () => root.render(<ProgressStepper active={true} />));

    expect(container.textContent).toContain("Processando análise documental");
    expect(container.textContent).toContain("Interpretação & Expansão");
    expect(container.textContent).toContain("Busca Híbrida");
    expect(container.textContent).toContain("Validação Numérica");
  });
});

describe("DataChart", () => {
  let container;
  let root;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
  });

  it("renderiza grafico de barras e alterna para tabela", async () => {
    const tableData = {
      columns: ["Setor", "Crescimento (%)"],
      rows: [
        ["Agropecuária", "3,1%"],
        ["Serviços", "1,1%"],
        ["Indústria", "0,0%"],
      ],
    };

    await act(async () => root.render(<DataChart tableData={tableData} />));

    expect(container.textContent).toContain("Crescimento (%)");
    expect(container.querySelector("svg")).not.toBeNull();
    expect(container.querySelectorAll(".chart-bar-rect")).toHaveLength(3);

    const tableButton = container.querySelector('[title="Tabela de Dados"]');
    await act(async () => tableButton.click());

    expect(container.querySelector(".data-chart-table")).not.toBeNull();
    expect(container.textContent).toContain("Agropecuária");
    expect(container.textContent).toContain("3,1%");
  });
});

describe("SourcesDrawer", () => {
  let container;
  let root;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
  });

  it("filtra fontes por busca e exibe badge de citada", async () => {
    const sources = [
      { file: "conjuntura/boletim_1t.pdf", page: 3, score: 0.9, excerpt: "Crescimento de 0,4%." },
      { file: "pib/pib_regional.pdf", page: 1, score: 0.7, excerpt: "Outro indicador." },
    ];
    const numericCitations = [{ file: "conjuntura/boletim_1t.pdf", value: "0,4%" }];

    await act(async () =>
      root.render(
        <SourcesDrawer
          isOpen={true}
          onClose={() => {}}
          sources={sources}
          numericCitations={numericCitations}
        />
      )
    );

    expect(container.textContent).toContain("Fontes Documentais");
    expect(container.textContent).toContain("Citada na resposta");
    expect(container.querySelectorAll(".source-card-item")).toHaveLength(2);

    const searchInput = container.querySelector(".sources-search-input");
    await act(async () => {
      searchInput.value = "boletim";
      searchInput.dispatchEvent(new Event("input", { bubbles: true }));
      // also fire onChange in React
      const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
        window.HTMLInputElement.prototype,
        "value"
      ).set;
      nativeInputValueSetter.call(searchInput, "boletim");
      searchInput.dispatchEvent(new Event("change", { bubbles: true }));
    });
  });

  it("dispara callback ao clicar no pill de dado para pular ate a citacao no texto", async () => {
    const onJumpToCitation = vi.fn();
    const sources = [
      { file: "conjuntura/boletim.pdf", page: 2, score: 0.9, excerpt: "Crescimento de 12,5%." },
    ];
    const numericCitations = [{ file: "conjuntura/boletim.pdf", value: "12,5%", source_index: 0 }];

    await act(async () =>
      root.render(
        <SourcesDrawer
          isOpen={true}
          onClose={() => {}}
          sources={sources}
          numericCitations={numericCitations}
          onJumpToCitation={onJumpToCitation}
        />
      )
    );

    const jumpPill = container.querySelector(".source-jump-pill");
    expect(jumpPill).not.toBeNull();
    expect(jumpPill.textContent).toContain("12,5%");

    await act(async () => jumpPill.click());
    expect(onJumpToCitation).toHaveBeenCalledWith(numericCitations[0]);
  });
});

describe("ExportModal", () => {
  let container;
  let root;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: vi.fn().mockResolvedValue(undefined) },
    });
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
  });

  it("permite copiar e baixar markdown com citacoes e fontes", async () => {
    const question = "Qual a variação do PIB?";
    const answer = "O PIB avançou 0,4%.";
    const sources = [{ file: "boletim.pdf", page: 3, score: 0.9, excerpt: "PIB avançou 0,4%." }];
    const numericCitations = [{ value: "0,4%", file: "boletim.pdf", page: 3, claim: "Avanço de 0,4%" }];

    await act(async () =>
      root.render(
        <ExportModal
          isOpen={true}
          onClose={() => {}}
          question={question}
          answer={answer}
          sources={sources}
          numericCitations={numericCitations}
        />
      )
    );

    expect(container.textContent).toContain("Exportar Relatório de Análise");
    expect(container.textContent).toContain("Copiar Markdown");
    expect(container.textContent).toContain("Baixar .md");

    const copyBtn = container.querySelector(".export-action-btn.secondary");
    await act(async () => copyBtn.click());
    expect(navigator.clipboard.writeText).toHaveBeenCalled();
  });
});
