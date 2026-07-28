// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";

describe("inicialização da interface", () => {
  let container;
  let root;

  beforeEach(() => {
    globalThis.IS_REACT_ACT_ENVIRONMENT = true;
    localStorage.clear();
    sessionStorage.clear();
    globalThis.fetch = vi.fn(() => Promise.reject(new TypeError("API indisponível")));
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
});
