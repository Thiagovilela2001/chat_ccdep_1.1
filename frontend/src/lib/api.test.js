import { describe, expect, it } from "vitest";

import { apiUrls, defaultApiUrl, isBackendReady, RAG_OPTIONS } from "./api";

describe("configuração da API", () => {
  it("monta endpoints locais usando hostname e porta da metodologia", () => {
    expect(
      defaultApiUrl(RAG_OPTIONS.principal, {
        protocol: "http:",
        hostname: "localhost",
      }),
    ).toBe("http://localhost:8000");
  });

  it("prioriza overrides e remove barras finais", () => {
    const urls = apiUrls(
      { meta: "https://rag.example/meta///" },
      { VITE_PRINCIPAL_URL: "https://rag.example/principal/" },
    );
    expect(urls.meta).toBe("https://rag.example/meta");
    expect(urls.principal).toBe("https://rag.example/principal");
  });

  it("normaliza readiness de orquestrador e engines", () => {
    expect(isBackendReady({ orchestrator_ready: true })).toBe(true);
    expect(isBackendReady({ engine_ready: true })).toBe(true);
    expect(isBackendReady({ engine_ready: false })).toBe(false);
  });
});
