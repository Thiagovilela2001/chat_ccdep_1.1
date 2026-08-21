import { describe, expect, it, vi } from "vitest";

import { apiUrls, defaultApiUrl, isBackendReady, queryBackend, RAG_OPTIONS } from "./api";

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

  it("ignora endpoints salvos com formato inválido", () => {
    const urls = apiUrls(
      { meta: 8010, principal: null },
      {},
    );
    expect(urls.meta).toBe("http://127.0.0.1:8010");
    expect(urls.principal).toBe("http://127.0.0.1:8000");
  });

  it("normaliza readiness de orquestrador e engines", () => {
    expect(isBackendReady({ orchestrator_ready: true })).toBe(true);
    expect(isBackendReady({ engine_ready: true })).toBe(true);
    expect(isBackendReady({ engine_ready: false })).toBe(false);
  });

  it("envia identificador e histórico da conversa", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: async () => ({ answer: "Resposta contextual." }),
    });

    await queryBackend("https://rag.example", "E em 2023?", "", {
      conversationId: "conversation_123456",
      history: [{ role: "user", content: "Qual foi o PIB em 2024?" }],
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "https://rag.example/query",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          question: "E em 2023?",
          conversation_id: "conversation_123456",
          history: [{ role: "user", content: "Qual foi o PIB em 2024?" }],
        }),
      }),
    );
    fetchMock.mockRestore();
  });
});
