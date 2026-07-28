import { describe, expect, it, vi } from "vitest";

import { readStorage, readStoredJson, writeStorage } from "./storage";

describe("armazenamento resiliente", () => {
  it("descarta JSON inválido e valores com formato inesperado", () => {
    const invalidJson = { getItem: () => "{" };
    const wrongShape = { getItem: () => '{"message":"não é uma lista"}' };

    expect(readStoredJson(invalidJson, "key", [], Array.isArray)).toEqual([]);
    expect(readStoredJson(wrongShape, "key", [], Array.isArray)).toEqual([]);
  });

  it("usa o fallback quando o navegador bloqueia o storage", () => {
    const blocked = {
      getItem: () => {
        throw new DOMException("Storage bloqueado", "SecurityError");
      },
    };

    expect(readStorage(blocked, "key", "fallback")).toBe("fallback");
    expect(readStoredJson(blocked, "key", [])).toEqual([]);
  });

  it("não derruba a aplicação se uma gravação falhar", () => {
    const blocked = { setItem: vi.fn(() => { throw new Error("quota"); }) };

    expect(() => writeStorage(blocked, "key", "value")).not.toThrow();
  });
});

