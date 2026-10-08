// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

const markdownRenders = vi.hoisted(() => ({ count: 0 }));
vi.mock("react-markdown", () => ({
  default: ({ children }) => {
    markdownRenders.count += 1;
    return <div>{children}</div>;
  },
}));

import App from "./App";

let container;
let root;

beforeEach(() => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  localStorage.clear();
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

it("digitar no composer não re-renderiza as mensagens da conversa", async () => {
  localStorage.setItem("nadia.messages.v1", JSON.stringify([
    { id: "u1", role: "user", content: "Pergunta", createdAt: "2026-10-06T10:00:00Z" },
    { id: "a1", role: "assistant", content: "**Resposta**", createdAt: "2026-10-06T10:00:10Z", meta: { answer: "x" } },
  ]));
  await act(async () => root.render(<App />));
  const rendersBeforeTyping = markdownRenders.count;
  expect(rendersBeforeTyping).toBeGreaterThan(0);

  const textarea = container.querySelector("textarea");
  const setValue = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value").set;
  for (const value of ["a", "ab", "abc"]) {
    await act(async () => {
      setValue.call(textarea, value);
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
    });
  }

  expect(textarea.value).toBe("abc");
  expect(markdownRenders.count).toBe(rendersBeforeTyping);
});
