import { describe, expect, it } from "vitest";

import { upsertConversation, validStoredConversations } from "./conversations";

const message = (id, role, content) => ({
  id,
  role,
  content,
  createdAt: `2026-08-24T10:0${id}:00.000Z`,
});

describe("histórico de conversas", () => {
  it("agrupa várias perguntas na mesma conversa", () => {
    const messages = [
      message("1", "user", "Quais municípios perderam população?"),
      message("2", "assistant", "Resposta inicial"),
      message("3", "user", "E quais ganharam população?"),
    ];

    const first = upsertConversation([], "conversation-1", messages.slice(0, 1));
    const updated = upsertConversation(first, "conversation-1", messages);

    expect(updated).toHaveLength(1);
    expect(updated[0].title).toBe("Quais municípios perderam população?");
    expect(updated[0].messages).toEqual(messages);
  });

  it("cria outro item somente para outro identificador de conversa", () => {
    const first = upsertConversation(
      [],
      "conversation-1",
      [message("1", "user", "Primeira análise")],
    );
    const second = upsertConversation(
      first,
      "conversation-2",
      [message("2", "user", "Segunda análise")],
    );

    expect(second.map(({ id }) => id)).toEqual(["conversation-2", "conversation-1"]);
  });

  it("não cria histórico para conversa vazia", () => {
    expect(upsertConversation([], "conversation-1", [])).toEqual([]);
  });

  it("rejeita histórico persistido inválido", () => {
    expect(validStoredConversations([])).toBe(true);
    expect(validStoredConversations([{ id: "1", title: "Sem mensagens" }])).toBe(false);
  });
});
