export const MAX_STORED_CONVERSATIONS = 12;
export const MAX_STORED_MESSAGES = 40;

export function validStoredConversations(value) {
  return (
    Array.isArray(value)
    && value.every((conversation) => (
      conversation !== null
      && typeof conversation === "object"
      && typeof conversation.id === "string"
      && typeof conversation.title === "string"
      && Array.isArray(conversation.messages)
    ))
  );
}

function normalizedTitle(content) {
  return String(content || "")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 120);
}

export function upsertConversation(conversations, conversationId, messages) {
  const current = Array.isArray(conversations) ? conversations : [];
  const safeMessages = Array.isArray(messages) ? messages : [];
  const firstQuestion = safeMessages.find(
    (message) => message?.role === "user" && normalizedTitle(message.content),
  );

  if (!conversationId || !firstQuestion) return current;

  const existing = current.find((conversation) => conversation.id === conversationId);
  const lastMessage = safeMessages.at(-1);
  const conversation = {
    id: conversationId,
    title: existing?.title || normalizedTitle(firstQuestion.content),
    createdAt: existing?.createdAt || firstQuestion.createdAt || new Date().toISOString(),
    updatedAt: lastMessage?.createdAt || existing?.updatedAt || new Date().toISOString(),
    messages: safeMessages.slice(-MAX_STORED_MESSAGES),
  };

  return [
    conversation,
    ...current.filter((item) => item.id !== conversationId),
  ].slice(0, MAX_STORED_CONVERSATIONS);
}
