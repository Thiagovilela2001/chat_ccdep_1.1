export const RAG_OPTIONS = {
  meta: {
    label: "Meta RAG",
    shortLabel: "Automático",
    description: "Seleciona a melhor metodologia para cada pergunta.",
    port: "8010",
    env: "VITE_META_URL",
  },
  principal: {
    label: "RAG Principal",
    shortLabel: "Principal",
    description: "Busca híbrida vetorial e lexical.",
    port: "8000",
    env: "VITE_PRINCIPAL_URL",
  },
  agentic: {
    label: "Agentic RAG",
    shortLabel: "Agentic",
    description: "Agente com ferramentas e decomposição de tarefas.",
    port: "8001",
    env: "VITE_AGENTIC_URL",
  },
  raptor: {
    label: "RAPTOR RAG",
    shortLabel: "RAPTOR",
    description: "Recuperação hierárquica para análises amplas.",
    port: "8002",
    env: "VITE_RAPTOR_URL",
  },
  selfrag: {
    label: "Self-RAG",
    shortLabel: "Self-RAG",
    description: "Recuperação com crítica e autocorreção.",
    port: "8003",
    env: "VITE_SELFRAG_URL",
  },
};

export function defaultApiUrl(
  option,
  location = globalThis.window?.location || { protocol: "http:", hostname: "127.0.0.1" },
) {
  const protocol = location.protocol === "https:" ? "https:" : "http:";
  const hostname = location.hostname || "127.0.0.1";
  return `${protocol}//${hostname}:${option.port}`;
}

export function apiUrls(overrides = {}, env = import.meta.env) {
  return Object.fromEntries(
    Object.entries(RAG_OPTIONS).map(([key, option]) => {
      const override = typeof overrides?.[key] === "string" ? overrides[key].trim() : "";
      const environmentUrl = typeof env?.[option.env] === "string" ? env[option.env].trim() : "";
      return [key, (override || environmentUrl || defaultApiUrl(option)).replace(/\/+$/, "")];
    }),
  );
}

async function errorDetail(response) {
  try {
    const payload = await response.json();
    if (payload?.detail) return String(payload.detail);
  } catch {
    // O corpo pode não ser JSON.
  }
  return `Falha na API (HTTP ${response.status}).`;
}

export async function checkHealth(baseUrl, { signal } = {}) {
  const response = await fetch(`${baseUrl}/health`, {
    headers: { Accept: "application/json" },
    cache: "no-store",
    signal,
  });
  if (!response.ok) throw new Error(await errorDetail(response));
  return response.json();
}

export async function queryBackend(baseUrl, question, apiKey = "", { signal } = {}) {
  const startedAt = performance.now();
  const headers = { Accept: "application/json", "Content-Type": "application/json" };
  if (apiKey.trim()) headers["x-api-key"] = apiKey.trim();

  const response = await fetch(`${baseUrl}/query`, {
    method: "POST",
    headers,
    body: JSON.stringify({ question }),
    signal,
  });
  if (!response.ok) {
    if (response.status === 401) {
      throw new Error("A API exige uma chave válida. Abra Configurações para informá-la.");
    }
    if (response.status === 429) {
      throw new Error("Limite de consultas atingido. Aguarde alguns instantes e tente novamente.");
    }
    throw new Error(await errorDetail(response));
  }

  const payload = await response.json();
  if (!payload || typeof payload.answer !== "string" || !payload.answer.trim()) {
    throw new Error("A API retornou uma resposta vazia ou incompatível.");
  }
  payload._client_roundtrip_ms ??= Math.round(performance.now() - startedAt);
  return payload;
}

export function isBackendReady(health) {
  if (!health) return false;
  if (typeof health.orchestrator_ready === "boolean") return health.orchestrator_ready;
  return Boolean(health.engine_ready);
}
