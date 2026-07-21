(function () {
  const STORAGE_KEYS = {
    apiBase: "ragFrontend.apiBase",
    apiKey: "ragFrontend.apiKey",
    history: "ragFrontend.history",
    ragType: "ragFrontend.ragType",
  };

  const RAG_OPTIONS = {
    principal: { label: "RAG Principal", port: "8000" },
    agentic: { label: "Agentic RAG", port: "8001" },
    raptor: { label: "RAPTOR RAG", port: "8002" },
    selfrag: { label: "Self-RAG", port: "8003" },
  };

  if (typeof marked !== "undefined") {
    marked.use({ gfm: true, breaks: false });
  }

  const elements = {
    activeRagLabel: document.getElementById("activeRagLabel"),
    apiBase: document.getElementById("apiBase"),
    apiKey: document.getElementById("apiKey"),
    answerMeta: document.getElementById("answerMeta"),
    chatApp: document.getElementById("chatApp"),
    cancelRequest: document.getElementById("cancelRequest"),
    clearHistory: document.getElementById("clearHistory"),
    composerPlus: document.querySelector(".composer-plus"),
    copyAnswer: document.getElementById("copyAnswer"),
    historyList: document.getElementById("historyList"),
    homeScreen: document.getElementById("homeScreen"),
    messages: document.getElementById("messages"),
    newChat: document.getElementById("newChat"),
    openChat: document.getElementById("openChat"),
    question: document.getElementById("question"),
    queryForm: document.getElementById("queryForm"),
    ragChoices: document.querySelectorAll("[data-rag-choice]"),
    ragType: document.getElementById("ragType"),
    refreshHealth: document.getElementById("refreshHealth"),
    rewrittenQuery: document.getElementById("rewrittenQuery"),
    saveApiBase: document.getElementById("saveApiBase"),
    sendButton: document.getElementById("sendButton"),
    sourceList: document.getElementById("sourceList"),
    sourcesUsed: document.getElementById("sourcesUsed"),
    statusDetail: document.getElementById("statusDetail"),
    statusDot: document.getElementById("statusDot"),
    statusLabel: document.getElementById("statusLabel"),
    unverifiedList: document.getElementById("unverifiedList"),
    validationSummary: document.getElementById("validationSummary"),
    voiceMode: document.getElementById("voiceMode"),
    workspace: document.querySelector(".workspace"),
  };

  const state = {
    activeController: null,
    currentAnswer: "",
    history: readHistory(),
  };

  function selectedRagType() {
    return elements.ragType.value in RAG_OPTIONS ? elements.ragType.value : "principal";
  }

  function selectedRagLabel() {
    return RAG_OPTIONS[selectedRagType()].label;
  }

  function apiBaseForRag(ragType) {
    const option = RAG_OPTIONS[ragType] || RAG_OPTIONS.principal;
    if (window.location.protocol.startsWith("http")) {
      return `${window.location.protocol}//${window.location.hostname}:${option.port}`;
    }
    return `http://127.0.0.1:${option.port}`;
  }

  function defaultApiBase() {
    return apiBaseForRag(selectedRagType());
  }

  function getApiBase() {
    return elements.apiBase.value.trim().replace(/\/+$/, "");
  }

  function requestHeaders(includeJson = false) {
    const headers = { Accept: "application/json" };
    const apiKey = elements.apiKey.value.trim();
    if (includeJson) {
      headers["Content-Type"] = "application/json";
    }
    if (apiKey) {
      headers["x-api-key"] = apiKey;
    }
    return headers;
  }

  const ALLOWED_MARKDOWN_TAGS = new Set([
    "A", "BLOCKQUOTE", "BR", "CODE", "DEL", "EM", "H1", "H2", "H3",
    "H4", "H5", "H6", "HR", "LI", "OL", "P", "PRE", "STRONG",
    "TABLE", "TBODY", "TD", "TH", "THEAD", "TR", "UL",
  ]);

  function safeLink(href) {
    if (!href) return null;
    if (href.startsWith("#")) return href;
    try {
      const url = new URL(href, window.location.origin);
      return ["http:", "https:", "mailto:"].includes(url.protocol) ? url.href : null;
    } catch {
      return null;
    }
  }

  function sanitizeHtml(html) {
    const template = document.createElement("template");
    template.innerHTML = String(html || "");
    const nodes = Array.from(template.content.querySelectorAll("*"));
    nodes.forEach((node) => {
      if (!ALLOWED_MARKDOWN_TAGS.has(node.tagName)) {
        node.replaceWith(document.createTextNode(node.textContent || ""));
        return;
      }

      const originalHref = node.tagName === "A" ? node.getAttribute("href") : null;
      Array.from(node.attributes).forEach((attribute) => {
        node.removeAttribute(attribute.name);
      });

      if (node.tagName === "A") {
        const href = safeLink(originalHref);
        if (href) {
          node.setAttribute("href", href);
          node.setAttribute("rel", "noopener noreferrer");
        }
      }
    });
    return template.innerHTML;
  }

  function renderMarkdown(text) {
    if (typeof marked === "undefined") {
      const fallback = document.createElement("p");
      fallback.textContent = String(text || "");
      return fallback.outerHTML;
    }
    return sanitizeHtml(marked.parse(String(text || "")));
  }

  function readHistory() {
    try {
      const stored = JSON.parse(localStorage.getItem(STORAGE_KEYS.history) || "[]");
      return Array.isArray(stored) ? stored.slice(0, 10) : [];
    } catch {
      return [];
    }
  }

  function writeHistory() {
    localStorage.setItem(STORAGE_KEYS.history, JSON.stringify(state.history.slice(0, 10)));
  }

  function formatTime(value) {
    return new Intl.DateTimeFormat("pt-BR", {
      day: "2-digit",
      month: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    }).format(new Date(value));
  }

  function truncate(text, size) {
    if (!text) {
      return "";
    }
    return text.length > size ? `${text.slice(0, size - 1)}...` : text;
  }

  function clearElement(element) {
    while (element.firstChild) {
      element.removeChild(element.firstChild);
    }
  }

  function makeId() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return window.crypto.randomUUID();
    }
    return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  function setStatus(stateName, label, detail) {
    elements.statusDot.dataset.state = stateName;
    elements.statusLabel.textContent = label;
    elements.statusDetail.textContent = detail;
  }

  function updateActiveRagLabel(label = selectedRagLabel()) {
    if (elements.activeRagLabel) {
      elements.activeRagLabel.textContent = label;
    }
    elements.ragChoices.forEach((button) => {
      button.classList.toggle("is-active", button.dataset.ragChoice === selectedRagType());
    });
  }

  function applyRagSelection(ragType) {
    if (ragType && RAG_OPTIONS[ragType]) {
      elements.ragType.value = ragType;
    }
    localStorage.setItem(STORAGE_KEYS.ragType, selectedRagType());
    elements.apiBase.value = defaultApiBase();
    localStorage.setItem(STORAGE_KEYS.apiBase, getApiBase());
    updateActiveRagLabel();
    checkHealth();
  }

  function openChatView() {
    elements.homeScreen.classList.add("is-hidden");
    elements.chatApp.classList.remove("is-hidden");
    window.setTimeout(() => elements.question.focus(), 80);
  }

  async function fetchWithTimeout(url, options, timeoutMs) {
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
    try {
      return await fetch(url, {
        ...options,
        signal: controller.signal,
      });
    } finally {
      window.clearTimeout(timeoutId);
    }
  }

  async function checkHealth() {
    const apiBase = getApiBase();
    if (!apiBase) {
      setStatus("offline", "Endpoint vazio", "Informe a URL da API");
      return;
    }

    setStatus("checking", "Verificando", `Conectando ao ${selectedRagLabel()}`);

    try {
      const response = await fetchWithTimeout(
        `${apiBase}/health`,
        {
          headers: requestHeaders(),
          cache: "no-store",
        },
        5000
      );
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const payload = await response.json();
      if (payload.engine_ready) {
        updateActiveRagLabel(payload.rag_label || selectedRagLabel());
        setStatus("ready", "Online", `${payload.rag_label || selectedRagLabel()} carregado`);
      } else {
        updateActiveRagLabel(payload.rag_label || selectedRagLabel());
        setStatus("checking", "Inicializando", `${payload.rag_label || selectedRagLabel()} ainda carregando`);
      }
    } catch (error) {
      updateActiveRagLabel();
      setStatus("offline", "Offline", error.name === "AbortError" ? "Tempo esgotado" : "Sem resposta da API");
    }
  }

  function appendMessage(role, text, options = {}) {
    const message = document.createElement("article");
    message.className = `message ${role === "user" ? "user-message" : "assistant-message"}`;
    if (options.loading) {
      message.classList.add("is-loading");
    }

    const meta = document.createElement("div");
    meta.className = "message-meta";
    meta.textContent = role === "user" ? "Você" : "Assistente";

    const body = document.createElement("div");
    body.className = "message-body";
    if (options.markdown) {
      body.innerHTML = renderMarkdown(text);
    } else {
      body.textContent = text;
    }

    message.append(meta, body);
    elements.messages.appendChild(message);
    elements.messages.scrollTop = elements.messages.scrollHeight;
    return { message, body };
  }

  function renderEmptyState() {
    const message = document.createElement("article");
    message.className = "message assistant-message empty-state";

    const meta = document.createElement("div");
    meta.className = "message-meta";
    meta.textContent = "Consulta CCDEP";

    const body = document.createElement("div");
    body.className = "message-body";

    const title = document.createElement("h2");
    title.textContent = "Tudo pronto? Então vamos lá.";

    const description = document.createElement("p");
    description.textContent = "Pergunte sobre emprego, PIB, indústria, serviços, comércio ou indicadores econômicos de São Paulo.";

    body.append(title, description);
    message.append(meta, body);
    elements.messages.appendChild(message);
  }

  function resetConversation(options = {}) {
    clearElement(elements.messages);
    if (options.empty) {
      renderEmptyState();
    }
  }

  function resetInspector() {
    state.currentAnswer = "";
    elements.workspace.classList.remove("has-response");
    elements.rewrittenQuery.textContent = "Nenhuma consulta enviada.";
    elements.answerMeta.textContent = "Aguardando consulta.";
    elements.copyAnswer.disabled = true;
    clearElement(elements.sourcesUsed);
    clearElement(elements.sourceList);
    clearElement(elements.validationSummary);
    clearElement(elements.unverifiedList);

    const verified = document.createElement("strong");
    verified.textContent = "0/0";
    const label = document.createElement("span");
    label.textContent = "números verificados";
    elements.validationSummary.append(verified, label);
  }

  function renderInspector(response, elapsedMs) {
    const validation = response.validation || { verified: 0, total: 0, unverified: [] };
    const citationValidation = response.citation_validation || {
      verified: 0, total: 0, unverified: []
    };
    state.currentAnswer = response.answer || "";
    elements.workspace.classList.add("has-response");

    elements.rewrittenQuery.textContent = response.rewritten_query || "Sem reescrita informada.";
    const ragLabel = response.rag_label || selectedRagLabel();
    elements.answerMeta.textContent = elapsedMs
      ? `${ragLabel} respondeu em ${(elapsedMs / 1000).toFixed(1)}s.`
      : `Resposta carregada do histórico (${ragLabel}).`;
    elements.copyAnswer.disabled = !state.currentAnswer;

    clearElement(elements.sourcesUsed);
    const sourcesUsed = Array.isArray(response.sources_used) ? response.sources_used : [];
    if (sourcesUsed.length === 0) {
      const empty = document.createElement("span");
      empty.className = "muted-text";
      empty.textContent = "Nenhuma fonte classificada.";
      elements.sourcesUsed.appendChild(empty);
    } else {
      sourcesUsed.forEach((source) => {
        const tag = document.createElement("span");
        tag.className = "tag";
        tag.textContent = source;
        elements.sourcesUsed.appendChild(tag);
      });
    }

    clearElement(elements.sourceList);
    const sourceList = Array.isArray(response.sources) ? response.sources : [];
    sourceList.forEach((source) => {
      const item = document.createElement("li");
      const name = document.createElement("span");
      name.className = "source-name";
      const sourcePage = source.page === null || source.page === undefined ? "" : ` · p./aba ${source.page}`;
      name.textContent = `${source.file || "Fonte sem nome"}${sourcePage}`;
      const score = document.createElement("span");
      score.className = "source-score";
      const numericScore = Number(source.score);
      score.textContent = Number.isFinite(numericScore) ? numericScore.toFixed(2) : "0.00";
      item.append(name, score);
      elements.sourceList.appendChild(item);
    });

    clearElement(elements.validationSummary);
    const verified = document.createElement("strong");
    verified.textContent = `${validation.verified || 0}/${validation.total || 0}`;
    const label = document.createElement("span");
    label.textContent = "números verificados";
    elements.validationSummary.append(verified, label);
    if (citationValidation.total) {
      const cited = document.createElement("strong");
      cited.textContent = `${citationValidation.verified || 0}/${citationValidation.total}`;
      const citedLabel = document.createElement("span");
      citedLabel.textContent = "citações verificadas";
      elements.validationSummary.append(cited, citedLabel);
    }

    clearElement(elements.unverifiedList);
    const unverified = Array.isArray(validation.unverified) ? validation.unverified : [];
    unverified.forEach((value) => {
      const item = document.createElement("li");
      item.textContent = value;
      elements.unverifiedList.appendChild(item);
    });
    const unverifiedCitations = Array.isArray(citationValidation.unverified)
      ? citationValidation.unverified : [];
    unverifiedCitations.forEach((value) => {
      const item = document.createElement("li");
      item.textContent = `Citação: ${value}`;
      elements.unverifiedList.appendChild(item);
    });
  }

  function renderHistory() {
    clearElement(elements.historyList);
    if (state.history.length === 0) {
      const item = document.createElement("li");
      const empty = document.createElement("p");
      empty.className = "muted-text";
      empty.textContent = "Sem consultas salvas.";
      item.appendChild(empty);
      elements.historyList.appendChild(item);
      return;
    }

    state.history.forEach((entry) => {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = truncate(entry.question, 78);
      const time = document.createElement("small");
      time.textContent = `${formatTime(entry.createdAt)} - ${entry.ragLabel || "RAG"}`;
      button.appendChild(time);
      button.addEventListener("click", () => {
        if (entry.ragType && RAG_OPTIONS[entry.ragType]) {
          elements.ragType.value = entry.ragType;
          localStorage.setItem(STORAGE_KEYS.ragType, entry.ragType);
        }
        if (entry.apiBase) {
          elements.apiBase.value = entry.apiBase;
          localStorage.setItem(STORAGE_KEYS.apiBase, entry.apiBase);
        }
        resetConversation();
        appendMessage("user", entry.question);
        appendMessage("assistant", entry.response.answer || "Sem resposta.", { markdown: true });
        renderInspector(entry.response);
        elements.question.value = entry.question;
      });
      item.appendChild(button);
      elements.historyList.appendChild(item);
    });
  }

  function saveToHistory(question, response) {
    state.history = [
      {
        id: makeId(),
        question,
        response,
        ragType: response.rag_type || selectedRagType(),
        ragLabel: response.rag_label || selectedRagLabel(),
        apiBase: getApiBase(),
        createdAt: new Date().toISOString(),
      },
      ...state.history.filter((entry) => entry.question !== question),
    ].slice(0, 10);
    writeHistory();
    renderHistory();
  }

  async function readError(response) {
    try {
      const payload = await response.json();
      return payload.detail || `HTTP ${response.status}`;
    } catch {
      return `HTTP ${response.status}`;
    }
  }

  async function submitQuestion(question) {
    const apiBase = getApiBase();
    const controller = new AbortController();
    const startedAt = performance.now();
    state.activeController = controller;

    elements.sendButton.disabled = true;
    elements.cancelRequest.hidden = false;

    const pending = appendMessage("assistant", `Analisando documentos e tabelas com ${selectedRagLabel()}`, { loading: true });

    try {
      const response = await fetch(`${apiBase}/query`, {
        method: "POST",
        headers: requestHeaders(true),
        body: JSON.stringify({ question }),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(await readError(response));
      }

      const payload = await response.json();
      const elapsedMs = performance.now() - startedAt;
      pending.message.classList.remove("is-loading");
      pending.body.innerHTML = renderMarkdown(
        payload.answer || "A API retornou uma resposta vazia."
      );
      renderInspector(payload, elapsedMs);
      saveToHistory(question, payload);
      checkHealth();
    } catch (error) {
      pending.message.classList.remove("is-loading");
      pending.body.textContent = error.name === "AbortError" ? "Consulta cancelada." : `Erro: ${error.message}`;
    } finally {
      state.activeController = null;
      elements.sendButton.disabled = false;
      elements.cancelRequest.hidden = true;
    }
  }

  elements.queryForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const question = elements.question.value.trim();
    if (!question || state.activeController) {
      return;
    }
    resetConversation();
    resetInspector();
    appendMessage("user", question);
    submitQuestion(question);
  });

  elements.cancelRequest.addEventListener("click", () => {
    if (state.activeController) {
      state.activeController.abort();
    }
  });

  elements.refreshHealth.addEventListener("click", checkHealth);

  elements.saveApiBase.addEventListener("click", () => {
    localStorage.setItem(STORAGE_KEYS.ragType, selectedRagType());
    localStorage.setItem(STORAGE_KEYS.apiBase, getApiBase());
    sessionStorage.setItem(STORAGE_KEYS.apiKey, elements.apiKey.value.trim());
    checkHealth();
  });

  elements.ragType.addEventListener("change", () => {
    applyRagSelection();
  });

  elements.ragChoices.forEach((button) => {
    button.addEventListener("click", () => {
      applyRagSelection(button.dataset.ragChoice);
    });
  });

  elements.question.addEventListener("input", () => {
    elements.question.style.height = "auto";
    elements.question.style.height = `${elements.question.scrollHeight}px`;
  });

  elements.newChat.addEventListener("click", () => {
    if (state.activeController) {
      state.activeController.abort();
    }
    elements.question.value = "";
    elements.question.style.height = "";
    resetConversation({ empty: true });
    resetInspector();
    elements.question.focus();
  });

  elements.openChat.addEventListener("click", openChatView);

  elements.voiceMode.addEventListener("click", () => {
    openChatView();
    elements.question.placeholder = "Modo de voz ainda não está ativo. Digite sua pergunta aqui.";
  });

  elements.composerPlus.addEventListener("click", () => {
    elements.question.focus();
  });

  elements.clearHistory.addEventListener("click", () => {
    state.history = [];
    writeHistory();
    renderHistory();
  });

  elements.copyAnswer.addEventListener("click", async () => {
    if (!state.currentAnswer) {
      return;
    }
    try {
      await navigator.clipboard.writeText(state.currentAnswer);
    } catch {
      const textarea = document.createElement("textarea");
      textarea.value = state.currentAnswer;
      textarea.setAttribute("readonly", "");
      textarea.style.position = "fixed";
      textarea.style.left = "-9999px";
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
    }
    elements.copyAnswer.textContent = "Copiado";
    window.setTimeout(() => {
      elements.copyAnswer.textContent = "Copiar";
    }, 1300);
  });

  document.querySelectorAll("[data-prompt]").forEach((button) => {
    button.addEventListener("click", () => {
      elements.question.value = button.dataset.prompt || "";
      elements.question.focus();
    });
  });

  const savedRagType = localStorage.getItem(STORAGE_KEYS.ragType);
  if (savedRagType && RAG_OPTIONS[savedRagType]) {
    elements.ragType.value = savedRagType;
  }
  elements.apiBase.value = localStorage.getItem(STORAGE_KEYS.apiBase) || defaultApiBase();
  elements.apiKey.value = sessionStorage.getItem(STORAGE_KEYS.apiKey) || "";
  updateActiveRagLabel();
  renderHistory();
  checkHealth();
})();

(function initSphere() {
  const canvas = document.getElementById("sphereCanvas");
  if (!canvas || !canvas.getContext) return;

  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const S = 430;

  canvas.width = S * dpr;
  canvas.height = S * dpr;
  ctx.scale(dpr, dpr);

  const N = 220;
  const golden = (1 + Math.sqrt(5)) / 2;
  const pts = [];
  for (let i = 0; i < N; i++) {
    const theta = Math.acos(1 - 2 * (i + 0.5) / N);
    const phi = 2 * Math.PI * i / golden;
    pts.push([
      Math.sin(theta) * Math.cos(phi),
      Math.sin(theta) * Math.sin(phi),
      Math.cos(theta),
    ]);
  }

  const cx = S / 2, cy = S / 2;
  const R = S * 0.41;
  const TILT = 0.28;

  const grad = ctx.createRadialGradient(
    cx - R * 0.22, cy - R * 0.26, R * 0.08,
    cx, cy, R
  );
  grad.addColorStop(0,    "rgba(255,255,255,0.95)");
  grad.addColorStop(0.15, "rgba(127,184,255,1)");
  grad.addColorStop(0.50, "rgba(0,93,168,0.95)");
  grad.addColorStop(0.80, "rgba(22,50,86,1)");
  grad.addColorStop(1,    "rgba(8,18,42,1)");

  const ring = ctx.createRadialGradient(cx, cy, R * 0.82, cx, cy, R * 1.18);
  ring.addColorStop(0,   "rgba(22,50,86,0)");
  ring.addColorStop(0.5, "rgba(22,50,86,0.65)");
  ring.addColorStop(1,   "rgba(22,50,86,0)");

  let angle = 0;
  const cosX = Math.cos(TILT);
  const sinX = Math.sin(TILT);
  const R2 = R * R;

  function render() {
    ctx.clearRect(0, 0, S, S);

    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();

    ctx.beginPath();
    ctx.arc(cx, cy, R * 1.18, 0, Math.PI * 2);
    ctx.fillStyle = ring;
    ctx.fill();

    const cosY = Math.cos(angle);
    const sinY = Math.sin(angle);
    const visible = [];

    for (let i = 0; i < pts.length; i++) {
      const [px0, py0, pz0] = pts[i];
      const x1 = px0 * cosY + pz0 * sinY;
      const z1 = -px0 * sinY + pz0 * cosY;
      const y2 = py0 * cosX - z1 * sinX;
      const z2 = py0 * sinX + z1 * cosX;
      const sx = cx + x1 * R;
      const sy = cy + y2 * R;
      const dx = sx - cx, dy = sy - cy;
      if (dx * dx + dy * dy <= R2) {
        visible.push([sx, sy, z2]);
      }
    }

    visible.sort((a, b) => a[2] - b[2]);

    for (const [sx, sy, depth] of visible) {
      const t = (depth + 1) / 2;
      ctx.beginPath();
      ctx.arc(sx, sy, 0.6 + 2.2 * t, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255,255,255,${(0.1 + 0.82 * t).toFixed(2)})`;
      ctx.fill();
    }

    angle += 0.006;
    requestAnimationFrame(render);
  }

  render();
}());
