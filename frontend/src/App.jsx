import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Menu,
  Moon,
  PanelLeftOpen,
  Sun,
} from "lucide-react";

import {
  ACTIVE_RAG_TYPE,
  apiUrls,
  checkHealth,
  isBackendReady,
  queryBackend,
} from "./lib/api";
import { upsertConversation, validStoredConversations } from "./lib/conversations";
import { readStorage, readStoredJson, writeStorage } from "./lib/storage";

import { Composer } from "./components/Composer";
import { EmptyState } from "./components/EmptyState";
import { ExportModal } from "./components/ExportModal";
import { Inspector } from "./components/Inspector";
import { LoadingMessage, Message } from "./components/Message";
import { SettingsDialog } from "./components/SettingsDialog";
import { Sidebar } from "./components/Sidebar";
import { SourcesDrawer } from "./components/SourcesDrawer";

const STORAGE = {
  messages: "nadia.messages.v1",
  conversations: "nadia.conversations.v1",
  theme: "nadia.theme.v1",
  endpoints: "nadia.endpoints.v1",
  apiKey: "nadia.apiKey.v1",
  conversationId: "nadia.conversationId.v1",
};

const ASSISTANT_MODE_LABEL = "Análise documental";

const CLIENT_DIAGNOSTIC_URL = "/client-error";

function reportClientDiagnostic(detail) {
  fetch(CLIENT_DIAGNOSTIC_URL, {
    method: "POST",
    mode: "no-cors",
    headers: { "Content-Type": "text/plain" },
    body: JSON.stringify({
      ...detail,
      href: window.location.href,
      occurredAt: new Date().toISOString(),
    }),
  }).catch(() => {});
}

const uid = () =>
  globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;

function storedMessages() {
  return readStoredJson(globalThis.localStorage, STORAGE.messages, [], Array.isArray);
}

function storedConversations() {
  return readStoredJson(
    globalThis.localStorage,
    STORAGE.conversations,
    [],
    validStoredConversations,
  );
}

function storedTheme() {
  const value = readStorage(globalThis.localStorage, STORAGE.theme, "");
  if (value === "dark" || value === "light") return value;
  return globalThis.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function storedEndpoints() {
  return readStoredJson(
    globalThis.localStorage,
    STORAGE.endpoints,
    {},
    (value) => value !== null && typeof value === "object" && !Array.isArray(value),
  );
}

export default function App() {
  const [messages, setMessages] = useState(storedMessages);
  const [conversations, setConversations] = useState(storedConversations);
  const [theme, setTheme] = useState(storedTheme);
  const [endpointOverrides, setEndpointOverrides] = useState(storedEndpoints);
  const [apiKey, setApiKey] = useState(() => readStorage(globalThis.sessionStorage, STORAGE.apiKey));
  const [conversationId, setConversationId] = useState(
    () => readStorage(globalThis.localStorage, STORAGE.conversationId) || uid(),
  );
  const [developerMode, setDeveloperMode] = useState(false);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState(null);
  const [healthState, setHealthState] = useState("checking");
  const [healthError, setHealthError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => {
    try {
      return localStorage.getItem("nadia.sidebar.collapsed") === "true";
    } catch {
      return false;
    }
  });

  const toggleSidebarCollapse = useCallback(() => {
    setSidebarCollapsed((prev) => {
      const next = !prev;
      try {
        localStorage.setItem("nadia.sidebar.collapsed", String(next));
      } catch {
        // A persistência é opcional; a interação continua válida nesta sessão.
      }
      return next;
    });
  }, []);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "b") {
        e.preventDefault();
        toggleSidebarCollapse();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [toggleSidebarCollapse]);

  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [inspectorTab, setInspectorTab] = useState("sources");
  const [selectedMeta, setSelectedMeta] = useState(null);
  const [sourceRequest, setSourceRequest] = useState(null);
  const [exportModalState, setExportModalState] = useState(null);
  const [sourcesDrawerState, setSourcesDrawerState] = useState(null);
  const activeController = useRef(null);
  const messageScroll = useRef(null);
  // Falso quando o usuário rolou para cima para ler; o auto-scroll não o puxa de volta.
  const stickToBottom = useRef(true);
  const endpoints = useMemo(() => apiUrls(endpointOverrides), [endpointOverrides]);
  const activeUrl = endpoints[ACTIVE_RAG_TYPE];

  const handleExport = useCallback((message) => {
    const messageIndex = messages.indexOf(message);
    const userMsg = messages.slice(0, messageIndex).reverse().find((m) => m.role === "user");
    setExportModalState({
      isOpen: true,
      question: userMsg?.content || "Consulta Documental",
      answer: message.content,
      sources: message.meta?.sources || [],
      numericCitations: message.meta?.numeric_citations || [],
    });
  }, [messages]);

  const handleOpenDrawer = useCallback((meta) => {
    setSourcesDrawerState({
      isOpen: true,
      sources: meta?.sources || [],
      numericCitations: meta?.numeric_citations || [],
    });
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  // Só persiste a escolha explícita; sem ela, segue o tema do sistema.
  function toggleTheme() {
    const next = theme === "light" ? "dark" : "light";
    setTheme(next);
    writeStorage(globalThis.localStorage, STORAGE.theme, next);
  }

  useEffect(() => writeStorage(globalThis.localStorage, STORAGE.endpoints, JSON.stringify(endpointOverrides)), [endpointOverrides]);
  useEffect(() => writeStorage(globalThis.sessionStorage, STORAGE.apiKey, apiKey), [apiKey]);
  useEffect(() => writeStorage(globalThis.localStorage, STORAGE.conversationId, conversationId), [conversationId]);
  useEffect(() => writeStorage(globalThis.localStorage, STORAGE.messages, JSON.stringify(messages.slice(-40))), [messages]);
  useEffect(() => {
    setConversations((current) => upsertConversation(current, conversationId, messages));
  }, [conversationId, messages]);
  useEffect(() => {
    writeStorage(globalThis.localStorage, STORAGE.conversations, JSON.stringify(conversations));
  }, [conversations]);
  useEffect(() => {
    const container = messageScroll.current;
    if (!container || (messages.length === 0 && !loading) || !stickToBottom.current) return undefined;
    const frame = window.requestAnimationFrame(() => {
      container.scrollTo({ top: container.scrollHeight, behavior: "smooth" });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [messages, loading]);

  const refreshHealth = useCallback(async () => {
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), 6000);
    setHealthState("checking");
    setHealthError("");
    try {
      const payload = await checkHealth(activeUrl, { signal: controller.signal });
      const nextState = isBackendReady(payload) ? "ready" : "initializing";
      setHealth(payload);
      setHealthState(nextState);
    } catch (error) {
      const detail = error?.name === "AbortError"
        ? `Tempo esgotado ao consultar ${activeUrl}/health`
        : `Falha de rede ou CORS em ${activeUrl}/health: ${error?.message || "sem detalhes"}`;
      setHealth(null);
      setHealthState("offline");
      setHealthError(detail);
      reportClientDiagnostic({
        kind: "health-check",
        activeUrl,
        result: "offline",
        errorName: error?.name,
        errorMessage: error?.message,
      });
    } finally {
      window.clearTimeout(timer);
    }
  }, [activeUrl]);

  useEffect(() => {
    refreshHealth();
    const interval = window.setInterval(refreshHealth, 30000);
    return () => window.clearInterval(interval);
  }, [refreshHealth]);

  useEffect(() => {
    function shortcuts(event) {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        newConversation();
      }
      if (event.key === "Escape") {
        setSettingsOpen(false);
        setSidebarOpen(false);
      }
    }
    window.addEventListener("keydown", shortcuts);
    return () => window.removeEventListener("keydown", shortcuts);
  }, []);

  function newConversation() {
    activeController.current?.abort();
    setMessages([]);
    setSelectedMeta(null);
    setInspectorOpen(false);
    setSourceRequest(null);
    setInput("");
    setLoading(false);
    setConversationId(uid());
  }

  function openConversation(conversation) {
    activeController.current?.abort();
    const restoredMessages = Array.isArray(conversation?.messages) ? conversation.messages : [];
    stickToBottom.current = true;
    setConversationId(conversation.id);
    setMessages(restoredMessages);
    setSelectedMeta(null);
    setInspectorOpen(false);
    setSourceRequest(null);
    setInput("");
    setLoading(false);
    setSidebarOpen(false);
  }

  function clearHistory() {
    activeController.current?.abort();
    setConversations([]);
    setMessages([]);
    setSelectedMeta(null);
    setInspectorOpen(false);
    setSourceRequest(null);
    setInput("");
    setLoading(false);
    setConversationId(uid());
  }

  // `baseMessages` permite reenviar descartando a tentativa que falhou (ver retryQuestion).
  async function submitQuestion(value, baseMessages = messages) {
    const question = value.trim();
    if (!question || loading) return;
    const userMessage = { id: uid(), role: "user", content: question, createdAt: new Date().toISOString() };
    stickToBottom.current = true;
    setMessages([...baseMessages, userMessage]);
    setInput("");
    setLoading(true);
    setSidebarOpen(false);
    const controller = new AbortController();
    activeController.current = controller;
    try {
      const history = baseMessages
        .filter((message) => !message.error && ["user", "assistant"].includes(message.role))
        .slice(-12)
        .map(({ role, content }) => ({ role, content: String(content).slice(0, 4000) }));
      const payload = await queryBackend(activeUrl, question, apiKey, {
        signal: controller.signal,
        conversationId,
        history,
      });
      const assistantMessage = {
        id: uid(),
        role: "assistant",
        content: payload.answer,
        meta: payload,
        createdAt: new Date().toISOString(),
      };
      setMessages((current) => [...current, assistantMessage]);
      setSelectedMeta(null);
      setSourceRequest(null);
      setInspectorOpen(false);
      setInspectorTab("sources");
    } catch (error) {
      if (error.name !== "AbortError") {
        setMessages((current) => [...current, {
          id: uid(), role: "assistant", content: error.message || "Não foi possível concluir a consulta.",
          error: true, question, createdAt: new Date().toISOString(),
        }]);
      }
    } finally {
      activeController.current = null;
      setLoading(false);
    }
  }

  function retryQuestion(errorMessage) {
    const index = messages.indexOf(errorMessage);
    // Remove a pergunta original e o erro; submitQuestion recoloca a pergunta.
    submitQuestion(errorMessage.question, messages.slice(0, Math.max(index - 1, 0)));
  }

  const inspect = useCallback((meta, sourceIndex = null) => {
    setSelectedMeta(meta);
    setSourceRequest({ id: uid(), meta, index: sourceIndex });
    setInspectorOpen(true);
    setInspectorTab("sources");
  }, []);

  return (
    <div className={`app-shell ${inspectorOpen ? "has-inspector" : ""} ${sidebarCollapsed ? "sidebar-collapsed" : ""}`} id="top">
      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        health={health}
        healthState={healthState}
        healthError={healthError}
        onRefresh={refreshHealth}
        conversations={conversations}
        activeConversationId={conversationId}
        onNewChat={newConversation}
        onConversationPick={openConversation}
        onOpenSettings={() => setSettingsOpen(true)}
        onClearHistory={clearHistory}
        onToggleCollapse={toggleSidebarCollapse}
      />

      <main className="main-area">
        <div className="government-bar">
          <div className="government-brand">
            <span>SP</span>
            <strong>Governo do Estado de São Paulo</strong>
          </div>
          <small>Fundação Sistema Estadual de Análise de Dados</small>
        </div>
        <header className="topbar">
          <div className="topbar-left">
            <button
              type="button"
              className={`icon-button menu-button ${sidebarCollapsed ? "is-collapsed-trigger" : ""}`}
              onClick={() => {
                if (window.innerWidth <= 980) {
                  setSidebarOpen(true);
                } else {
                  toggleSidebarCollapse();
                }
              }}
              aria-label={sidebarCollapsed ? "Expandir histórico" : "Abrir menu"}
              title={sidebarCollapsed ? "Expandir histórico (Ctrl+B)" : "Abrir menu"}
            >
              {sidebarCollapsed ? <PanelLeftOpen size={20} /> : <Menu size={20} />}
            </button>
            <div><small>Nadia · Assistente Seade</small><strong>{ASSISTANT_MODE_LABEL}</strong></div>
            <span
              className={`top-status top-status--${healthState}`}
              title={healthError || `${activeUrl}/health`}
            >
              {healthState === "ready" ? "Online" : healthState === "initializing" ? "Inicializando" : healthState === "checking" ? "Verificando" : "Offline"}
            </span>
          </div>
          <div className="topbar-actions">
            <button className="icon-button" onClick={toggleTheme} aria-label="Alternar tema">
              {theme === "light" ? <Moon size={18} /> : <Sun size={18} />}
            </button>
          </div>
        </header>

        <div className="conversation">
          <div
            className="message-scroll"
            ref={messageScroll}
            onScroll={(event) => {
              const element = event.currentTarget;
              stickToBottom.current = element.scrollHeight - element.scrollTop - element.clientHeight < 120;
            }}
          >
            <div className={`message-column ${messages.length === 0 ? "is-empty" : ""}`}>
              {messages.length === 0 ? (
                <EmptyState
                  input={input}
                  setInput={setInput}
                  onSubmit={submitQuestion}
                />
              ) : (
                messages.map((message, index) => (
                  <Message
                    key={message.id}
                    message={message}
                    onInspect={inspect}
                    onExport={handleExport}
                    onOpenDrawer={handleOpenDrawer}
                    onRetry={
                      message.error && message.question && !loading && index === messages.length - 1
                        ? retryQuestion
                        : undefined
                    }
                  />
                ))
              )}
              {loading && <LoadingMessage developerMode={developerMode} />}
            </div>
          </div>
          {(messages.length > 0 || loading) && (
            <Composer value={input} setValue={setInput} onSubmit={submitQuestion} loading={loading} onCancel={() => activeController.current?.abort()} />
          )}
        </div>
      </main>

      <Inspector open={inspectorOpen} onClose={() => setInspectorOpen(false)} meta={selectedMeta} tab={inspectorTab} setTab={setInspectorTab} developerMode={developerMode} sourceRequest={sourceRequest} />

      <SourcesDrawer
        isOpen={Boolean(sourcesDrawerState?.isOpen)}
        onClose={() => setSourcesDrawerState(null)}
        sources={sourcesDrawerState?.sources || []}
        numericCitations={sourcesDrawerState?.numericCitations || []}
      />

      <ExportModal
        isOpen={Boolean(exportModalState?.isOpen)}
        onClose={() => setExportModalState(null)}
        question={exportModalState?.question || ""}
        answer={exportModalState?.answer || ""}
        sources={exportModalState?.sources || []}
        numericCitations={exportModalState?.numericCitations || []}
      />

      <SettingsDialog
        key={`${ACTIVE_RAG_TYPE}-${endpoints[ACTIVE_RAG_TYPE]}-${settingsOpen}`}
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        endpoints={endpoints}
        setEndpoints={setEndpointOverrides}
        apiKey={apiKey}
        setApiKey={setApiKey}
        developerMode={developerMode}
        setDeveloperMode={(value) => { setDeveloperMode(value); if (!value && inspectorTab === "pipeline") setInspectorTab("sources"); }}
      />
    </div>
  );
}
