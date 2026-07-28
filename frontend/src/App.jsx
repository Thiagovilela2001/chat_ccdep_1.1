import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Activity,
  AlertTriangle,
  ArrowUp,
  Check,
  CheckCircle2,
  CircleStop,
  Clock3,
  Code2,
  Copy,
  Database,
  FileText,
  Gauge,
  History,
  Layers3,
  Menu,
  MessageSquarePlus,
  Moon,
  PanelRightClose,
  PanelRightOpen,
  RefreshCw,
  Route,
  Search,
  Settings,
  ShieldCheck,
  Sparkles,
  Sun,
  X,
} from "lucide-react";

import {
  apiUrls,
  checkHealth,
  isBackendReady,
  queryBackend,
} from "./lib/api";
import { readStorage, readStoredJson, writeStorage } from "./lib/storage";

const STORAGE = {
  messages: "nadia.messages.v1",
  theme: "nadia.theme.v1",
  endpoints: "nadia.endpoints.v1",
  apiKey: "nadia.apiKey.v1",
};

const ACTIVE_RAG_TYPE = "principal";
const ASSISTANT_MODE_LABEL = "Análise documental";

const CLIENT_DIAGNOSTIC_URL = "http://127.0.0.1:8501/client-error";

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

const SUGGESTIONS = [
  {
    icon: Activity,
    eyebrow: "Mercado de trabalho",
    title: "Como evoluiu o emprego formal paulista nos boletins mais recentes?",
  },
  {
    icon: Gauge,
    eyebrow: "Atividade econômica",
    title: "Quais setores mais contribuíram para o crescimento de São Paulo?",
  },
  {
    icon: Layers3,
    eyebrow: "Análise comparada",
    title: "Compare indústria, serviços e comércio no período analisado.",
  },
];

const uid = () =>
  globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;

function storedMessages() {
  return readStoredJson(globalThis.localStorage, STORAGE.messages, [], Array.isArray);
}

function storedTheme() {
  const value = readStorage(globalThis.localStorage, STORAGE.theme, "light");
  return value === "dark" ? "dark" : "light";
}

function storedEndpoints() {
  return readStoredJson(
    globalThis.localStorage,
    STORAGE.endpoints,
    {},
    (value) => value !== null && typeof value === "object" && !Array.isArray(value),
  );
}

function formatMs(value) {
  if (!Number.isFinite(Number(value))) return "—";
  const ms = Number(value);
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}

function relativeTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "agora";
  return new Intl.DateTimeFormat("pt-BR", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function MetricChip({ icon: Icon, children, tone = "neutral" }) {
  return (
    <span className={`metric-chip metric-chip--${tone}`}>
      <Icon size={13} aria-hidden="true" />
      {children}
    </span>
  );
}

function ValidationBadge({ validation, label = "números" }) {
  const total = Number(validation?.total || 0);
  const verified = Number(validation?.verified || 0);
  if (!total) return null;
  const complete = total === verified;
  return (
    <MetricChip icon={complete ? CheckCircle2 : AlertTriangle} tone={complete ? "ok" : "warn"}>
      {verified}/{total} {label}
    </MetricChip>
  );
}

function AnswerMetrics({ meta }) {
  const elapsed = meta?.timings?.total_ms ?? meta?._client_roundtrip_ms;
  return (
    <div className="answer-metrics">
      <ValidationBadge validation={meta?.validation} />
      {elapsed != null && <MetricChip icon={Clock3}>{formatMs(elapsed)}</MetricChip>}
    </div>
  );
}

function BrandMark() {
  return (
    <div className="brand-mark" aria-hidden="true">
      <span>N</span>
    </div>
  );
}

function Sidebar({
  open,
  onClose,
  health,
  healthState,
  healthError,
  onRefresh,
  messages,
  onNewChat,
  onHistoryPick,
  onOpenSettings,
}) {
  const questions = messages.filter((item) => item.role === "user").slice(-8).reverse();
  const ready = isBackendReady(health);

  return (
    <>
      <button
        className={`mobile-scrim ${open ? "is-open" : ""}`}
        onClick={onClose}
        aria-label="Fechar menu"
      />
      <aside className={`sidebar ${open ? "is-open" : ""}`}>
        <div className="sidebar-government">
          <span>SP</span>
          <strong>Governo do Estado de São Paulo</strong>
        </div>
        <div className="sidebar-head">
          <a className="brand" href="#top" aria-label="Nadia — início">
            <span className="seade-wordmark">SEADE</span>
            <span>
              <strong>Nadia</strong>
              <small>Assistente de dados</small>
            </span>
          </a>
          <button className="icon-button sidebar-close" onClick={onClose} aria-label="Fechar menu">
            <X size={18} />
          </button>
        </div>

        <button className="new-chat" onClick={onNewChat}>
          <MessageSquarePlus size={18} />
          Nova análise
          <span>⌘ K</span>
        </button>

        <section className="sidebar-section history-section">
          <div className="sidebar-label">
            <span>Histórico</span>
            <History size={14} />
          </div>
          <div className="history-list">
            {questions.length === 0 ? (
              <p className="history-empty">Suas perguntas recentes aparecerão aqui.</p>
            ) : (
              questions.map((item) => (
                <button key={item.id} onClick={() => onHistoryPick(item.content)}>
                  <span>{item.content}</span>
                  <small>{relativeTime(item.createdAt)}</small>
                </button>
              ))
            )}
          </div>
        </section>

        <div className="sidebar-foot">
          <button className="connection-card" onClick={onRefresh} disabled={healthState === "checking"}>
            <span className={`status-orb status-orb--${ready ? "ready" : healthState}`} />
            <span>
              <strong>{ready ? "Sistema operacional" : healthState === "checking" ? "Verificando" : "Indisponível"}</strong>
              <small title={healthError || undefined}>
                {healthError || "Base documental"}
              </small>
            </span>
            <RefreshCw size={15} className={healthState === "checking" ? "is-spinning" : ""} />
          </button>
          <button className="settings-button" onClick={onOpenSettings}>
            <Settings size={17} />
            Configurações
          </button>
        </div>
      </aside>
    </>
  );
}

function EmptyState({ onPick, input, setInput, onSubmit }) {
  return (
    <section className="empty-state">
      <div className="theme-ribbon" aria-label="Áreas de conhecimento">
        <span><Activity size={17} /> Mercado de trabalho</span>
        <span><Gauge size={17} /> Economia</span>
        <span><Layers3 size={17} /> Indicadores sociais</span>
      </div>
      <div className="hero-kicker">Inteligência para dados públicos</div>
      <h1>Informação confiável<br />para entender São Paulo.</h1>
      <p>
        Consulte os Boletins de Conjuntura Paulista em linguagem natural. A Nadia
        encontra evidências, compara períodos e mostra a origem de cada informação.
      </p>
      <Composer
        value={input}
        setValue={setInput}
        onSubmit={onSubmit}
        loading={false}
        onCancel={() => {}}
      />
      <div className="suggestion-label">
        <strong>Perguntas em destaque</strong>
        <span>Escolha um tema para começar</span>
      </div>
      <div className="suggestion-grid">
        {SUGGESTIONS.map(({ icon: Icon, eyebrow, title }) => (
          <button key={title} onClick={() => onPick(title)}>
            <span className="suggestion-icon"><Icon size={18} /></span>
            <small>{eyebrow}</small>
            <strong>{title}</strong>
            <ArrowUp size={16} />
          </button>
        ))}
      </div>
      <div className="trust-row">
        <span><ShieldCheck size={15} /> Fontes rastreáveis</span>
        <span><CheckCircle2 size={15} /> Validação numérica</span>
        <span><Sparkles size={15} /> Síntese técnica</span>
      </div>
    </section>
  );
}

function Message({ message, onInspect }) {
  const isUser = message.role === "user";
  return (
    <article className={`message message--${message.role}`}>
      <div className="message-avatar">{isUser ? "Você" : <BrandMark />}</div>
      <div className="message-content">
        <div className="message-heading">
          <strong>{isUser ? "Você" : "Nadia"}</strong>
          <time>{relativeTime(message.createdAt)}</time>
        </div>
        {message.error ? (
          <div className="error-card"><AlertTriangle size={18} /> {message.content}</div>
        ) : isUser ? (
          <p className="user-copy">{message.content}</p>
        ) : (
          <div className="markdown-body">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                a: (props) => <a {...props} target="_blank" rel="noreferrer" />,
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        )}
        {!isUser && !message.error && message.meta && (
          <>
            <AnswerMetrics meta={message.meta} />
            <div className="answer-actions">
              <button onClick={() => navigator.clipboard.writeText(message.content)}>
                <Copy size={14} /> Copiar
              </button>
              <button onClick={() => onInspect(message.meta)}>
                <PanelRightOpen size={14} /> Ver evidências
              </button>
            </div>
          </>
        )}
      </div>
    </article>
  );
}

function LoadingMessage() {
  return (
    <article className="message message--assistant">
      <div className="message-avatar"><BrandMark /></div>
      <div className="message-content">
        <div className="message-heading"><strong>Nadia</strong><span>Analisando</span></div>
        <div className="thinking-card">
          <span className="thinking-mark"><Search size={17} /></span>
          <div>
            <strong>Consultando as melhores fontes</strong>
            <small>Recuperando documentos, comparando evidências e validando os números.</small>
          </div>
          <i /><i /><i />
        </div>
      </div>
    </article>
  );
}

function Composer({ value, setValue, onSubmit, loading, onCancel }) {
  const textarea = useRef(null);

  useEffect(() => {
    const element = textarea.current;
    if (!element) return;
    element.style.height = "0px";
    element.style.height = `${Math.min(element.scrollHeight, 160)}px`;
  }, [value]);

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (!loading && value.trim()) onSubmit(value);
    }
  }

  return (
    <div className="composer-wrap">
      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault();
          if (!loading && value.trim()) onSubmit(value);
        }}
      >
        <textarea
          ref={textarea}
          value={value}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Pergunte sobre emprego, PIB, indústria, serviços…"
          rows="1"
          maxLength="4000"
          aria-label="Digite sua pergunta"
        />
        <div className="composer-bottom">
          <span><Sparkles size={14} /> {ASSISTANT_MODE_LABEL}</span>
          {loading ? (
            <button className="send-button is-stop" type="button" onClick={onCancel} aria-label="Cancelar consulta">
              <CircleStop size={18} />
            </button>
          ) : (
            <button className="send-button" type="submit" disabled={!value.trim()} aria-label="Enviar pergunta">
              <ArrowUp size={19} />
            </button>
          )}
        </div>
      </form>
      <small className="composer-note">A Nadia pode cometer erros. Confira as fontes e a validação antes de usar os dados.</small>
    </div>
  );
}

function Inspector({ open, onClose, meta, tab, setTab, developerMode }) {
  const sources = meta?.sources || [];
  const validation = meta?.validation || {};
  const route = meta?.route || {};
  const timings = meta?.timings || {};

  return (
    <aside className={`inspector ${open ? "is-open" : ""}`}>
      <header>
        <div>
          <small>Evidências da resposta</small>
          <strong>Transparência</strong>
        </div>
        <button className="icon-button" onClick={onClose} aria-label="Fechar painel">
          <PanelRightClose size={18} />
        </button>
      </header>
      {!meta ? (
        <div className="inspector-empty">
          <Database size={22} />
          <strong>Nenhuma resposta selecionada</strong>
          <p>As fontes, validações e detalhes do pipeline aparecem aqui após uma consulta.</p>
        </div>
      ) : (
        <>
          <nav className="inspector-tabs">
            <button className={tab === "sources" ? "is-active" : ""} onClick={() => setTab("sources")}>Fontes</button>
            <button className={tab === "validation" ? "is-active" : ""} onClick={() => setTab("validation")}>Validação</button>
            {developerMode && <button className={tab === "pipeline" ? "is-active" : ""} onClick={() => setTab("pipeline")}>Pipeline</button>}
          </nav>

          <div className="inspector-body">
            {tab === "sources" && (
              <section>
                {meta.rewritten_query && (
                  <div className="rewritten-query">
                    <Search size={15} />
                    <span><small>Consulta reescrita</small>{meta.rewritten_query}</span>
                  </div>
                )}
                <div className="section-title"><span>Trechos recuperados</span><b>{sources.length}</b></div>
                <div className="source-cards">
                  {sources.length === 0 ? <p className="muted">Nenhuma fonte detalhada retornada.</p> : sources.map((source, index) => (
                    <article key={`${source.file || "source"}-${index}`}>
                      <span className="file-icon"><FileText size={17} /></span>
                      <div>
                        <strong>{source.file || `Fonte ${index + 1}`}</strong>
                        <small>{source.page != null ? `Página/aba ${source.page}` : "Localização não informada"}</small>
                        <div className="score-line"><i style={{ width: `${Math.round((source.score || 0) * 100)}%` }} /><span>{Math.round((source.score || 0) * 100)}%</span></div>
                      </div>
                    </article>
                  ))}
                </div>
              </section>
            )}

            {tab === "validation" && (
              <section className="validation-panel">
                <ValidationCard title="Validação numérica" icon={CheckCircle2} data={validation} />
              </section>
            )}

            {tab === "pipeline" && developerMode && (
              <section className="pipeline-panel">
                <InfoBlock icon={Route} label="Engine escolhida" value={route.engine_label || route.engine || meta.rag_label || "—"} />
                <InfoBlock icon={Gauge} label="Confiança" value={Number.isFinite(route.confidence) ? `${Math.round(route.confidence * 100)}%` : "—"} />
                <InfoBlock icon={Clock3} label="Tempo total" value={formatMs(timings.total_ms ?? meta._client_roundtrip_ms)} />
                {route.reasoning && <div className="reasoning"><small>Motivo do roteamento</small><p>{route.reasoning}</p></div>}
                <details>
                  <summary><Code2 size={15} /> Resposta técnica completa</summary>
                  <pre>{JSON.stringify(meta, null, 2)}</pre>
                </details>
              </section>
            )}
          </div>
        </>
      )}
    </aside>
  );
}

function ValidationCard({ title, icon: Icon, data }) {
  const total = Number(data?.total || 0);
  const verified = Number(data?.verified || 0);
  const complete = total > 0 && verified === total;
  const percentage = total ? Math.round((verified / total) * 100) : 0;
  return (
    <article className={`validation-card ${complete ? "is-complete" : ""}`}>
      <div className="validation-top">
        <span><Icon size={18} /></span>
        <div><small>{title}</small><strong>{total ? `${verified} de ${total}` : "Sem ocorrências"}</strong></div>
        {total > 0 && <b>{percentage}%</b>}
      </div>
      {total > 0 && <div className="validation-bar"><i style={{ width: `${percentage}%` }} /></div>}
      {data?.unverified?.length > 0 ? (
        <div className="unverified-values">
          <small>Itens não localizados</small>
          {data.unverified.map((value, index) => <code key={`${value}-${index}`}>{String(value)}</code>)}
        </div>
      ) : total > 0 ? <p><Check size={14} /> Todos os itens conferem com as fontes.</p> : null}
    </article>
  );
}

function InfoBlock({ icon: Icon, label, value }) {
  return <div className="info-block"><span><Icon size={16} /></span><div><small>{label}</small><strong>{value}</strong></div></div>;
}

function SettingsDialog({ open, onClose, endpoints, setEndpoints, apiKey, setApiKey, developerMode, setDeveloperMode }) {
  const [draft, setDraft] = useState(endpoints[ACTIVE_RAG_TYPE] || "");
  if (!open) return null;
  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="settings-dialog" role="dialog" aria-modal="true" aria-labelledby="settings-title">
        <header>
          <div><small>Preferências</small><h2 id="settings-title">Configurações</h2></div>
          <button className="icon-button" onClick={onClose}><X size={18} /></button>
        </header>
        <div className="dialog-body">
          <label>
            <span>Endpoint do assistente</span>
            <input value={draft} onChange={(event) => setDraft(event.target.value)} spellCheck="false" />
            <small>O endereço é salvo somente neste navegador.</small>
          </label>
          <label>
            <span>Chave da API</span>
            <input type="password" value={apiKey} onChange={(event) => setApiKey(event.target.value)} placeholder="Opcional" autoComplete="off" />
            <small>A chave fica apenas na sessão atual e nunca é persistida no histórico.</small>
          </label>
          <label className="toggle-row">
            <span><strong>Modo desenvolvedor</strong><small>Exibe roteamento, tempos e o JSON técnico.</small></span>
            <input type="checkbox" checked={developerMode} onChange={(event) => setDeveloperMode(event.target.checked)} />
            <i />
          </label>
        </div>
        <footer>
          <button className="secondary-button" onClick={onClose}>Cancelar</button>
          <button className="primary-button" onClick={() => { setEndpoints({ ...endpoints, [ACTIVE_RAG_TYPE]: draft.trim().replace(/\/+$/, "") }); onClose(); }}>Salvar alterações</button>
        </footer>
      </section>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState(storedMessages);
  const [theme, setTheme] = useState(storedTheme);
  const [endpointOverrides, setEndpointOverrides] = useState(storedEndpoints);
  const [apiKey, setApiKey] = useState(() => readStorage(globalThis.sessionStorage, STORAGE.apiKey));
  const [developerMode, setDeveloperMode] = useState(false);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState(null);
  const [healthState, setHealthState] = useState("checking");
  const [healthError, setHealthError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [inspectorTab, setInspectorTab] = useState("sources");
  const [selectedMeta, setSelectedMeta] = useState(() => {
    const last = [...storedMessages()].reverse().find((item) => item?.meta);
    return last?.meta || null;
  });
  const activeController = useRef(null);
  const messageScroll = useRef(null);
  const endpoints = useMemo(() => apiUrls(endpointOverrides), [endpointOverrides]);
  const activeUrl = endpoints[ACTIVE_RAG_TYPE];

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    writeStorage(globalThis.localStorage, STORAGE.theme, theme);
  }, [theme]);

  useEffect(() => writeStorage(globalThis.localStorage, STORAGE.endpoints, JSON.stringify(endpointOverrides)), [endpointOverrides]);
  useEffect(() => writeStorage(globalThis.sessionStorage, STORAGE.apiKey, apiKey), [apiKey]);
  useEffect(() => writeStorage(globalThis.localStorage, STORAGE.messages, JSON.stringify(messages.slice(-40))), [messages]);
  useEffect(() => {
    const container = messageScroll.current;
    if (!container) return undefined;
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
      reportClientDiagnostic({
        kind: "health-check",
        activeUrl,
        result: nextState,
        payload,
      });
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
    setInput("");
    setLoading(false);
  }

  async function submitQuestion(value) {
    const question = value.trim();
    if (!question || loading) return;
    const userMessage = { id: uid(), role: "user", content: question, createdAt: new Date().toISOString() };
    setMessages((current) => [...current, userMessage]);
    setInput("");
    setLoading(true);
    setSidebarOpen(false);
    const controller = new AbortController();
    activeController.current = controller;
    try {
      const payload = await queryBackend(activeUrl, question, apiKey, { signal: controller.signal });
      const assistantMessage = {
        id: uid(),
        role: "assistant",
        content: payload.answer,
        meta: payload,
        createdAt: new Date().toISOString(),
      };
      setMessages((current) => [...current, assistantMessage]);
      setSelectedMeta(payload);
      setInspectorTab("sources");
    } catch (error) {
      if (error.name !== "AbortError") {
        setMessages((current) => [...current, {
          id: uid(), role: "assistant", content: error.message || "Não foi possível concluir a consulta.",
          error: true, createdAt: new Date().toISOString(),
        }]);
      }
    } finally {
      activeController.current = null;
      setLoading(false);
    }
  }

  function inspect(meta) {
    setSelectedMeta(meta);
    setInspectorOpen(true);
    setInspectorTab("sources");
  }

  return (
    <div className={`app-shell ${inspectorOpen ? "has-inspector" : ""}`} id="top">
      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        health={health}
        healthState={healthState}
        healthError={healthError}
        onRefresh={refreshHealth}
        messages={messages}
        onNewChat={newConversation}
        onHistoryPick={(value) => { setInput(value); setSidebarOpen(false); }}
        onOpenSettings={() => setSettingsOpen(true)}
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
            <button className="icon-button menu-button" onClick={() => setSidebarOpen(true)} aria-label="Abrir menu"><Menu size={20} /></button>
            <div><small>Nadia · Assistente Seade</small><strong>{ASSISTANT_MODE_LABEL}</strong></div>
            <span
              className={`top-status top-status--${healthState}`}
              title={healthError || `${activeUrl}/health`}
            >
              {healthState === "ready" ? "Online" : healthState === "initializing" ? "Inicializando" : healthState === "checking" ? "Verificando" : "Offline"}
            </span>
          </div>
          <div className="topbar-actions">
            <button className="icon-button" onClick={() => setTheme(theme === "light" ? "dark" : "light")} aria-label="Alternar tema">
              {theme === "light" ? <Moon size={18} /> : <Sun size={18} />}
            </button>
            <button className="icon-button inspector-toggle" onClick={() => setInspectorOpen(!inspectorOpen)} aria-label="Alternar painel de evidências">
              {inspectorOpen ? <PanelRightClose size={18} /> : <PanelRightOpen size={18} />}
            </button>
          </div>
        </header>

        <div className="conversation">
          <div className="message-scroll" ref={messageScroll}>
            <div className="message-column">
              {messages.length === 0 ? (
                <EmptyState
                  onPick={submitQuestion}
                  input={input}
                  setInput={setInput}
                  onSubmit={submitQuestion}
                />
              ) : (
                messages.map((message) => <Message key={message.id} message={message} onInspect={inspect} />)
              )}
              {loading && <LoadingMessage />}
            </div>
          </div>
          {(messages.length > 0 || loading) && (
            <Composer value={input} setValue={setInput} onSubmit={submitQuestion} loading={loading} onCancel={() => activeController.current?.abort()} />
          )}
        </div>
      </main>

      <Inspector open={inspectorOpen} onClose={() => setInspectorOpen(false)} meta={selectedMeta} tab={inspectorTab} setTab={setInspectorTab} developerMode={developerMode} />

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
