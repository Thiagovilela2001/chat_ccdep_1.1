import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Activity,
  AlertTriangle,
  ArrowUp,
  Building2,
  Check,
  CheckCircle2,
  CircleStop,
  ChevronDown,
  ChevronRight,
  Clock3,
  Code2,
  Copy,
  Database,
  Download,
  FileText,
  Gauge,
  History,
  Layers3,
  MapPin,
  Menu,
  MessageSquarePlus,
  Moon,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  RefreshCw,
  Route,
  Search,
  Settings,
  ShieldCheck,
  Sparkles,
  Sun,
  Users,
  X,
} from "lucide-react";

import {
  apiUrls,
  checkHealth,
  isBackendReady,
  queryBackend,
} from "./lib/api";
import {
  isCalendarYearCitation,
} from "./lib/numericCitations";
import { upsertConversation, validStoredConversations } from "./lib/conversations";
import { readStorage, readStoredJson, writeStorage } from "./lib/storage";
import { rotateFeaturedQuestions } from "./lib/suggestions";

import { ProgressStepper } from "./components/ProgressStepper";
import { SourcesDrawer } from "./components/SourcesDrawer";
import { ExportModal } from "./components/ExportModal";

const STORAGE = {
  messages: "nadia.messages.v1",
  conversations: "nadia.conversations.v1",
  theme: "nadia.theme.v1",
  endpoints: "nadia.endpoints.v1",
  apiKey: "nadia.apiKey.v1",
  conversationId: "nadia.conversationId.v1",
};

const ACTIVE_RAG_TYPE = "principal";
const ASSISTANT_MODE_LABEL = "Análise documental";

const CLIENT_DIAGNOSTIC_URL = "/client-error";

function escapeRegExp(value) {
  return String(value || "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

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

function highlightVerifiedValues(text, values = []) {
  const content = String(text || "");
  const uniqueValues = [...new Set(values.map((value) => String(value || "").trim()).filter(Boolean))]
    .sort((left, right) => right.length - left.length);
  if (!content || uniqueValues.length === 0) return content;

  const pattern = new RegExp(`(${uniqueValues.map(escapeRegExp).join("|")})`, "g");
  const parts = content.split(pattern);
  return parts.map((part, index) => (
    uniqueValues.includes(part)
      ? <mark className="verified-source-value" key={`${part}-${index}`}>{part}</mark>
      : part
  ));
}

const SUGGESTION_ICONS = {
  labor: Activity,
  activity: Gauge,
  comparison: Layers3,
  demography: Users,
  investment: Building2,
  regional: MapPin,
};

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

function AnswerMetrics({ meta }) {
  const elapsed = meta?.timings?.total_ms ?? meta?._client_roundtrip_ms;
  if (elapsed == null) return null;
  return (
    <div className="answer-metrics">
      <MetricChip icon={Clock3}>{formatMs(elapsed)}</MetricChip>
    </div>
  );
}

function BrandMark() {
  return (
    <div className="brand-mark" aria-hidden="true">
      <span>n</span>
    </div>
  );
}

function ClearHistoryButton({ onClearHistory }) {
  const [confirming, setConfirming] = useState(false);
  const timerRef = useRef(null);

  function handleClick() {
    if (!confirming) {
      setConfirming(true);
      timerRef.current = setTimeout(() => setConfirming(false), 3000);
    } else {
      clearTimeout(timerRef.current);
      setConfirming(false);
      onClearHistory();
    }
  }

  useEffect(() => () => clearTimeout(timerRef.current), []);

  return (
    <button
      className={`clear-history-btn ${confirming ? "is-confirming" : ""}`}
      onClick={handleClick}
      aria-label={confirming ? "Confirmar limpeza do histórico" : "Limpar histórico de conversas"}
    >
      <X size={13} />
      {confirming ? "Confirmar limpeza" : "Limpar histórico"}
    </button>
  );
}

function Sidebar({
  open,
  onClose,
  health,
  healthState,
  healthError,
  onRefresh,
  conversations,
  activeConversationId,
  onNewChat,
  onConversationPick,
  onOpenSettings,
  onClearHistory,
  onToggleCollapse,
}) {
  const [historyCollapsed, setHistoryCollapsed] = useState(false);
  const history = [...conversations].sort(
    (left, right) => new Date(right.updatedAt || 0) - new Date(left.updatedAt || 0),
  );
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
          <div className="sidebar-head-actions">
            <button
              type="button"
              className="icon-button sidebar-collapse-btn"
              onClick={onToggleCollapse}
              aria-label="Recolher histórico"
              title="Recolher barra lateral (Ctrl+B)"
            >
              <PanelLeftClose size={18} />
            </button>
            <button className="icon-button sidebar-close" onClick={onClose} aria-label="Fechar menu">
              <X size={18} />
            </button>
          </div>
        </div>

        <button className="new-chat" onClick={onNewChat}>
          <MessageSquarePlus size={18} />
          Nova análise
        </button>

        <section className="sidebar-section history-section">
          <button
            type="button"
            className="sidebar-label sidebar-label-toggle"
            onClick={() => setHistoryCollapsed((prev) => !prev)}
            aria-expanded={!historyCollapsed}
            aria-label={historyCollapsed ? "Expandir histórico de conversas" : "Recolher histórico de conversas"}
            title={historyCollapsed ? "Expandir histórico" : "Recolher histórico"}
          >
            <span className="sidebar-label-main">
              <History size={14} />
              <span>Histórico</span>
              {history.length > 0 && (
                <span className="history-count-badge">{history.length}</span>
              )}
            </span>
            <span className="sidebar-label-chevron">
              {historyCollapsed ? <ChevronRight size={14} /> : <ChevronDown size={14} />}
            </span>
          </button>
          <div className={`history-list ${historyCollapsed ? "is-collapsed" : ""}`}>
            {history.length === 0 ? (
              <p className="history-empty">Suas análises aparecerão aqui.</p>
            ) : (
              history.map((conversation) => (
                <button
                  className={conversation.id === activeConversationId ? "is-active" : ""}
                  key={conversation.id}
                  onClick={() => onConversationPick(conversation)}
                  aria-current={conversation.id === activeConversationId ? "page" : undefined}
                >
                  <span>{conversation.title}</span>
                  <small>{relativeTime(conversation.createdAt)}</small>
                </button>
              ))
            )}
          </div>
          {!historyCollapsed && history.length > 0 && (
            <ClearHistoryButton onClearHistory={onClearHistory} />
          )}
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

function EmptyState({ onPick, input, setInput, onSubmit, suggestions }) {
  const visibleSuggestions = suggestions.slice(0, 6);

  return (
    <section className="empty-state">
      <div className="hero-intro">
        <div className="welcome-mark">
          <BrandMark />
          <span>
            <small>Inteligência pública aplicada</small>
            Nadia · Fundação Seade
          </span>
        </div>
        <h1>Transforme dados de São Paulo em decisões mais claras.</h1>
        <p>
          Investigue trabalho, economia e sociedade em linguagem natural. A Nadia cruza
          documentos oficiais e apresenta cada resposta com fontes rastreáveis.
        </p>
      </div>
      <Composer
        value={input}
        setValue={setInput}
        onSubmit={onSubmit}
        loading={false}
        onCancel={() => {}}
      />
      <div className="trust-row" aria-label="Compromissos da resposta">
        <span><ShieldCheck size={15} /> Fontes oficiais</span>
        <span><CheckCircle2 size={15} /> Números conferidos</span>
        <span><Sparkles size={15} /> Síntese objetiva</span>
      </div>

      <section className="suggestion-section" aria-labelledby="suggestion-title">
        <div className="suggestion-label">
          <span>
            <small>Comece por uma pergunta</small>
            <strong id="suggestion-title">Explore os dados do Estado</strong>
          </span>
          <div className="theme-ribbon" aria-label="Áreas de conhecimento">
            <span><Activity size={14} /> Trabalho</span>
            <span><Gauge size={14} /> Economia</span>
            <span><Layers3 size={14} /> Sociedade</span>
          </div>
        </div>
        <div className="suggestion-carousel" aria-label="Perguntas sugeridas">
          {visibleSuggestions.map(({ id, eyebrow, title }) => {
            const isDemography = id.startsWith("demography");
            const Icon = SUGGESTION_ICONS[id] || (isDemography ? Users : Sparkles);
            return (
              <button
                type="button"
                className={`suggestion-card ${isDemography ? "is-demography" : ""}`}
                key={id}
                onClick={() => onPick(title)}
              >
                <span className="suggestion-icon"><Icon size={18} /></span>
                <span className="suggestion-copy">
                  <small>{eyebrow}</small>
                  <strong>{title}</strong>
                </span>
                <ArrowUp size={16} />
              </button>
            );
          })}
        </div>
      </section>
    </section>
  );
}

function CopyButton({ text }) {
  const [status, setStatus] = useState("idle");
  const timerRef = useRef(null);

  useEffect(() => () => clearTimeout(timerRef.current), []);

  async function handleClick() {
    try {
      await navigator.clipboard.writeText(text);
      setStatus("copied");
    } catch {
      setStatus("failed");
    }
    clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => setStatus("idle"), 2000);
  }

  return (
    <button type="button" onClick={handleClick} aria-live="polite">
      {status === "copied" ? <Check size={14} /> : <Copy size={14} />}
      {status === "copied" ? "Copiado" : status === "failed" ? "Falha ao copiar" : "Copiar"}
    </button>
  );
}

function Message({ message, onInspect, onExport, onOpenDrawer, onRetry }) {
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
          <>
            <div className="error-card"><AlertTriangle size={18} /> {message.content}</div>
            {onRetry && (
              <div className="answer-actions">
                <button className="is-primary" type="button" onClick={() => onRetry(message)}>
                  <RefreshCw size={14} /> Tentar novamente
                </button>
              </div>
            )}
          </>
        ) : isUser ? (
          <p className="user-copy">{message.content}</p>
        ) : (
          <div className="markdown-body">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </ReactMarkdown>
          </div>
        )}
        {!isUser && !message.error && message.meta && (
          <>
            <AnswerMetrics meta={message.meta} />
            <div className="answer-actions">
              <CopyButton text={message.content} />
              {message.meta.sources && message.meta.sources.length > 0 && (
                <button className="is-primary" type="button" onClick={() => onOpenDrawer?.(message.meta)}>
                  <Database size={14} /> Fontes ({message.meta.sources.length})
                </button>
              )}
              <button type="button" onClick={() => onInspect?.(message.meta)}>
                <PanelRightOpen size={14} /> Detalhes
              </button>
              <button type="button" onClick={() => onExport?.(message)}>
                <Download size={14} /> Exportar
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
        <ProgressStepper active={true} />
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
          placeholder="Pergunte alguma coisa sobre São Paulo"
          rows="1"
          maxLength="4000"
          aria-label="Digite sua pergunta"
        />
        <div className="composer-bottom">
          <span><Sparkles size={14} /> Resposta com fontes do Seade</span>
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
      <small className="composer-note">A Nadia pode cometer erros. Confirme informações importantes nas fontes.</small>
    </div>
  );
}

function Inspector({ open, onClose, meta, tab, setTab, developerMode, sourceRequest }) {
  const requestedSourceIndex = sourceRequest?.meta === meta ? sourceRequest.index : null;
  const allSources = meta?.sources || [];
  const sources = requestedSourceIndex == null
    ? allSources
    : allSources
      .map((source, index) => ({ source, originalIndex: index }))
      .filter(({ originalIndex }) => originalIndex === Number(requestedSourceIndex));
  const numericCitations = (meta?.numeric_citations || [])
    .filter((citation) => !isCalendarYearCitation(citation?.value));
  const validation = meta?.validation || {};
  const route = meta?.route || {};
  const timings = meta?.timings || {};
  const [expandedSource, setExpandedSource] = useState(null);
  const requestedIndex = requestedSourceIndex == null ? null : 0;
  const expandedIndex = (
    expandedSource?.meta === meta
    && expandedSource?.requestId === sourceRequest?.id
  ) ? expandedSource.index : requestedIndex;

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
                  {sources.length === 0 ? <p className="muted">Nenhuma fonte detalhada retornada.</p> : sources.map((sourceEntry, index) => {
                    const source = sourceEntry?.source || sourceEntry;
                    const sourceIndex = sourceEntry?.originalIndex ?? index;
                    const verifiedValues = numericCitations
                      .filter((citation) => Number(citation.source_index) === sourceIndex)
                      .map((citation) => citation.value);
                    return (
                    <article className={expandedIndex === index ? "is-expanded" : ""} key={`${source.file || "source"}-${index}`}>
                      <button
                        type="button"
                        className="source-card-trigger"
                        aria-expanded={expandedIndex === index}
                        aria-controls={`source-excerpt-${index}`}
                        onClick={() => setExpandedSource({
                          meta,
                          requestId: sourceRequest?.id,
                          index: expandedIndex === index ? null : index,
                        })}
                      >
                        <span className="file-icon"><FileText size={17} /></span>
                        <span className="source-card-summary">
                          <strong>{source.file || `Fonte ${index + 1}`}</strong>
                          <small>{source.page != null ? `Página/aba ${source.page}` : "Localização não informada"}</small>
                          <span className="score-line"><i style={{ width: `${Math.round((source.score || 0) * 100)}%` }} /><span>{Math.round((source.score || 0) * 100)}%</span></span>
                        </span>
                        <ChevronDown className="source-card-chevron" size={15} aria-hidden="true" />
                      </button>
                      {expandedIndex === index && (
                        <div className={`source-excerpt ${verifiedValues.length > 0 ? "has-verified-values" : ""}`} id={`source-excerpt-${index}`}>
                          <small>Trecho utilizado</small>
                          {verifiedValues.length > 0 && (
                            <div className="verified-source-values" aria-label="Dados verificados nesta fonte">
                              <span>Dados verificados</span>
                              {verifiedValues.map((value) => <b key={value}>{value}</b>)}
                            </div>
                          )}
                          {source.excerpt && verifiedValues.length > 0 ? (
                            <p className="source-excerpt-highlighted">
                              {highlightVerifiedValues(source.excerpt, verifiedValues)}
                            </p>
                          ) : (
                            <p>{source.excerpt || "O texto deste trecho não está disponível em respostas salvas anteriormente. Faça uma nova consulta para visualizá-lo."}</p>
                          )}
                        </div>
                      )}
                    </article>
                    );
                  })}
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
  const closeButtonRef = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    const previousFocus = document.activeElement;
    const frame = window.requestAnimationFrame(() => closeButtonRef.current?.focus());
    const handleKeyDown = (event) => {
      if (event.key === "Escape") onClose?.();
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      window.cancelAnimationFrame(frame);
      document.removeEventListener("keydown", handleKeyDown);
      previousFocus?.focus?.();
    };
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="settings-dialog" role="dialog" aria-modal="true" aria-labelledby="settings-title">
        <header>
          <div><small>Preferências</small><h2 id="settings-title">Configurações</h2></div>
          <button ref={closeButtonRef} className="icon-button" onClick={onClose} aria-label="Fechar configurações"><X size={18} /></button>
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
  const [conversations, setConversations] = useState(storedConversations);
  const [featuredQuestions] = useState(
    () => rotateFeaturedQuestions(globalThis.localStorage),
  );
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

  function handleExport(message) {
    const messageIndex = messages.indexOf(message);
    const userMsg = messages.slice(0, messageIndex).reverse().find((m) => m.role === "user");
    setExportModalState({
      isOpen: true,
      question: userMsg?.content || "Consulta Documental",
      answer: message.content,
      sources: message.meta?.sources || [],
      numericCitations: message.meta?.numeric_citations || [],
    });
  }

  function handleOpenDrawer(meta) {
    setSourcesDrawerState({
      isOpen: true,
      sources: meta?.sources || [],
      numericCitations: meta?.numeric_citations || [],
    });
  }

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

  function inspect(meta, sourceIndex = null) {
    setSelectedMeta(meta);
    setSourceRequest({ id: uid(), meta, index: sourceIndex });
    setInspectorOpen(true);
    setInspectorTab("sources");
  }

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
                  onPick={submitQuestion}
                  input={input}
                  setInput={setInput}
                  onSubmit={submitQuestion}
                  suggestions={featuredQuestions}
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
              {loading && <LoadingMessage />}
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
