import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
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
  ChevronLeft,
  ChevronRight,
  Clock3,
  Code2,
  Copy,
  Database,
  FileText,
  Gauge,
  History,
  Layers3,
  MapPin,
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
  annotateNumericCitations,
  conceptualizeTabularSnippet,
  isCalendarYearCitation,
  parseTabularSnippet,
} from "./lib/numericCitations";
import { upsertConversation, validStoredConversations } from "./lib/conversations";
import { readStorage, readStoredJson, writeStorage } from "./lib/storage";
import { rotateFeaturedQuestions } from "./lib/suggestions";

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

const CLIENT_DIAGNOSTIC_URL = "http://127.0.0.1:8501/client-error";

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
      <span>n</span>
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
  conversations,
  activeConversationId,
  onNewChat,
  onConversationPick,
  onOpenSettings,
}) {
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
          <button className="icon-button sidebar-close" onClick={onClose} aria-label="Fechar menu">
            <X size={18} />
          </button>
        </div>

        <button className="new-chat" onClick={onNewChat}>
          <MessageSquarePlus size={18} />
          Nova análise
        </button>

        <section className="sidebar-section history-section">
          <div className="sidebar-label">
            <span>Histórico</span>
            <History size={14} />
          </div>
          <div className="history-list">
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
  const suggestionCarousel = useRef(null);

  const scrollSuggestions = (direction) => {
    const carousel = suggestionCarousel.current;
    if (!carousel) return;
    const card = carousel.querySelector(".suggestion-card");
    const cardWidth = card?.getBoundingClientRect().width || carousel.clientWidth * 0.8;
    carousel.scrollBy({ left: direction * cardWidth * 2, behavior: "smooth" });
  };

  return (
    <section className="empty-state">
      <div className="welcome-mark">
        <BrandMark />
        <span>Nadia · inteligência Seade</span>
      </div>
      <h1>O que você quer saber<br />sobre São Paulo?</h1>
      <p>
        Converse com os dados da Fundação Seade. Pergunte em linguagem natural e
        receba respostas objetivas, com fontes e números verificáveis.
      </p>
      <Composer
        value={input}
        setValue={setInput}
        onSubmit={onSubmit}
        loading={false}
        onCancel={() => {}}
      />
      <div className="suggestion-label">
        <strong>Experimente perguntar</strong>
        <div className="suggestion-actions">
          <span>deslize para explorar</span>
          <div className="suggestion-controls">
            <button type="button" onClick={() => scrollSuggestions(-1)} aria-label="Ver perguntas anteriores">
              <ChevronLeft size={16} />
            </button>
            <button type="button" onClick={() => scrollSuggestions(1)} aria-label="Ver próximas perguntas">
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      </div>
      <div className="suggestion-carousel" ref={suggestionCarousel} aria-label="Perguntas sugeridas">
        {suggestions.map(({ id, eyebrow, title }) => {
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
              <small>{eyebrow}</small>
              <strong>{title}</strong>
              <ArrowUp size={16} />
            </button>
          );
        })}
      </div>
      <div className="trust-row">
        <span><ShieldCheck size={15} /> Fontes rastreáveis</span>
        <span><CheckCircle2 size={15} /> Validação numérica</span>
        <span><Sparkles size={15} /> Síntese técnica</span>
      </div>
      <div className="theme-ribbon" aria-label="Áreas de conhecimento">
        <span><Activity size={15} /> Trabalho</span>
        <span><Gauge size={15} /> Economia</span>
        <span><Layers3 size={15} /> Sociedade</span>
      </div>
    </section>
  );
}

function NumericCitation({ citation, children, onOpenSource }) {
  const tooltipId = `numeric-source-${useId().replaceAll(":", "")}`;
  const buttonRef = useRef(null);
  const popoverRef = useRef(null);
  const hideTimer = useRef(null);
  const copyTimer = useRef(null);
  const [hovered, setHovered] = useState(false);
  const [pinned, setPinned] = useState(false);
  const [copied, setCopied] = useState(false);
  const [position, setPosition] = useState(null);
  const visible = hovered || pinned;
  const sourceName = String(citation.file || "Documento")
    .split(/[\\/]/)
    .filter(Boolean)
    .at(-1);
  const tableEvidence = citation.content_type === "table"
    ? parseTabularSnippet(citation.snippet)
    : null;
  const tableConcept = tableEvidence
    ? conceptualizeTabularSnippet(tableEvidence, citation.value)
    : null;
  const cleanClaim = String(citation.claim || "")
    .replace(/^…|…$/g, "")
    .replace(/[*_`#]/g, "")
    .trim();
  const generatedExplanation = String(citation.explanation || "").trim();
  const fallbackConceptText = cleanClaim && !cleanClaim.includes("|")
    ? cleanClaim
    : tableConcept
      ? `${citation.value} é o valor registrado para ${tableConcept.subject}.`
      : `O valor confirmado nesta fonte é ${citation.value}.`;
  const conceptText = generatedExplanation || fallbackConceptText;
  const conceptQualifiers = (tableConcept?.qualifiers || [])
    .filter(({ label, value }) => label && value && !/^coluna \d+$/i.test(label))
    .slice(0, 4);

  function highlightValue(text) {
    const value = String(text || "");
    const index = value.indexOf(citation.value);
    if (index < 0) return value;
    return (
      <>
        {value.slice(0, index)}
        <mark>{citation.value}</mark>
        {value.slice(index + citation.value.length)}
      </>
    );
  }

  function highlightTableNumbers(text) {
    const value = String(text || "");
    const pattern = /[-−]?(?:\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?)(?:\s*%|\s*p\.p\.)?/gi;
    const parts = [];
    let cursor = 0;
    let match;
    while ((match = pattern.exec(value)) !== null) {
      if (match.index > cursor) parts.push(value.slice(cursor, match.index));
      const number = match[0];
      parts.push(
        number.trim() === citation.value.trim()
          ? <mark key={`${match.index}-${number}`}>{number}</mark>
          : <span className="numeric-table-value" key={`${match.index}-${number}`}>{number}</span>,
      );
      cursor = match.index + number.length;
    }
    if (cursor === 0) return value;
    if (cursor < value.length) parts.push(value.slice(cursor));
    return parts;
  }

  const updatePosition = useCallback(() => {
    const button = buttonRef.current;
    if (!button) return;
    const rect = button.getBoundingClientRect();
    const mobile = window.innerWidth <= 680;
    if (mobile) {
      setPosition({
        mobile: true,
        left: 8,
        bottom: Math.max(8, Number.parseInt(getComputedStyle(document.documentElement).getPropertyValue("--safe-bottom"), 10) || 8),
        width: window.innerWidth - 16,
      });
      return;
    }
    const width = Math.min(410, Math.max(300, window.innerWidth - 32));
    const left = Math.min(
      window.innerWidth - width - 16,
      Math.max(16, rect.left + rect.width / 2 - width / 2),
    );
    const below = rect.top < 250;
    setPosition({
      mobile: false,
      below,
      left,
      top: below ? rect.bottom + 12 : rect.top - 12,
      width,
    });
  }, []);

  useEffect(() => {
    if (!visible) return undefined;
    const reposition = () => updatePosition();
    window.addEventListener("resize", reposition);
    document.addEventListener("scroll", reposition, true);
    return () => {
      window.removeEventListener("resize", reposition);
      document.removeEventListener("scroll", reposition, true);
    };
  }, [updatePosition, visible]);

  useEffect(() => {
    if (!pinned) return undefined;
    const close = (event) => {
      if (
        !buttonRef.current?.contains(event.target)
        && !popoverRef.current?.contains(event.target)
      ) {
        setPinned(false);
      }
    };
    document.addEventListener("pointerdown", close);
    return () => document.removeEventListener("pointerdown", close);
  }, [pinned]);

  useEffect(() => () => {
    window.clearTimeout(hideTimer.current);
    window.clearTimeout(copyTimer.current);
  }, []);

  const cancelHide = () => {
    window.clearTimeout(hideTimer.current);
    hideTimer.current = null;
  };

  const scheduleHide = () => {
    cancelHide();
    hideTimer.current = window.setTimeout(() => setHovered(false), 180);
  };

  const show = () => {
    cancelHide();
    updatePosition();
    setHovered(true);
  };

  async function copyEvidence() {
    const evidence = String(citation.snippet || conceptText || "").trim();
    if (!evidence || !navigator.clipboard?.writeText) return;
    try {
      await navigator.clipboard.writeText(evidence);
      setCopied(true);
      window.clearTimeout(copyTimer.current);
      copyTimer.current = window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  }

  return (
    <span className="numeric-citation-wrap">
      <button
        ref={buttonRef}
        type="button"
        className={`numeric-citation ${pinned ? "is-pinned" : ""}`}
        aria-describedby={visible ? tooltipId : undefined}
        aria-expanded={pinned}
        aria-label={`Ver fonte do valor ${citation.value}`}
        onMouseEnter={show}
        onMouseLeave={scheduleHide}
        onFocus={show}
        onBlur={scheduleHide}
        onClick={() => {
          updatePosition();
          setPinned((current) => !current);
        }}
      >
        <span>{children}</span>
        <sup>{Number(citation.source_index) + 1}</sup>
      </button>
      {visible && position && createPortal(
        <div
          ref={popoverRef}
          id={tooltipId}
          role={pinned ? "dialog" : "tooltip"}
          aria-label={`Fonte do valor ${citation.value}`}
          className={`numeric-source-tooltip ${position.below ? "is-below" : ""} ${position.mobile ? "is-mobile" : ""}`}
          style={{
            left: position.left,
            top: position.top ?? "auto",
            bottom: position.bottom ?? "auto",
            width: position.width,
          }}
          onMouseEnter={show}
          onMouseLeave={scheduleHide}
          onFocus={show}
          onBlur={(event) => {
            if (!event.currentTarget.contains(event.relatedTarget)) scheduleHide();
          }}
        >
          <header>
            <span className="numeric-source-heading">
              <span><CheckCircle2 size={17} /></span>
              <span><small>Dado verificado</small><strong>{citation.value}</strong></span>
            </span>
            <button
              type="button"
              className="numeric-source-close"
              aria-label="Fechar fonte"
              onClick={() => {
                setPinned(false);
                setHovered(false);
              }}
            >
              <X size={15} />
            </button>
          </header>
          <div className={`numeric-source-excerpt ${tableEvidence ? "is-table" : ""}`}>
            <div className="numeric-source-concept">
              <small>O que esse dado mostra</small>
              <p>{highlightValue(conceptText)}</p>
              {conceptQualifiers.length > 0 && (
                <div className="numeric-source-context">
                  {conceptQualifiers.map(({ label, value }, index) => (
                    <span key={`${label}-${index}`}>
                      <small>{label}</small>
                      <strong>{highlightTableNumbers(value)}</strong>
                    </span>
                  ))}
                </div>
              )}
            </div>

            <div className="numeric-source-origin">
              <span className="numeric-source-file-icon"><FileText size={16} /></span>
              <span className="numeric-source-file">
                <small>Fonte</small>
                <strong title={citation.file}>{sourceName}</strong>
              </span>
              <div className="numeric-source-meta">
                <span>{citation.page != null ? `p. ${citation.page}` : "Sem página"}</span>
                {Number.isFinite(Number(citation.score)) && <span>{Math.round(Number(citation.score) * 100)}% relevante</span>}
              </div>
            </div>

            {citation.snippet && (
              <details className="numeric-source-raw">
                <summary>
                  <span>{tableEvidence ? "Ver dados de origem" : "Ver trecho usado"}</span>
                  <ChevronDown size={15} />
                </summary>
                {tableEvidence ? (
                  <div className={`numeric-source-table ${tableEvidence.kind === "key-value" ? "is-key-value" : ""}`}>
                    <table>
                      <thead>
                        <tr>{tableEvidence.headers.map((header, index) => <th key={`${header}-${index}`}>{header}</th>)}</tr>
                      </thead>
                      <tbody>
                        {tableEvidence.rows.map((row, rowIndex) => (
                          <tr key={`row-${rowIndex}`}>
                            {row.map((cell, cellIndex) => (
                              <td
                                className={/\d/.test(cell) ? "has-number" : ""}
                                key={`cell-${rowIndex}-${cellIndex}`}
                              >
                                {highlightTableNumbers(cell)}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p>{highlightValue(citation.snippet)}</p>
                )}
              </details>
            )}
          </div>
          <footer>
            <button
              type="button"
              className="numeric-source-copy"
              onClick={copyEvidence}
              disabled={!citation.snippet && !conceptText}
            >
              {copied ? <Check size={15} /> : <Copy size={15} />}
              {copied ? "Copiado" : "Copiar trecho"}
            </button>
            <button
              type="button"
              className="numeric-source-open"
              onClick={() => {
                setPinned(false);
                setHovered(false);
                onOpenSource?.();
              }}
            >
              <PanelRightOpen size={15} /> Ver fonte completa
            </button>
          </footer>
        </div>,
        document.body,
      )}
    </span>
  );
}

function Message({ message, onInspect }) {
  const isUser = message.role === "user";
  const numericCitations = (message.meta?.numeric_citations || [])
    .filter((citation) => !isCalendarYearCitation(citation?.value));
  const annotatedContent = annotateNumericCitations(message.content, numericCitations);
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
                a: ({ href, children, ...props }) => {
                  const match = href?.match(/^#numeric-citation-(\d+)$/);
                  const citation = match ? numericCitations[Number(match[1])] : null;
                  if (citation) {
                    return (
                      <NumericCitation
                        citation={citation}
                        onOpenSource={() => onInspect(message.meta, citation.source_index)}
                      >
                        {children}
                      </NumericCitation>
                    );
                  }
                  return <a href={href} {...props} target="_blank" rel="noreferrer">{children}</a>;
                },
              }}
            >
              {annotatedContent}
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
                          <p>{source.excerpt || "O texto deste trecho não está disponível em respostas salvas anteriormente. Faça uma nova consulta para visualizá-lo."}</p>
                          {source.excerpt && verifiedValues.length > 0 && (
                            <p className="source-excerpt-highlighted">
                              {highlightVerifiedValues(source.excerpt, verifiedValues)}
                            </p>
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
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [inspectorTab, setInspectorTab] = useState("sources");
  const [selectedMeta, setSelectedMeta] = useState(null);
  const [sourceRequest, setSourceRequest] = useState(null);
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
    setInspectorOpen(false);
    setSourceRequest(null);
    setInput("");
    setLoading(false);
    setConversationId(uid());
  }

  function openConversation(conversation) {
    activeController.current?.abort();
    const restoredMessages = Array.isArray(conversation?.messages) ? conversation.messages : [];
    setConversationId(conversation.id);
    setMessages(restoredMessages);
    setSelectedMeta(null);
    setInspectorOpen(false);
    setSourceRequest(null);
    setInput("");
    setLoading(false);
    setSidebarOpen(false);
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
      const history = messages
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
          error: true, createdAt: new Date().toISOString(),
        }]);
      }
    } finally {
      activeController.current = null;
      setLoading(false);
    }
  }

  function inspect(meta, sourceIndex = null) {
    setSelectedMeta(meta);
    setSourceRequest({ id: uid(), meta, index: sourceIndex });
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
        conversations={conversations}
        activeConversationId={conversationId}
        onNewChat={newConversation}
        onConversationPick={openConversation}
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
                  suggestions={featuredQuestions}
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

      <Inspector open={inspectorOpen} onClose={() => setInspectorOpen(false)} meta={selectedMeta} tab={inspectorTab} setTab={setInspectorTab} developerMode={developerMode} sourceRequest={sourceRequest} />

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
