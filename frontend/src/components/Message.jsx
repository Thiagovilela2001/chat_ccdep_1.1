import { memo, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  AlertTriangle,
  Check,
  Clock3,
  Copy,
  Database,
  Download,
  PanelRightOpen,
  RefreshCw,
  Search,
} from "lucide-react";

import { formatMs, relativeTime } from "../lib/format";
import { BrandMark } from "./BrandMark";
import { ProgressStepper } from "./ProgressStepper";

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

// memo: digitar no composer re-renderiza o App; sem isso, cada tecla reprocessa
// o markdown de toda a conversa. O App passa callbacks estáveis (useCallback).
export const Message = memo(function Message({ message, onInspect, onExport, onOpenDrawer, onRetry }) {
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
});

export function LoadingMessage() {
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
