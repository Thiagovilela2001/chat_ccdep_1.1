import { useEffect, useRef, useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  History,
  MessageSquarePlus,
  PanelLeftClose,
  RefreshCw,
  Settings,
  X,
} from "lucide-react";

import { isBackendReady } from "../lib/api";
import { relativeTime } from "../lib/format";

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

export function Sidebar({
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
