import { useState } from "react";
import {
  Check,
  CheckCircle2,
  ChevronDown,
  Clock3,
  Code2,
  Database,
  FileText,
  Gauge,
  PanelRightClose,
  Route,
  Search,
} from "lucide-react";

import { formatMs } from "../lib/format";
import { isCalendarYearCitation } from "../lib/numericCitations";

function escapeRegExp(value) {
  return String(value || "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
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

export function Inspector({ open, onClose, meta, tab, setTab, developerMode, sourceRequest }) {
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
