import { useEffect, useMemo, useRef, useState } from "react";
import { BadgeCheck, CheckCircle2, Database, FileText, Filter, Search, X } from "lucide-react";

export function SourcesDrawer({
  isOpen = false,
  onClose,
  sources = [],
  numericCitations = [],
}) {
  const [searchQuery, setSearchQuery] = useState("");
  const [filterMode, setFilterMode] = useState("all"); // "all" | "cited" | "context"
  const closeButtonRef = useRef(null);

  useEffect(() => {
    if (!isOpen) return undefined;
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
  }, [isOpen, onClose]);

  const citedFiles = useMemo(() => {
    const set = new Set();
    for (const citation of numericCitations || []) {
      if (citation && citation.file) {
        set.add(String(citation.file).toLowerCase());
      }
    }
    return set;
  }, [numericCitations]);

  const filteredSources = useMemo(() => {
    return (sources || []).map((source, originalIndex) => ({ source, originalIndex })).filter(({ source }) => {
      const fileName = String(source.file || "").toLowerCase();
      const excerpt = String(source.excerpt || "").toLowerCase();
      const page = String(source.page || "").toLowerCase();
      const query = searchQuery.trim().toLowerCase();

      const matchesSearch =
        !query ||
        fileName.includes(query) ||
        excerpt.includes(query) ||
        page.includes(query);

      if (!matchesSearch) return false;

      const isCited = citedFiles.has(fileName);
      if (filterMode === "cited") return isCited;
      if (filterMode === "context") return !isCited;
      return true;
    });
  }, [sources, searchQuery, filterMode, citedFiles]);

  if (!isOpen) return null;

  return (
    <aside
      className="sources-drawer-overlay"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="sources-drawer-title"
    >
      <div
        className="sources-drawer-panel"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="sources-drawer-header">
          <div className="sources-drawer-header-info">
            <Database size={18} className="sources-drawer-icon" />
            <h2 id="sources-drawer-title">Fontes Documentais</h2>
            <span className="sources-count-badge">{sources.length}</span>
          </div>
          <button
            ref={closeButtonRef}
            type="button"
            className="sources-drawer-close-btn"
            onClick={onClose}
            aria-label="Fechar painel de fontes"
          >
            <X size={18} />
          </button>
        </div>

        {/* Barra de Filtros e Busca */}
        <div className="sources-drawer-controls">
          <div className="sources-search-input-wrap">
            <Search size={14} className="sources-search-icon" />
            <input
              type="text"
              className="sources-search-input"
              placeholder="Buscar em arquivos, páginas ou trechos..."
              aria-label="Buscar nas fontes documentais"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            {searchQuery && (
              <button
                type="button"
                className="sources-search-clear"
                onClick={() => setSearchQuery("")}
                aria-label="Limpar busca de fontes"
              >
                <X size={12} />
              </button>
            )}
          </div>

          <div className="sources-filter-tabs">
            <button
              type="button"
              className={`sources-tab-btn ${filterMode === "all" ? "active" : ""}`}
              onClick={() => setFilterMode("all")}
              aria-pressed={filterMode === "all"}
            >
              Todas ({sources.length})
            </button>
            <button
              type="button"
              className={`sources-tab-btn ${filterMode === "cited" ? "active" : ""}`}
              onClick={() => setFilterMode("cited")}
              aria-pressed={filterMode === "cited"}
            >
              Citadas ({sources.filter((s) => citedFiles.has(String(s.file).toLowerCase())).length})
            </button>
            <button
              type="button"
              className={`sources-tab-btn ${filterMode === "context" ? "active" : ""}`}
              onClick={() => setFilterMode("context")}
              aria-pressed={filterMode === "context"}
            >
              Contexto ({sources.filter((s) => !citedFiles.has(String(s.file).toLowerCase())).length})
            </button>
          </div>
        </div>

        {/* Lista de Fontes */}
        <div className="sources-drawer-list">
          {filteredSources.length === 0 ? (
            <div className="sources-empty-state">
              <Filter size={24} />
              <p>Nenhuma fonte encontrada com os filtros atuais.</p>
            </div>
          ) : (
            filteredSources.map(({ source, originalIndex }) => {
              const fileName = String(source.file || "");
              const isCited = citedFiles.has(fileName.toLowerCase());
              const scorePct = Math.round((source.score || 0) * 100);

              // Find citations that originated from this source
              const matchingCitations = (numericCitations || []).filter(
                (cit) =>
                  Number(cit.source_index) === originalIndex ||
                  String(cit.file || "").toLowerCase() === fileName.toLowerCase()
              );

              return (
                <article
                  key={`${fileName}-${source.page}-${originalIndex}`}
                  className={`source-card-item ${isCited ? "is-cited" : ""}`}
                  id={`source-card-${originalIndex}`}
                >
                  <div className="source-card-header">
                    <div className="source-card-title-group">
                      <span className="source-index-num">#{originalIndex + 1}</span>
                      <FileText size={15} className="source-doc-icon" />
                      <span className="source-card-filename" title={fileName}>
                        {fileName}
                      </span>
                    </div>

                    <div className="source-card-badges">
                      {isCited ? (
                        <span className="source-badge badge-cited" title="Esta fonte sustentou dados e números na resposta">
                          <CheckCircle2 size={11} /> Citada na resposta
                        </span>
                      ) : (
                        <span className="source-badge badge-context" title="Recuperada para contexto e cruzamento documental">
                          Contexto
                        </span>
                      )}
                      {source.page && (
                        <span className="source-badge badge-page">
                          Pág. {source.page}
                        </span>
                      )}
                      <span className="source-badge badge-score" title={`Relevância: ${scorePct}%`}>
                        {scorePct}% rel.
                      </span>
                    </div>
                  </div>

                  {/* Valores citados permanecem informativos; o texto da resposta é deliberadamente limpo. */}
                  {matchingCitations.length > 0 && (
                    <div className="source-cited-numbers-wrap">
                      <span className="source-cited-label">
                        <BadgeCheck size={12} /> Dados validados no texto:
                      </span>
                      <div className="source-cited-pills">
                        {matchingCitations.map((cit, cIdx) => (
                          <span
                            key={`${cit.value}-${cIdx}`}
                            className="source-data-pill"
                            title={`Valor citado na resposta: ${cit.value}`}
                          >
                            <strong>{cit.value}</strong>
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {source.excerpt && (
                    <div className="source-card-excerpt">
                      <p>{source.excerpt}</p>
                    </div>
                  )}
                </article>
              );
            })
          )}
        </div>
      </div>
    </aside>
  );
}

export default SourcesDrawer;
