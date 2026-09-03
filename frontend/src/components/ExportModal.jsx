import { useState } from "react";
import { Check, Copy, Download, FileText, Printer, X } from "lucide-react";

export function ExportModal({
  isOpen = false,
  onClose,
  question = "",
  answer = "",
  sources = [],
  numericCitations = [],
  conversationHistory = [],
}) {
  const [copied, setCopied] = useState(false);

  if (!isOpen) return null;

  const exportDate = new Date().toLocaleString("pt-BR");

  const buildMarkdownReport = () => {
    let md = `# Relatório de Análise Documental — Nadia RAG\n`;
    md += `*Gerado em: ${exportDate}*\n\n`;
    md += `## Consulta\n> ${question}\n\n`;
    md += `## Síntese Analítica\n${answer}\n\n`;

    if (numericCitations && numericCitations.length > 0) {
      md += `## Citações e Fatos Numéricos Validados (${numericCitations.length})\n\n`;
      md += `| Valor | Documento de Origem | Página | Trecho de Evidência |\n`;
      md += `|---|---|---|---|\n`;
      for (const c of numericCitations) {
        const val = c.value || "";
        const f = c.file || "";
        const p = c.page || "—";
        const snip = (c.claim || c.snippet || "").replace(/\|/g, "\\|").replace(/\n/g, " ");
        md += `| **${val}** | \`${f}\` | ${p} | ${snip} |\n`;
      }
      md += `\n`;
    }

    if (sources && sources.length > 0) {
      md += `## Fontes Recuperadas (${sources.length})\n\n`;
      for (const [idx, s] of sources.entries()) {
        const score = Math.round((s.score || 0) * 100);
        md += `### ${idx + 1}. \`${s.file}\` (Pág: ${s.page || "—"} | Relevância: ${score}%)\n`;
        if (s.excerpt) {
          md += `> ${s.excerpt.slice(0, 300)}...\n\n`;
        }
      }
    }

    return md;
  };

  const handleCopy = async () => {
    const text = buildMarkdownReport();
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadMarkdown = () => {
    const text = buildMarkdownReport();
    const blob = new Blob([text], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `analise-nadia-${new Date().toISOString().slice(0, 10)}.md`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const handlePrint = () => {
    window.print();
  };

  return (
    <div
      className="export-modal-backdrop"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="export-modal-title"
    >
      <div className="export-modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="export-modal-header">
          <div className="export-modal-title-group">
            <FileText size={18} className="export-modal-icon" />
            <h3 id="export-modal-title">Exportar Relatório de Análise</h3>
          </div>
          <button
            type="button"
            className="export-modal-close-btn"
            onClick={onClose}
            aria-label="Fechar modal"
          >
            <X size={18} />
          </button>
        </div>

        <div className="export-modal-body">
          <p className="export-modal-desc">
            Exporte o relatório completo com a pergunta, resposta detalhada, tabela de citações numéricas validadas e fontes documentais.
          </p>

          <div className="export-preview-box">
            <div className="export-preview-header">
              <span className="export-preview-tag">Prévia Markdown</span>
              <span>{exportDate}</span>
            </div>
            <pre className="export-preview-text">{buildMarkdownReport().slice(0, 500)}...</pre>
          </div>
        </div>

        <div className="export-modal-actions">
          <button
            type="button"
            className="export-action-btn secondary"
            onClick={handleCopy}
          >
            {copied ? <Check size={15} /> : <Copy size={15} />}
            <span>{copied ? "Copiado!" : "Copiar Markdown"}</span>
          </button>

          <button
            type="button"
            className="export-action-btn secondary"
            onClick={handlePrint}
          >
            <Printer size={15} />
            <span>Imprimir / PDF</span>
          </button>

          <button
            type="button"
            className="export-action-btn primary"
            onClick={handleDownloadMarkdown}
          >
            <Download size={15} />
            <span>Baixar .md</span>
          </button>
        </div>
      </div>
    </div>
  );
}

export default ExportModal;
