export function isCalendarYearCitation(value) {
  return /^(?:18|19|20|21)\d{2}$/.test(String(value || "").trim());
}

export function annotateNumericCitations(content, citations = []) {
  if (!content || !Array.isArray(citations) || citations.length === 0) return content;
  const characters = Array.from(content);
  const ordered = citations
    .map((citation, index) => ({ citation, index }))
    .filter(({ citation }) => (
      Number.isInteger(citation?.start)
      && Number.isInteger(citation?.end)
      && citation.start >= 0
      && citation.end > citation.start
      && citation.end <= characters.length
    ))
    .sort((left, right) => left.citation.start - right.citation.start);

  let cursor = 0;
  let annotated = "";
  for (const { citation, index } of ordered) {
    if (citation.start < cursor) continue;
    const value = characters.slice(citation.start, citation.end).join("");
    if (value !== citation.value) continue;
    annotated += characters.slice(cursor, citation.start).join("");
    annotated += `[${value}](#numeric-citation-${index})`;
    cursor = citation.end;
  }
  return annotated + characters.slice(cursor).join("");
}

export function parseTabularSnippet(snippet) {
  const lines = String(snippet || "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length === 0) return null;

  const pipeLines = lines.filter((line) => line.includes("|"));
  if (pipeLines.length >= 2) {
    const cells = (line) => line
      .replace(/^\||\|$/g, "")
      .split("|")
      .map((cell) => cell.trim());
    const parsed = pipeLines.map(cells);
    const hasSeparator = parsed[1]?.every((cell) => /^:?-{3,}:?$/.test(cell));
    const headers = hasSeparator
      ? parsed[0]
      : parsed[0].map((_, index) => `Coluna ${index + 1}`);
    const rows = hasSeparator ? parsed.slice(2) : parsed;
    if (rows.length > 0) return { kind: "grid", headers, rows };
  }

  const entries = lines
    .map((line) => {
      const separator = line.indexOf(":");
      if (separator <= 0) return null;
      const label = line.slice(0, separator).trim();
      const value = line.slice(separator + 1).trim();
      if (!value || label.toLowerCase() === "fonte") return null;
      return [label, value];
    })
    .filter(Boolean);

  if (entries.length === 0) return null;
  return {
    kind: "key-value",
    headers: ["Campo", "Valor"],
    rows: entries,
  };
}

export function conceptualizeTabularSnippet(table, citedValue) {
  if (!table?.rows?.length) return null;
  const target = String(citedValue || "").trim();

  if (table.kind === "key-value") {
    const citedRow = table.rows.find((row) => String(row[1]).includes(target));
    if (!citedRow) return null;
    const context = table.rows
      .filter((row) => row !== citedRow)
      .map(([label, value]) => ({ label, value }));
    const genericLabels = new Set(["valor", "resultado", "total", "índice", "indice"]);
    const subjectEntry = genericLabels.has(citedRow[0].toLowerCase())
      ? context.find(({ label }) => /indicador|setor|categoria|atividade|região|regiao|produto/i.test(label))
      : null;
    const subject = subjectEntry?.value || citedRow[0];
    const qualifiers = context.filter((entry) => entry !== subjectEntry);
    return { subject, qualifiers, citedRow };
  }

  for (const row of table.rows) {
    const cellIndex = row.findIndex((cell) => String(cell).includes(target));
    if (cellIndex < 0) continue;
    return {
      subject: table.headers[cellIndex] || "indicador",
      qualifiers: row
        .map((value, index) => ({ label: table.headers[index], value }))
        .filter((_, index) => index !== cellIndex),
      citedRow: row,
    };
  }
  return null;
}
