import { useMemo, useState } from "react";
import { BarChart3, LineChart as LineChartIcon, Table as TableIcon } from "lucide-react";

function parseNumericValue(val) {
  if (typeof val === "number") return val;
  if (!val) return null;
  const cleaned = String(val)
    .replace(/[R$\s%]/g, "")
    .replace(/\./g, "")
    .replace(",", ".");
  const num = parseFloat(cleaned);
  return Number.isFinite(num) ? num : null;
}

export function DataChart({ tableData = null, rawMarkdownTable = "" }) {
  const [viewMode, setViewMode] = useState("bar"); // "bar" | "line" | "table"
  const [hoveredIndex, setHoveredIndex] = useState(null);

  const parsed = useMemo(() => {
    if (tableData && tableData.columns && tableData.rows) {
      return tableData;
    }
    if (!rawMarkdownTable) return null;

    const lines = rawMarkdownTable
      .trim()
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l.startsWith("|") && l.endsWith("|"));

    if (lines.length < 3) return null;

    const columns = lines[0]
      .slice(1, -1)
      .split("|")
      .map((c) => c.trim());

    const rows = lines.slice(2).map((l) =>
      l
        .slice(1, -1)
        .split("|")
        .map((c) => c.trim())
    );

    return { columns, rows };
  }, [tableData, rawMarkdownTable]);

  const series = useMemo(() => {
    if (!parsed || parsed.rows.length < 2) return null;

    // First column is label, second column is primary metric
    const labels = [];
    const values = [];

    for (const row of parsed.rows) {
      if (row.length < 2) continue;
      const label = row[0];
      const val = parseNumericValue(row[1]);
      if (label && val !== null) {
        labels.push(label);
        values.push({ label, rawValue: row[1], numValue: val });
      }
    }

    if (values.length < 2) return null;

    const minVal = Math.min(0, ...values.map((v) => v.numValue));
    const maxVal = Math.max(...values.map((v) => v.numValue));
    const range = maxVal - minVal || 1;

    return {
      title: parsed.columns[1] || "Valores",
      labelHeader: parsed.columns[0] || "Período / Categoria",
      items: values,
      minVal,
      maxVal,
      range,
    };
  }, [parsed]);

  if (!parsed || !series) return null;

  const width = 540;
  const height = 240;
  const padding = { top: 30, right: 20, bottom: 50, left: 55 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  const zeroY =
    padding.top +
    chartHeight -
    ((0 - series.minVal) / series.range) * chartHeight;

  return (
    <div className="data-chart-card">
      <div className="data-chart-header">
        <div className="data-chart-title">
          <span>{series.title}</span>
          <span className="data-chart-subtitle">({series.labelHeader})</span>
        </div>
        <div className="data-chart-toggle-group">
          <button
            type="button"
            className={`data-chart-toggle-btn ${viewMode === "bar" ? "active" : ""}`}
            onClick={() => setViewMode("bar")}
            title="Gráfico de Barras"
          >
            <BarChart3 size={14} />
          </button>
          <button
            type="button"
            className={`data-chart-toggle-btn ${viewMode === "line" ? "active" : ""}`}
            onClick={() => setViewMode("line")}
            title="Gráfico de Linha"
          >
            <LineChartIcon size={14} />
          </button>
          <button
            type="button"
            className={`data-chart-toggle-btn ${viewMode === "table" ? "active" : ""}`}
            onClick={() => setViewMode("table")}
            title="Tabela de Dados"
          >
            <TableIcon size={14} />
          </button>
        </div>
      </div>

      <div className="data-chart-body">
        {viewMode === "table" ? (
          <div className="data-chart-table-wrap">
            <table className="data-chart-table">
              <thead>
                <tr>
                  {parsed.columns.map((col, idx) => (
                    <th key={idx}>{col}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {parsed.rows.map((row, rIdx) => (
                  <tr key={rIdx}>
                    {row.map((cell, cIdx) => (
                      <td key={cIdx}>{cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="data-chart-svg-wrap">
            <svg
              viewBox={`0 0 ${width} ${height}`}
              className="data-chart-svg"
              role="img"
              aria-label={`Gráfico de ${series.title}`}
            >
              {/* Eixo Zero */}
              <line
                x1={padding.left}
                y1={zeroY}
                x2={width - padding.right}
                y2={zeroY}
                className="chart-axis-zero"
              />

              {/* Grid Lines */}
              {[0, 0.5, 1].map((pct) => {
                const val = series.minVal + pct * series.range;
                const y = padding.top + chartHeight - pct * chartHeight;
                return (
                  <g key={pct}>
                    <line
                      x1={padding.left}
                      y1={y}
                      x2={width - padding.right}
                      y2={y}
                      className="chart-grid-line"
                    />
                    <text
                      x={padding.left - 8}
                      y={y + 4}
                      textAnchor="end"
                      className="chart-axis-text"
                    >
                      {val.toFixed(1)}
                    </text>
                  </g>
                );
              })}

              {/* Gráfico de Barras */}
              {viewMode === "bar" && (
                <g className="chart-bars-group">
                  {series.items.map((item, idx) => {
                    const barWidth = Math.max(12, (chartWidth / series.items.length) * 0.6);
                    const slotWidth = chartWidth / series.items.length;
                    const x = padding.left + idx * slotWidth + (slotWidth - barWidth) / 2;

                    const itemY =
                      padding.top +
                      chartHeight -
                      ((item.numValue - series.minVal) / series.range) * chartHeight;

                    const y = item.numValue >= 0 ? itemY : zeroY;
                    const h = Math.max(2, Math.abs(zeroY - itemY));
                    const isHovered = hoveredIndex === idx;

                    return (
                      <g
                        key={idx}
                        onMouseEnter={() => setHoveredIndex(idx)}
                        onMouseLeave={() => setHoveredIndex(null)}
                        className="chart-interactive-item"
                      >
                        <rect
                          x={x}
                          y={y}
                          width={barWidth}
                          height={h}
                          rx={3}
                          className={`chart-bar-rect ${isHovered ? "is-hovered" : ""}`}
                        />
                        <text
                          x={x + barWidth / 2}
                          y={height - padding.bottom + 16}
                          textAnchor="middle"
                          className="chart-axis-label-text"
                        >
                          {item.label.length > 9 ? `${item.label.slice(0, 8)}…` : item.label}
                        </text>
                      </g>
                    );
                  })}
                </g>
              )}

              {/* Gráfico de Linha */}
              {viewMode === "line" && (
                <g className="chart-line-group">
                  {/* Linha conectora */}
                  <polyline
                    fill="none"
                    strokeWidth="2.5"
                    className="chart-polyline"
                    points={series.items
                      .map((item, idx) => {
                        const slotWidth = chartWidth / series.items.length;
                        const x = padding.left + idx * slotWidth + slotWidth / 2;
                        const y =
                          padding.top +
                          chartHeight -
                          ((item.numValue - series.minVal) / series.range) * chartHeight;
                        return `${x},${y}`;
                      })
                      .join(" ")}
                  />
                  {/* Pontos */}
                  {series.items.map((item, idx) => {
                    const slotWidth = chartWidth / series.items.length;
                    const x = padding.left + idx * slotWidth + slotWidth / 2;
                    const y =
                      padding.top +
                      chartHeight -
                      ((item.numValue - series.minVal) / series.range) * chartHeight;
                    const isHovered = hoveredIndex === idx;

                    return (
                      <g
                        key={idx}
                        onMouseEnter={() => setHoveredIndex(idx)}
                        onMouseLeave={() => setHoveredIndex(null)}
                        className="chart-interactive-item"
                      >
                        <circle
                          cx={x}
                          cy={y}
                          r={isHovered ? 6 : 4}
                          className={`chart-line-point ${isHovered ? "is-hovered" : ""}`}
                        />
                        <text
                          x={x}
                          y={height - padding.bottom + 16}
                          textAnchor="middle"
                          className="chart-axis-label-text"
                        >
                          {item.label.length > 9 ? `${item.label.slice(0, 8)}…` : item.label}
                        </text>
                      </g>
                    );
                  })}
                </g>
              )}
            </svg>

            {/* Tooltip */}
            {hoveredIndex !== null && series.items[hoveredIndex] && (
              <div className="chart-tooltip">
                <strong>{series.items[hoveredIndex].label}</strong>: {series.items[hoveredIndex].rawValue}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default DataChart;
