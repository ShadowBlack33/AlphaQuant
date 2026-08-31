/** Generic ledger-style table. `columns`: [{ key, label, format?, numeric? }]
 * numeric columns are right-aligned with tabular monospace figures, the
 * way a real risk report lines up its numbers. */
export default function DataTable({ rows, columns, emptyLabel = "Sin datos aun" }) {
  if (!rows || rows.length === 0) {
    return <div className="chart-empty">{emptyLabel}</div>;
  }
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key} className={c.numeric ? "numeric" : ""}>{c.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {columns.map((c) => (
                <td key={c.key} className={c.numeric ? "numeric" : ""}>
                  {c.format ? c.format(row[c.key], row) : row[c.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
