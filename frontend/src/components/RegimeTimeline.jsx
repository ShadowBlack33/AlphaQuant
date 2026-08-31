/** Horizontal strip of regime membership over time. Consecutive same-regime
 * days merge into one block so the strip reads as episodes, not noise. */
const REGIME_COLORS = {
  low_vol: "var(--safe)",
  mid_vol: "var(--amber)",
  high_vol: "var(--danger)",
};

const colorFor = (regime) => REGIME_COLORS[regime] || "var(--text-dim)";

export default function RegimeTimeline({ labels }) {
  if (!labels || labels.length === 0) {
    return <div className="chart-empty">Sin datos de regimen</div>;
  }

  const blocks = [];
  let start = 0;
  for (let i = 1; i <= labels.length; i++) {
    if (i === labels.length || labels[i].regime !== labels[start].regime) {
      blocks.push({ regime: labels[start].regime, count: i - start, from: labels[start].Datetime, to: labels[i - 1].Datetime });
      start = i;
    }
  }
  const total = labels.length;
  const uniqueRegimes = [...new Set(labels.map((l) => l.regime))];

  return (
    <div>
      <div className="regime-strip">
        {blocks.map((b, i) => (
          <div
            key={i}
            title={`${b.regime}: ${b.from?.slice(0, 10)} a ${b.to?.slice(0, 10)} (${b.count} dias)`}
            style={{ width: `${(b.count / total) * 100}%`, background: colorFor(b.regime) }}
          />
        ))}
      </div>
      <div className="regime-legend">
        {uniqueRegimes.map((r) => (
          <span key={r} className="legend-item">
            <span className="legend-dot" style={{ background: colorFor(r) }} />
            {r}
          </span>
        ))}
      </div>
    </div>
  );
}
