import { useState } from "react";

/** Dependency-free SVG area chart with a baseline, min/max ticks, and a
 * hover tooltip showing the date + value under the cursor -- useful for
 * spotting whether a spike is a real historical event or an artifact.
 * `format`: optional custom formatter (e.g. correlation should show as a
 * plain decimal, not a percentage, even though its magnitude is < 1 like
 * volatility). Falls back to the volatility-style %/decimal heuristic. */
export default function LineChart({ points, dates, width = 720, height = 190, color = "var(--amber)", unit = "", format }) {
  const [hoverIdx, setHoverIdx] = useState(null);

  if (!points || points.length < 2) {
    return <div className="chart-empty">Sin datos suficientes</div>;
  }
  const padX = 10;
  const padTop = 14;
  const padBottom = 18;
  const min = Math.min(...points);
  const max = Math.max(...points);
  const range = max - min || 1;
  const stepX = (width - padX * 2) / (points.length - 1);
  const yFor = (v) => height - padBottom - ((v - min) / range) * (height - padTop - padBottom);

  const coords = points.map((v, i) => [padX + i * stepX, yFor(v)]);
  const linePath = coords.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const areaPath = `${linePath} L${coords[coords.length - 1][0].toFixed(1)},${height - padBottom} L${coords[0][0].toFixed(1)},${height - padBottom} Z`;

  const fmt = format || ((v) => (Math.abs(v) < 1 ? (v * 100).toFixed(1) + "%" : v.toFixed(2)) + unit);

  const handleMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const relX = ((e.clientX - rect.left) / rect.width) * width;
    const idx = Math.round((relX - padX) / stepX);
    setHoverIdx(Math.max(0, Math.min(points.length - 1, idx)));
  };

  const hover = hoverIdx !== null ? coords[hoverIdx] : null;
  const hoverDate = hoverIdx !== null && dates ? dates[hoverIdx] : null;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="line-chart"
      preserveAspectRatio="none"
      onMouseMove={handleMove}
      onMouseLeave={() => setHoverIdx(null)}
    >
      <line x1={padX} y1={padTop} x2={padX} y2={height - padBottom} stroke="var(--rule)" strokeWidth="1" />
      <line x1={padX} y1={height - padBottom} x2={width - padX} y2={height - padBottom} stroke="var(--rule)" strokeWidth="1" />
      <path d={areaPath} fill={color} opacity="0.10" />
      <path d={linePath} fill="none" stroke={color} strokeWidth="1.5" />
      <text x={padX + 4} y={padTop + 4} fill="var(--text-dim)" fontSize="10" fontFamily="var(--font-mono)">{fmt(max)}</text>
      <text x={padX + 4} y={height - padBottom - 4} fill="var(--text-dim)" fontSize="10" fontFamily="var(--font-mono)">{fmt(min)}</text>

      {hover && (
        <>
          <line x1={hover[0]} y1={padTop} x2={hover[0]} y2={height - padBottom} stroke="var(--text-dim)" strokeWidth="1" strokeDasharray="2,2" />
          <circle cx={hover[0]} cy={hover[1]} r="3" fill={color} />
          <g transform={`translate(${Math.min(hover[0] + 8, width - 130)}, ${padTop + 2})`}>
            <rect width="120" height={hoverDate ? 32 : 18} fill="var(--paper-raised)" stroke="var(--rule)" />
            <text x="6" y="13" fill="var(--text)" fontSize="10" fontFamily="var(--font-mono)">{fmt(points[hoverIdx])}</text>
            {hoverDate && <text x="6" y="26" fill="var(--text-dim)" fontSize="9" fontFamily="var(--font-mono)">{String(hoverDate).slice(0, 10)}</text>}
          </g>
        </>
      )}
    </svg>
  );
}