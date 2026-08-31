import { useEffect, useState } from "react";
import { api } from "../lib/api";

const DOT_COLOR = {
  low_vol: "var(--safe)",
  mid_vol: "var(--amber)",
  high_vol: "var(--danger)",
};

/** Signature element: a persistent ticker-tape strip showing every
 * ticker's current regime at a glance, right under the header -- the
 * project's core thesis (regime-aware risk) stated before any tab is
 * even opened. Wrapped in a non-scrolling parent so the edge fade stays
 * fixed in place while the tape itself scrolls underneath it. */
export default function RegimeTape() {
  const [rows, setRows] = useState(null);

  useEffect(() => {
    api.regimeSummary().then((data) => setRows(data || []));
  }, []);

  if (rows === null) return null;
  if (rows.length === 0) return null;

  return (
    <div className="regime-tape-wrap">
      <div className="regime-tape">
        {rows.map((r) => (
          <div className="tape-item" key={r.ticker}>
            <span className="tape-dot" style={{ background: DOT_COLOR[r.current_regime] || "var(--text-dim)" }} />
            <span className="tape-ticker">{r.ticker}</span>
            <span>{r.current_regime}</span>
          </div>
        ))}
      </div>
    </div>
  );
}