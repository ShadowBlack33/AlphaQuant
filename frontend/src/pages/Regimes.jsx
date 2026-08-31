import { useEffect, useState } from "react";
import { api } from "../lib/api";
import RegimeTimeline from "../components/RegimeTimeline";
import DataTable from "../components/DataTable";
import { pct } from "../lib/format";

export default function Regimes() {
  const [tickers, setTickers] = useState([]);
  const [selected, setSelected] = useState(null);
  const [summary, setSummary] = useState([]);
  const [labels, setLabels] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([api.tickers(), api.regimeSummary()]).then(([t, s]) => {
      setTickers(t || []);
      setSummary(s || []);
      if (t && t.length) setSelected(t[0]);
      setLoading(false);
    });
  }, []);

  useEffect(() => {
    if (!selected) return;
    setLabels(null);
    api.regimeLabels(selected).then(setLabels);
  }, [selected]);

  if (loading) return <div className="loading">Cargando</div>;
  if (!tickers.length) return <p className="hint">No hay deteccion de regimenes aun. Corre el pipeline con el analisis de riesgo activado.</p>;

  const current = summary.find((r) => r.ticker === selected);

  const cols = [
    { key: "ticker", label: "Ticker" },
    { key: "current_regime", label: "Regimen actual", format: (v) => <span className={`pill ${v === "high_vol" ? "pill-down" : "pill-up"}`}>{v}</span> },
    { key: "pct_time_low_vol", label: "Tiempo tranquilo", numeric: true, format: (v) => pct(v, 1) },
    { key: "pct_time_high_vol", label: "Tiempo volatil", numeric: true, format: (v) => pct(v, 1) },
  ];

  return (
    <div>
      <h2>Regimenes de mercado</h2>
      <div className="ticker-select">
        {tickers.map((t) => (
          <button key={t} className={t === selected ? "chip chip-active" : "chip"} onClick={() => setSelected(t)}>
            {t}
          </button>
        ))}
      </div>

      <h3>{selected} · historia de regimenes {current ? `· ahora en ${current.current_regime}` : ""}</h3>
      <RegimeTimeline labels={labels} />

      <h3>Distribucion por ticker</h3>
      <DataTable rows={summary} columns={cols} />
    </div>
  );
}
