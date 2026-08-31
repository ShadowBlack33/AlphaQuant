import { useEffect, useState } from "react";
import { api } from "../lib/api";
import LineChart from "../components/LineChart";
import DataTable from "../components/DataTable";
import { num, pct } from "../lib/format";

// GARCH's recursive variance starts from an initial estimate that can be
// far off the true early-window volatility, producing a transient spike
// before it converges. Trimming this burn-in from the *chart* (not the
// underlying data) is standard practice -- the model itself is unaffected.
const GARCH_WARMUP = 60;

export default function Volatility() {
  const [tickers, setTickers] = useState([]);
  const [selected, setSelected] = useState(null);
  const [summary, setSummary] = useState([]);
  const [series, setSeries] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([api.tickers(), api.garchSummary()]).then(([t, s]) => {
      setTickers(t || []);
      setSummary(s || []);
      if (t && t.length) setSelected(t[0]);
      setLoading(false);
    });
  }, []);

  useEffect(() => {
    if (!selected) return;
    setSeries(null);
    api.volSeries(selected).then(setSeries);
  }, [selected]);

  if (loading) return <div className="loading">Cargando</div>;
  if (!tickers.length) return <p className="hint">No hay resultados de GARCH aun. Corre el pipeline con el analisis de riesgo activado.</p>;

  const cols = [
    { key: "ticker", label: "Ticker" },
    { key: "aic", label: "AIC", numeric: true, format: (v) => num(v, 1) },
    { key: "alpha[1]", label: "Alpha", numeric: true, format: (v) => num(v) },
    { key: "beta[1]", label: "Beta", numeric: true, format: (v) => num(v) },
    { key: "last_vol", label: "Vol. actual", numeric: true, format: (v) => pct(v) },
    { key: "forecast_vol_h1", label: "Pronost. 1d", numeric: true, format: (v) => pct(v) },
  ];

  return (
    <div>
      <h2>Volatilidad condicional</h2>
      <div className="ticker-select">
        {tickers.map((t) => (
          <button key={t} className={t === selected ? "chip chip-active" : "chip"} onClick={() => setSelected(t)}>
            {t}
          </button>
        ))}
      </div>

      <h3>{selected} · volatilidad anualizada (GARCH)</h3>
      <LineChart
        points={(series || []).slice(GARCH_WARMUP).map((r) => r.cond_vol_annualized)}
        dates={(series || []).slice(GARCH_WARMUP).map((r) => r.Datetime)}
        color="var(--amber)"
      />

      <h3>Parametros del modelo por ticker</h3>
      <DataTable rows={summary} columns={cols} />
    </div>
  );
}