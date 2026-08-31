import { useEffect, useState } from "react";
import { api } from "../lib/api";
import LineChart from "../components/LineChart";
import DataTable from "../components/DataTable";
import { pct, num } from "../lib/format";

export default function PortfolioRisk() {
  const [tickers, setTickers] = useState([]);
  const [selected, setSelected] = useState(null);
  const [overall, setOverall] = useState([]);
  const [corr, setCorr] = useState([]);
  const [byRegime, setByRegime] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([api.tickers(), api.riskOverall(), api.rollingCorr()]).then(([t, o, c]) => {
      setTickers(t || []);
      setOverall(o || []);
      setCorr(c || []);
      if (t && t.length) setSelected(t[0]);
      setLoading(false);
    });
  }, []);

  useEffect(() => {
    if (!selected) return;
    setByRegime([]);
    api.riskByRegime(selected).then((r) => setByRegime(r || []));
  }, [selected]);

  if (loading) return <div className="loading">Cargando</div>;
  if (!tickers.length) return <p className="hint">No hay analisis de riesgo aun. Corre el pipeline con el analisis de riesgo activado.</p>;

  const overallCols = [
    { key: "ticker", label: "Ticker" },
    { key: "var_hist", label: "VaR 5%", numeric: true, format: (v) => pct(v) },
    { key: "cvar_hist", label: "CVaR 5%", numeric: true, format: (v) => pct(v) },
    { key: "max_drawdown", label: "Max DD", numeric: true, format: (v) => pct(v) },
    { key: "sharpe", label: "Sharpe", numeric: true, format: (v) => num(v, 2) },
    { key: "calmar", label: "Calmar", numeric: true, format: (v) => num(v, 2) },
    { key: "current_regime", label: "Regimen", format: (v) => <span className={`pill ${v === "high_vol" ? "pill-down" : "pill-up"}`}>{v}</span> },
  ];

  const byRegimeCols = [
    { key: "regime", label: "Regimen" },
    { key: "n_obs", label: "Dias", numeric: true },
    { key: "var_hist", label: "VaR 5%", numeric: true, format: (v) => pct(v) },
    { key: "cvar_hist", label: "CVaR 5%", numeric: true, format: (v) => pct(v) },
    { key: "worst_drawdown", label: "Peor DD", numeric: true, format: (v) => pct(v) },
    { key: "sharpe", label: "Sharpe", numeric: true, format: (v) => num(v, 2) },
  ];

  const validCorr = corr.filter((r) => r.avg_rolling_corr !== null && r.avg_rolling_corr !== undefined);

  return (
    <div>
      <h2>Riesgo de portafolio</h2>

      <h3>Correlacion promedio entre activos (rolling)</h3>
      <LineChart
        points={validCorr.map((r) => r.avg_rolling_corr)}
        dates={validCorr.map((r) => r.Datetime)}
        color="var(--text-dim)"
        format={(v) => v.toFixed(2)}
      />

      <h3>Riesgo por activo</h3>
      <DataTable rows={overall} columns={overallCols} />

      <h3>Riesgo condicionado al regimen</h3>
      <div className="ticker-select">
        {tickers.map((t) => (
          <button key={t} className={t === selected ? "chip chip-active" : "chip"} onClick={() => setSelected(t)}>
            {t}
          </button>
        ))}
      </div>
      <DataTable rows={byRegime} columns={byRegimeCols} emptyLabel="Selecciona un ticker" />
    </div>
  );
}