import { useEffect, useState } from "react";
import { api } from "../lib/api";
import StatCard from "../components/StatCard";
import { pct, num } from "../lib/format";

export default function Overview() {
  const [portfolio, setPortfolio] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.portfolioSummary()
      .then((p) => setPortfolio(p))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="loading">Cargando</div>;

  const hasPortfolio = portfolio && Object.keys(portfolio).length > 0;

  return (
    <div>
      <h2>Resumen del portafolio</h2>
      {!hasPortfolio && <p className="hint">No hay analisis de riesgo aun. Corre el pipeline con al menos 2 tickers y el analisis de riesgo activado.</p>}
      {hasPortfolio && (
        <div className="stat-grid">
          <StatCard label="Tickers" value={portfolio.n_tickers} />
          <StatCard label="Correlacion prom." value={num(portfolio.avg_rolling_corr_last, 2)} />
          <StatCard label="VaR 5%" value={pct(portfolio.portfolio_var_hist)} tone="warn" />
          <StatCard label="CVaR 5%" value={pct(portfolio.portfolio_cvar_hist)} tone="danger" />
          <StatCard label="Max drawdown" value={pct(portfolio.portfolio_max_drawdown)} tone="danger" />
          <StatCard label="Sharpe" value={num(portfolio.portfolio_sharpe, 2)} tone="good" />
          <StatCard label="Calmar" value={num(portfolio.portfolio_calmar, 2)} tone="good" />
        </div>
      )}
    </div>
  );
}
