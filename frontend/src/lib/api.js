const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function get(path) {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) {
    if (res.status === 404) return null;
    throw new Error(`API ${path} -> ${res.status}`);
  }
  return res.json();
}

export const api = {
  health: () => get("/api/health"),
  tickers: () => get("/api/tickers"),
  signals: () => get("/api/signals"),
  metricsFull: () => get("/api/metrics-full"),
  garchSummary: () => get("/api/garch-summary"),
  regimeSummary: () => get("/api/regime-summary"),
  riskOverall: () => get("/api/risk-overall"),
  portfolioSummary: () => get("/api/portfolio-summary"),
  rollingCorr: () => get("/api/rolling-corr"),
  volSeries: (ticker) => get(`/api/vol-series/${ticker}`),
  regimeLabels: (ticker) => get(`/api/regime-labels/${ticker}`),
  riskByRegime: (ticker) => get(`/api/risk-by-regime/${ticker}`),
};
