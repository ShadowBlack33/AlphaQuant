"""Read-only FastAPI backend for the AlphaQuant dashboard.

Every endpoint just reads a CSV already produced by
`alphaquant.cli.menu` (data/raw/, models/, models/risk/) and returns it as
JSON. No computation happens here -- this is purely a JSON view over the
pipeline's file outputs, kept intentionally thin so the pipeline stays the
single source of truth.

If the pipeline hasn't produced a given file yet (e.g. GARCH hasn't run),
folder-level endpoints return an empty list with a 200 rather than an
error -- the frontend is expected to render an empty/"no data yet" state.
Per-ticker endpoints (vol-series, regime-labels, risk-by-regime) return a
404 when that specific ticker's file is missing, since the frontend
already knows which tickers exist (from /api/tickers) before requesting
their detail.

Run with: uvicorn alphaquant.api.main:app --reload --port 8000
(from the repo root, so the relative data/models paths resolve correctly
-- same working-directory convention as alphaquant.cli.menu).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

DATA_DIR = Path("data/raw")
MODELS_DIR = Path("models")
RISK_DIR = MODELS_DIR / "risk"

app = FastAPI(title="AlphaQuant API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_csv_records(path: Path) -> list[dict]:
    """CSV -> list of JSON-safe dicts. Missing file -> empty list (not an
    error: absence of an optional pipeline output is a normal state)."""
    if not path.exists():
        return []
    df = pd.read_csv(path)
    df = df.where(pd.notnull(df), None)
    return df.to_dict(orient="records")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/tickers")
def tickers() -> list[str]:
    if not DATA_DIR.exists():
        return []
    return sorted({f.stem.split("_")[0] for f in DATA_DIR.glob("*_1d.csv")})


@app.get("/api/signals")
def signals() -> list[dict]:
    """Directional classifier summary (LogReg + RF ensemble, UP/DOWN)."""
    return _read_csv_records(MODELS_DIR / "prob_summary.csv")


@app.get("/api/metrics-full")
def metrics_full() -> list[dict]:
    """Regression model metrics (RMSE/MAE per ticker/model/split)."""
    return _read_csv_records(MODELS_DIR / "metrics_full.csv")


@app.get("/api/garch-summary")
def garch_summary() -> list[dict]:
    return _read_csv_records(RISK_DIR / "garch_summary.csv")


@app.get("/api/regime-summary")
def regime_summary() -> list[dict]:
    return _read_csv_records(RISK_DIR / "regime_summary.csv")


@app.get("/api/risk-overall")
def risk_overall() -> list[dict]:
    return _read_csv_records(RISK_DIR / "risk_overall_summary.csv")


@app.get("/api/portfolio-summary")
def portfolio_summary() -> dict:
    rows = _read_csv_records(RISK_DIR / "portfolio_summary.csv")
    return rows[0] if rows else {}


@app.get("/api/rolling-corr")
def rolling_corr() -> list[dict]:
    return _read_csv_records(RISK_DIR / "rolling_corr.csv")


@app.get("/api/vol-series/{ticker}")
def vol_series(ticker: str) -> list[dict]:
    p = RISK_DIR / "vol_series" / f"{ticker}_1d_garch_vol.csv"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"No hay serie de volatilidad para {ticker}")
    return _read_csv_records(p)


@app.get("/api/regime-labels/{ticker}")
def regime_labels(ticker: str) -> list[dict]:
    p = RISK_DIR / "regime_labels" / f"{ticker}_1d_regimes.csv"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"No hay etiquetas de regimen para {ticker}")
    return _read_csv_records(p)


@app.get("/api/risk-by-regime/{ticker}")
def risk_by_regime(ticker: str) -> list[dict]:
    p = RISK_DIR / "risk_by_regime" / f"{ticker}_1d_risk_by_regime.csv"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"No hay riesgo por regimen para {ticker}")
    return _read_csv_records(p)