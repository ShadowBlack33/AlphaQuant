"""Read-only FastAPI backend for the AlphaQuant dashboard.

Every endpoint reads a file already produced by the pipelines
(data/raw/ and artifacts/risk/) and returns it as JSON. No computation
happens here: the pipelines stay the single source of truth.

Paths are resolved at request time through alphaquant.core.paths, so the
server works from any working directory.

Missing folder-level outputs return an empty list (200): the frontend
renders a "no data yet" state. Per-ticker endpoints return 404 when that
ticker's file is missing.

Run: python -m uvicorn alphaquant.api.main:app --reload --port 8000
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from alphaquant import __version__
from alphaquant.core import paths

app = FastAPI(title="AlphaQuant API", version=__version__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


TICKER_RE = re.compile(r"^[A-Za-z0-9.\-]{1,15}$")


def _validated_ticker(ticker: str) -> str:
    """Reject anything that is not a plain ticker symbol, so a path
    parameter can never be used to read files outside artifacts/risk/."""
    if not TICKER_RE.fullmatch(ticker):
        raise HTTPException(status_code=422, detail="Ticker invalido")
    return ticker


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
    raw = paths.raw_dir()
    if not raw.exists():
        return []
    return sorted({f.stem.split("_")[0] for f in raw.glob("*_1d.csv")})


@app.get("/api/garch-summary")
def garch_summary() -> list[dict]:
    return _read_csv_records(paths.risk_dir() / "garch_summary.csv")


@app.get("/api/regime-summary")
def regime_summary() -> list[dict]:
    return _read_csv_records(paths.risk_dir() / "regime_summary.csv")


@app.get("/api/risk-overall")
def risk_overall() -> list[dict]:
    return _read_csv_records(paths.risk_dir() / "risk_overall_summary.csv")


@app.get("/api/portfolio-summary")
def portfolio_summary() -> dict:
    rows = _read_csv_records(paths.risk_dir() / "portfolio_summary.csv")
    return rows[0] if rows else {}


@app.get("/api/rolling-corr")
def rolling_corr() -> list[dict]:
    return _read_csv_records(paths.risk_dir() / "rolling_corr.csv")


@app.get("/api/vol-series/{ticker}")
def vol_series(ticker: str) -> list[dict]:
    ticker = _validated_ticker(ticker)
    p = paths.risk_dir() / "vol_series" / f"{ticker}_1d_garch_vol.csv"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"No hay serie de volatilidad para {ticker}")
    return _read_csv_records(p)


@app.get("/api/regime-labels/{ticker}")
def regime_labels(ticker: str) -> list[dict]:
    ticker = _validated_ticker(ticker)
    p = paths.risk_dir() / "regime_labels" / f"{ticker}_1d_regimes.csv"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"No hay etiquetas de regimen para {ticker}")
    return _read_csv_records(p)


@app.get("/api/risk-by-regime/{ticker}")
def risk_by_regime(ticker: str) -> list[dict]:
    ticker = _validated_ticker(ticker)
    p = paths.risk_dir() / "risk_by_regime" / f"{ticker}_1d_risk_by_regime.csv"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"No hay riesgo por regimen para {ticker}")
    return _read_csv_records(p)