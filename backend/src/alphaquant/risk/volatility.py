"""GARCH conditional volatility forecasting.

Fits univariate GARCH(p, q) models (via the `arch` package) to a returns
series and exposes:

- in-sample conditional volatility (the model's estimate of vol at each
  historical point)
- out-of-sample volatility forecasts (h-step-ahead)
- a folder-level runner (`run_for_folder`) reading `data/processed/*_1d.csv`
  and writing a summary CSV under `artifacts/risk/`

Expects the same `ret` column already computed by
`alphaquant.data.transform.transform_frame` (log or simple returns).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult

from alphaquant.core import paths
from alphaquant.data.calendars import is_crypto

# arch's optimizer is numerically happier with returns of O(1) rather than
# O(1e-2) (typical daily log returns). We scale up before fitting and scale
# back down on every output so callers always see volatility in the same
# units as the input returns.
DEFAULT_SCALE = 100.0
MIN_OBS = 50


def fit_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    mean: str = "Zero",
    vol: str = "GARCH",
    dist: str = "normal",
    scale: float = DEFAULT_SCALE,
) -> ARCHModelResult:
    """Fit a GARCH(p, q) model to a return series.

    `returns` should be a plain (non-percentage) return series, e.g. the
    `ret` column produced by `transform_frame` (log returns by default).
    NaNs are dropped before fitting.
    """
    r = pd.Series(returns).dropna().astype(float)
    if len(r) < MIN_OBS:
        raise ValueError(
            f"Se necesitan al menos {MIN_OBS} observaciones para ajustar GARCH, hay {len(r)}"
        )
    am = arch_model(r * scale, mean=mean, vol=vol, p=p, q=q, dist=dist)
    res = am.fit(disp="off")
    return res


def conditional_volatility(
    res: ARCHModelResult,
    scale: float = DEFAULT_SCALE,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> pd.Series:
    """In-sample conditional volatility, rescaled back to the input's units."""
    vol = res.conditional_volatility / scale
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    vol = vol.copy()
    vol.name = "cond_vol_annualized" if annualize else "cond_vol"
    return vol


def forecast_volatility(
    res: ARCHModelResult,
    horizon: int = 10,
    scale: float = DEFAULT_SCALE,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> pd.Series:
    """h-step-ahead volatility forecast from the last fitted observation.

    Returns a Series indexed h1..h{horizon} (h1 = next period).
    """
    fc = res.forecast(horizon=horizon, reindex=False)
    variance = fc.variance.iloc[-1]
    vol = np.sqrt(variance) / scale
    vol.index = [f"h{i + 1}" for i in range(horizon)]
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    vol = vol.copy()
    vol.name = "forecast_vol_annualized" if annualize else "forecast_vol"
    return vol


def load_returns(p: Path, ret_col: str = "ret") -> pd.Series:
    """Load the `ret` column from a processed `*_1d.csv` file, indexed by date.

    Accepts the v3 processed schema (`date`) and the v0.2 schema (`Datetime`).
    """
    df = pd.read_csv(p)
    date_col = "date" if "date" in df.columns else "Datetime"
    if date_col not in df.columns:
        raise ValueError(f"{p.name} no tiene columna de fecha (date/Datetime)")
    if ret_col not in df.columns:
        raise ValueError(f"Columna '{ret_col}' no encontrada en {p.name}")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).dropna(subset=[ret_col]).reset_index(drop=True)
    s = df.set_index(date_col)[ret_col]
    s.index.name = "date"
    s.name = ret_col
    return s


def analyze_file(
    p: Path,
    p_order: int = 1,
    q_order: int = 1,
    horizon: int = 10,
    ret_col: str = "ret",
    annualize: bool = True,
    periods_per_year: int = 252,
) -> Tuple[dict, pd.Series, pd.Series]:
    """Fit GARCH(p_order, q_order) on one file's returns and summarize results.

    Returns (summary_dict, conditional_volatility_series, forecast_series).
    """
    r = load_returns(p, ret_col=ret_col)
    res = fit_garch(r, p=p_order, q=q_order)
    cv = conditional_volatility(res, annualize=annualize, periods_per_year=periods_per_year)
    fc = forecast_volatility(res, horizon=horizon, annualize=annualize, periods_per_year=periods_per_year)

    summary = {
        "file": p.name,
        "ticker": p.stem.split("_")[0],
        "n_obs": int(res.nobs),
        "p": p_order,
        "q": q_order,
        "aic": float(res.aic),
        "bic": float(res.bic),
        "alpha[1]": float(res.params.get("alpha[1]", np.nan)),
        "beta[1]": float(res.params.get("beta[1]", np.nan)),
        "last_vol": float(cv.iloc[-1]),
        "forecast_vol_h1": float(fc.iloc[0]),
        f"forecast_vol_h{horizon}": float(fc.iloc[-1]),
    }
    return summary, cv, fc


def run_for_folder(
    folder: str | Path | None = None,
    pattern: str = "*_1d.csv",
    p_order: int = 1,
    q_order: int = 1,
    horizon: int = 10,
    ret_col: str = "ret",
    annualize: bool = True,
    periods_per_year: int = 252,
    save_summary: bool = True,
    summary_path: Path | None = None,
    save_series: bool = False,
    series_dir: Path | None = None,
) -> pd.DataFrame | None:
    """Fit GARCH per ticker across a folder of `*_1d.csv` files.

    Skips files that fail (too few observations, missing columns, non-convergence) with
    a warning instead of aborting the whole batch.
    """
    folder = Path(folder) if folder is not None else paths.processed_dir()
    summary_path = Path(summary_path) if summary_path is not None else paths.risk_dir() / "garch_summary.csv"
    series_dir = Path(series_dir) if series_dir is not None else paths.risk_dir() / "vol_series"
    rows = []
    for f in sorted(folder.glob(pattern)):
        try:
            # Crypto trades 365 days a year: annualizing with 252 understates
            # its volatility by ~17%.
            ppy = 365 if is_crypto(f.stem.split("_")[0]) else periods_per_year
            summary, cv, _fc = analyze_file(
                f, p_order=p_order, q_order=q_order, horizon=horizon,
                ret_col=ret_col, annualize=annualize, periods_per_year=ppy,
            )
            rows.append(summary)
            if save_series:
                series_dir.mkdir(parents=True, exist_ok=True)
                cv.to_frame(cv.name).to_csv(series_dir / f"{f.stem}_garch_vol.csv")
        except Exception as e:
            print(f"[WARN] GARCH fallo en {f.name}: {e}")

    if not rows:
        print("Sin resultados GARCH.")
        return None

    df = pd.DataFrame(rows)
    if save_summary:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(summary_path, index=False)
    return df