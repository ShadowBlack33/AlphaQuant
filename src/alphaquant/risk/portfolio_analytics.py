"""Portfolio-level risk analytics: rolling correlation, VaR/CVaR, and
drawdown segmented by market regime.

Composes with alphaquant.risk.regime: `analyze_file` and `run_for_folder`
call `detect_regimes_for_file` internally and reuse its `regime_df` (which
already carries `ret` + `regime` per Datetime) rather than recomputing
GARCH/clustering separately.

VaR/CVaR convention: both are reported as *positive loss magnitudes*.
A `var_hist=0.03` at alpha=0.05 means "5% of days, the loss exceeds 3%".
CVaR (expected shortfall) is always >= VaR at the same alpha by
construction, since it averages the tail beyond the VaR quantile.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from alphaquant.risk.volatility import load_returns
from alphaquant.risk.regime import detect_regimes_for_file

MIN_OBS_FOR_RISK = 20


# ---------------------------------------------------------------------------
# Single-series risk metrics
# ---------------------------------------------------------------------------

def historical_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """Historical (empirical) VaR at level alpha, as a positive loss magnitude."""
    r = pd.Series(returns).dropna()
    if len(r) == 0:
        return float("nan")
    return float(-r.quantile(alpha))


def historical_cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """Historical CVaR / Expected Shortfall: mean loss beyond the VaR quantile."""
    r = pd.Series(returns).dropna()
    if len(r) == 0:
        return float("nan")
    threshold = r.quantile(alpha)
    tail = r[r <= threshold]
    if len(tail) == 0:
        tail = r.nsmallest(1)
    return float(-tail.mean())


def parametric_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """Gaussian (variance-covariance) VaR: assumes normally distributed
    returns, using the sample mean/std. Faster and smoother than the
    historical estimate but underestimates fat-tail risk."""
    r = pd.Series(returns).dropna()
    if len(r) == 0:
        return float("nan")
    mu, sigma = float(r.mean()), float(r.std(ddof=1))
    z = norm.ppf(alpha)
    return float(-(mu + z * sigma))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Continuous drawdown curve from a return series: 0 at each new high,
    negative while below the running peak."""
    r = pd.Series(returns).dropna()
    equity = (1.0 + r).cumprod()
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    dd.name = "drawdown"
    return dd


def max_drawdown(returns: pd.Series) -> float:
    """Worst drawdown observed, as a positive magnitude (0.25 = -25%)."""
    dd = drawdown_series(returns)
    if len(dd) == 0:
        return float("nan")
    return float(-dd.min())


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> float:
    """Annualized Sharpe ratio. `rf` is an annualized risk-free rate."""
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return float("nan")
    excess = r - rf / periods_per_year
    sigma = excess.std(ddof=1)
    if sigma == 0:
        return float("nan")
    return float(excess.mean() / sigma * np.sqrt(periods_per_year))


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """CAGR / max drawdown -- reward per unit of worst-case pain."""
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return float("nan")
    total_return = float((1.0 + r).prod() - 1.0)
    n_years = len(r) / periods_per_year
    if n_years <= 0:
        return float("nan")
    cagr = (1.0 + total_return) ** (1.0 / n_years) - 1.0
    mdd = max_drawdown(r)
    if mdd == 0:
        return float("nan")
    return float(cagr / mdd)


# ---------------------------------------------------------------------------
# Regime-conditioned risk
# ---------------------------------------------------------------------------

def risk_by_regime(
    regime_df: pd.DataFrame,
    ret_col: str = "ret",
    regime_col: str = "regime",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """VaR/CVaR/drawdown stats sliced by regime membership.

    `regime_df` is expected to be the output of
    `alphaquant.risk.regime.detect_regimes_for_file` (or anything with the
    same [ret_col, regime_col] shape): one row per Datetime, continuous in
    time. Drawdown is computed once over the *full* continuous series (so
    it reflects a real equity curve) and then grouped by regime -- this
    reports "how deep are drawdowns typically/at worst while the market is
    in this regime", not a drawdown of a synthetic concatenated sub-series.

    Sharpe/Calmar, by contrast, are computed directly on each regime's
    (discontiguous) return subset -- standard practice for regime-conditional
    performance stats, but keep in mind Calmar's CAGR/max-drawdown inside it
    treats those days as if consecutive, which is a simplification for
    regimes that are scattered across many separate episodes.
    """
    dd = drawdown_series(regime_df[ret_col])
    df = regime_df[[ret_col, regime_col]].copy()
    df["drawdown"] = dd.reindex(df.index)
    df = df.dropna(subset=["drawdown"])

    rows = []
    for regime, grp in df.groupby(regime_col):
        rows.append({
            "regime": regime,
            "n_obs": int(len(grp)),
            "mean_ret": float(grp[ret_col].mean()),
            "vol": float(grp[ret_col].std(ddof=0)),
            "var_hist": historical_var(grp[ret_col], alpha=alpha),
            "cvar_hist": historical_cvar(grp[ret_col], alpha=alpha),
            "avg_drawdown": float(grp["drawdown"].mean()),
            "worst_drawdown": float(-grp["drawdown"].min()),
            "sharpe": sharpe_ratio(grp[ret_col]),
            "calmar": calmar_ratio(grp[ret_col]),
        })
    return pd.DataFrame(rows).set_index("regime").sort_index()


def analyze_file(
    p: Path,
    ret_col: str = "ret",
    alpha: float = 0.05,
    p_order: int = 1,
    q_order: int = 1,
    n_regimes: int = 2,
    roll_window: int = 21,
    method: str = "kmeans",
    random_state: int = 42,
) -> Tuple[dict, pd.DataFrame]:
    """Fit GARCH + regimes for one ticker (via alphaquant.risk.regime) and
    compute overall + per-regime risk metrics.

    Returns (overall_summary_dict, risk_by_regime_table).
    """
    _regime_summary, regime_df = detect_regimes_for_file(
        p, p_order=p_order, q_order=q_order, n_regimes=n_regimes,
        roll_window=roll_window, ret_col=ret_col, method=method,
        random_state=random_state,
    )
    risk_table = risk_by_regime(regime_df, ret_col=ret_col, alpha=alpha)

    overall = {
        "file": p.name,
        "ticker": p.stem.split("_")[0],
        "n_obs": int(len(regime_df)),
        "var_hist": historical_var(regime_df[ret_col], alpha=alpha),
        "cvar_hist": historical_cvar(regime_df[ret_col], alpha=alpha),
        "max_drawdown": max_drawdown(regime_df[ret_col]),
        "sharpe": sharpe_ratio(regime_df[ret_col]),
        "calmar": calmar_ratio(regime_df[ret_col]),
        "current_regime": regime_df["regime"].iloc[-1],
    }
    return overall, risk_table


# ---------------------------------------------------------------------------
# Portfolio-level (multi-ticker)
# ---------------------------------------------------------------------------

def load_returns_matrix(
    folder: str | Path = "data/raw",
    pattern: str = "*_1d.csv",
    ret_col: str = "ret",
) -> pd.DataFrame:
    """Load the `ret` column from every ticker CSV in `folder`, aligned by
    Datetime, one column per ticker (column name = ticker, from filename
    stem before the first underscore). Files that fail to load are skipped
    with a warning."""
    folder = Path(folder)
    series = {}
    for f in sorted(folder.glob(pattern)):
        try:
            s = load_returns(f, ret_col=ret_col)
            ticker = f.stem.split("_")[0]
            series[ticker] = s
        except Exception as e:
            print(f"[WARN] No se pudo cargar {f.name} para la matriz de retornos: {e}")
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series)


def rolling_avg_correlation(returns_df: pd.DataFrame, window: int = 63) -> pd.Series:
    """Average pairwise rolling correlation across all ticker columns --
    a simple diversification-benefit proxy (lower = more diversified)."""
    tickers = list(returns_df.columns)
    if len(tickers) < 2:
        raise ValueError("Se necesitan al menos 2 tickers para calcular correlacion.")
    pairwise = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            pairwise.append(returns_df[tickers[i]].rolling(window).corr(returns_df[tickers[j]]))
    avg = pd.concat(pairwise, axis=1).mean(axis=1)
    avg.name = "avg_rolling_corr"
    return avg


def portfolio_returns(returns_df: pd.DataFrame, weights: Mapping[str, float] | None = None) -> pd.Series:
    """Combine per-ticker return columns into a single portfolio return
    series. Equal-weighted by default; rows with any missing ticker return
    are dropped (no look-ahead/partial-fill assumptions)."""
    df = returns_df.dropna(how="any")
    if df.empty:
        return pd.Series(dtype=float, name="portfolio_ret")
    if weights is None:
        w = pd.Series(1.0 / df.shape[1], index=df.columns)
    else:
        w = pd.Series(weights).reindex(df.columns).fillna(0.0)
        total = w.sum()
        if total == 0:
            raise ValueError("La suma de los pesos del portafolio es 0.")
        w = w / total
    port = (df * w).sum(axis=1)
    port.name = "portfolio_ret"
    return port


def run_for_folder(
    folder: str | Path = "data/raw",
    pattern: str = "*_1d.csv",
    ret_col: str = "ret",
    alpha: float = 0.05,
    corr_window: int = 63,
    weights: Mapping[str, float] | None = None,
    p_order: int = 1,
    q_order: int = 1,
    n_regimes: int = 2,
    roll_window: int = 21,
    method: str = "kmeans",
    random_state: int = 42,
    save_outputs: bool = True,
    output_dir: Path = Path("models/risk"),
) -> dict:
    """Full portfolio risk analysis over a folder of ticker CSVs:
    - portfolio-level: rolling avg correlation + portfolio VaR/CVaR/drawdown
    - per-ticker: VaR/CVaR/drawdown overall and split by regime

    Every stage degrades gracefully: fewer than 2 tickers skips the
    portfolio-level analysis (with a warning) rather than raising, and a
    single ticker's GARCH/regime failure is skipped rather than aborting
    the batch (same convention as risk.volatility/risk.regime).
    """
    folder = Path(folder)
    results: dict = {}

    returns_matrix = load_returns_matrix(folder, pattern=pattern, ret_col=ret_col)
    if returns_matrix.shape[1] >= 2:
        avg_corr = rolling_avg_correlation(returns_matrix, window=corr_window)
        port_ret = portfolio_returns(returns_matrix, weights=weights)
        avg_corr_valid = avg_corr.dropna()
        portfolio_summary = {
            "n_tickers": int(returns_matrix.shape[1]),
            "avg_rolling_corr_last": float(avg_corr_valid.iloc[-1]) if len(avg_corr_valid) else float("nan"),
            "portfolio_var_hist": historical_var(port_ret, alpha=alpha),
            "portfolio_cvar_hist": historical_cvar(port_ret, alpha=alpha),
            "portfolio_max_drawdown": max_drawdown(port_ret),
            "portfolio_sharpe": sharpe_ratio(port_ret),
            "portfolio_calmar": calmar_ratio(port_ret),
        }
        results["portfolio_summary"] = portfolio_summary
        if save_outputs:
            output_dir.mkdir(parents=True, exist_ok=True)
            avg_corr.to_frame("avg_rolling_corr").to_csv(output_dir / "rolling_corr.csv")
            pd.DataFrame([portfolio_summary]).to_csv(output_dir / "portfolio_summary.csv", index=False)
    else:
        print("[WARN] Se necesitan al menos 2 tickers para correlacion/VaR de portafolio; se omite ese analisis.")

    per_ticker_rows = []
    for f in sorted(folder.glob(pattern)):
        try:
            overall, risk_table = analyze_file(
                f, ret_col=ret_col, alpha=alpha, p_order=p_order, q_order=q_order,
                n_regimes=n_regimes, roll_window=roll_window, method=method,
                random_state=random_state,
            )
            per_ticker_rows.append(overall)
            if save_outputs:
                by_regime_dir = output_dir / "risk_by_regime"
                by_regime_dir.mkdir(parents=True, exist_ok=True)
                risk_table.to_csv(by_regime_dir / f"{f.stem}_risk_by_regime.csv")
        except Exception as e:
            print(f"[WARN] Analisis de riesgo fallo en {f.name}: {e}")

    if per_ticker_rows:
        df = pd.DataFrame(per_ticker_rows)
        results["per_ticker_summary"] = df
        if save_outputs:
            output_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_dir / "risk_overall_summary.csv", index=False)
    else:
        print("Sin resultados de riesgo por ticker.")

    return results