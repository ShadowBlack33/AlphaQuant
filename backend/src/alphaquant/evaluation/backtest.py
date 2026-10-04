from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_PERIODS_PER_YEAR = 252


def backtest_signals(preds_file: str, threshold: float = 0.0, kind: str = "ret",
                     raw_dir: str = "data/raw", cost_bps: float = 0.0) -> pd.DataFrame:
    """Backtest a predictions CSV against realised returns.

    Alignment convention: the prediction stamped at day t is a forecast for
    day t+1, so each signal is paired with the NEXT trading day's return
    (`ret_next`) of the base series. This is robust to sparse prediction
    files (it never relies on row adjacency in the predictions file).

    cost_bps: one-way cost in basis points, charged on every change of
    position (|signal_t - signal_{t-1}| * cost_bps / 10_000). Default 0.
    """
    pf = Path(preds_file)
    dfp = pd.read_csv(pf, parse_dates=["Datetime"])
    stem = pf.stem
    parts = stem.split("_")
    ticker = parts[0]
    interval = parts[1] if len(parts) > 1 else "1d"

    base = pd.read_csv(Path(raw_dir) / f"{ticker}_{interval}.csv", parse_dates=["Datetime"])
    base = base.sort_values("Datetime").reset_index(drop=True)
    base["ret_next"] = base["ret"].shift(-1)

    m = (dfp.merge(base[["Datetime", "ret", "ret_next"]], on="Datetime", how="left")
            .dropna(subset=["ret_next"])
            .sort_values("Datetime")
            .reset_index(drop=True))

    pred_col = None
    for c in ["y_pred", "pred", "yhat", "pred_ret"]:
        if c in m.columns:
            pred_col = c
            break
    if kind == "ret" and pred_col:
        signal = (m[pred_col] > threshold).astype(int) - (m[pred_col] < -threshold).astype(int)
    elif "proba_up" in m.columns:
        signal = (m["proba_up"] >= 0.5).astype(int) - (m["proba_up"] < 0.5).astype(int)
    else:
        signal = np.sign(m["ret_next"]).astype(int)

    m["signal"] = signal
    turnover = m["signal"].diff().abs().fillna(m["signal"].abs())
    m["cost"] = turnover * (cost_bps / 10_000.0)
    m["strategy_ret"] = m["signal"] * m["ret_next"] - m["cost"]
    m["equity"] = (1 + m["strategy_ret"]).cumprod()
    m["bench"] = (1 + m["ret_next"]).cumprod()
    return m


def summarize_backtest(df: pd.DataFrame, periods_per_year: int = DEFAULT_PERIODS_PER_YEAR) -> dict:
    strat = df["strategy_ret"]
    ret_total = df["equity"].iloc[-1] - 1.0
    bench_total = df["bench"].iloc[-1] - 1.0
    sharpe = strat.mean() / (strat.std() + 1e-12) * (periods_per_year ** 0.5)
    max_dd = float((df["equity"] / df["equity"].cummax() - 1.0).min())
    return {
        "ret_total": float(ret_total),
        "bench_total": float(bench_total),
        f"sharpe_{periods_per_year}": float(sharpe),
        "max_drawdown": max_dd,
    }
