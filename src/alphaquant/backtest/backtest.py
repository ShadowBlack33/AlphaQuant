from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_PERIODS_PER_YEAR = 252


def backtest_signals(preds_file: str, threshold: float = 0.0, kind: str = "ret",
                      raw_dir: str = "data/raw") -> pd.DataFrame:
    pf = Path(preds_file)
    dfp = pd.read_csv(pf, parse_dates=["Datetime"])
    stem = pf.stem
    parts = stem.split("_")
    ticker = parts[0]
    interval = parts[1] if len(parts) > 1 else "1d"
    base = pd.read_csv(Path(raw_dir) / f"{ticker}_{interval}.csv", parse_dates=["Datetime"])
    m = dfp.merge(base[["Datetime", "ret"]], on="Datetime", how="left").dropna(subset=["ret"])

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
        signal = np.sign(m["ret"]).astype(int)

    m["signal"] = signal
    m["strategy_ret"] = m["signal"] * m["ret"]
    m["equity"] = (1 + m["strategy_ret"]).cumprod()
    m["bench"] = (1 + m["ret"]).cumprod()
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
