"""Technical indicator feature engineering.

FIX (Fase 0/2): previously `etl/transform.py` computed SMA/EMA manually AND
also called `add_technical_features`, which itself did not compute SMA/EMA —
a duplicated, easy-to-desync responsibility split. Now this module owns ALL
technical indicators (SMA, EMA, RSI, MACD, Bollinger, ATR, lags) and
`transform_frame` simply forwards the full features config here.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def add_sma(df: pd.DataFrame, windows: list[int], col: str = "Close") -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        try:
            w = int(w)
            df[f"sma_{w}"] = df[col].rolling(w, min_periods=w).mean()
        except Exception:
            pass
    return df


def add_ema(df: pd.DataFrame, windows: list[int], col: str = "Close") -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        try:
            w = int(w)
            df[f"ema_{w}"] = df[col].ewm(span=w, adjust=False).mean()
        except Exception:
            pass
    return df


def add_rsi(df: pd.DataFrame, period: int = 14, col: str = "Close") -> pd.DataFrame:
    df = df.copy()
    delta = df[col].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, col: str = "Close") -> pd.DataFrame:
    df = df.copy()
    ema_fast = df[col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[col].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    df[f"macd_{fast}_{slow}"] = macd
    df[f"macd_signal_{signal}"] = macd_signal
    df[f"macd_hist_{fast}_{slow}_{signal}"] = macd_hist
    return df


def add_bbands(df: pd.DataFrame, window: int = 20, k: float = 2.0, col: str = "Close") -> pd.DataFrame:
    df = df.copy()
    ma = df[col].rolling(window, min_periods=window).mean()
    sd = df[col].rolling(window, min_periods=window).std(ddof=0)
    upper = ma + k * sd
    lower = ma - k * sd
    df[f"bb_ma_{window}"] = ma
    df[f"bb_upper_{window}_{k}"] = upper
    df[f"bb_lower_{window}_{k}"] = lower
    df[f"bb_pctB_{window}_{k}"] = (df[col] - lower) / (upper - lower + 1e-12)
    df[f"bb_bw_{window}_{k}"] = (upper - lower) / (ma + 1e-12)
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        (df["High"] - df["Low"]).abs(),
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df[f"atr_{period}"] = tr.ewm(span=period, adjust=False).mean()
    return df


def add_lags(df: pd.DataFrame, lags: list[int], col: str = "ret") -> pd.DataFrame:
    df = df.copy()
    for k in lags:
        df[f"{col}_lag_{k}"] = df[col].shift(k)
    return df


def add_technical_features(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Single entry point for ALL technical indicators driven by `features` config.

    `cfg` accepts (all optional):
      sma_windows, ema_windows, rsi_windows, macd: {fast,slow,signal},
      bollinger: {window,k}, atr_window, lags
    """
    cfg = cfg or {}
    df = df.copy()

    sma_windows = cfg.get("sma_windows", [])
    if sma_windows:
        df = add_sma(df, sma_windows)

    ema_windows = cfg.get("ema_windows", [])
    if ema_windows:
        df = add_ema(df, ema_windows)

    for p in cfg.get("rsi_windows", []):
        try:
            df = add_rsi(df, period=int(p))
        except Exception:
            pass

    macd_cfg = cfg.get("macd")
    if isinstance(macd_cfg, dict):
        f, s, sig = int(macd_cfg.get("fast", 12)), int(macd_cfg.get("slow", 26)), int(macd_cfg.get("signal", 9))
        df = add_macd(df, fast=f, slow=s, signal=sig)

    bb_cfg = cfg.get("bollinger")
    if isinstance(bb_cfg, dict):
        w, k = int(bb_cfg.get("window", 20)), float(bb_cfg.get("k", 2.0))
        df = add_bbands(df, window=w, k=k)

    atr_w = cfg.get("atr_window")
    if atr_w:
        df = add_atr(df, period=int(atr_w))

    lags = cfg.get("lags", [])
    if lags:
        df = add_lags(df, lags=list(map(int, lags)), col="ret")

    return df
