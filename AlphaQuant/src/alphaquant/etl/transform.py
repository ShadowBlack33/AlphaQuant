from __future__ import annotations
import logging
import re
from typing import Dict, List
import numpy as np
import pandas as pd
from alphaquant.features.technical import add_technical_features


def _flatten_columns(cols) -> List[str]:
    if isinstance(cols, pd.MultiIndex):
        flat = []
        for tup in cols:
            parts = [str(x) for x in tup if x is not None and str(x) != ""]
            flat.append("_".join(parts).strip())
        return flat
    return [str(c) for c in cols]


def _normalize_core_names(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.copy()

    def norm_one(c: str) -> str:
        s = c.strip()
        low = s.lower()
        base = re.split(r"[_\s]", low)[0]
        if base in ("open", "high", "low", "close", "volume"):
            return base.capitalize()
        if low.replace(" ", "").replace("_", "") == "adjclose" or low.startswith("adj close"):
            return "AdjClose"
        if base in ("datetime", "date", "timestamp"):
            return "Datetime"
        if base in ("ticker", "symbol"):
            return "Ticker"
        return s

    df.columns = [norm_one(c) for c in df.columns]
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    for base_name in ("Open", "High", "Low", "Close", "Volume", "AdjClose"):
        candidates = [c for c in df.columns if c.startswith(base_name + "_")]
        for c in candidates:
            if c.endswith("_" + ticker) and base_name not in df.columns:
                df = df.rename(columns={c: base_name})
    return df


def transform_frame(df: pd.DataFrame, features_cfg: Dict, ticker: str) -> pd.DataFrame:
    df = df.copy()
    logging.info(f"Transformando datos para {ticker}...")
    df.columns = _flatten_columns(df.columns)
    df = _normalize_core_names(df, ticker=ticker)

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    # Returns
    if features_cfg.get("returns", "log") == "log":
        df["ret"] = np.log(df["Close"]).diff()
    else:
        df["ret"] = df["Close"].pct_change()

    # FIX (Fase 0/2): all technical indicators (incl. SMA/EMA) now come from a
    # single place -- alphaquant.features.technical.add_technical_features --
    # instead of being computed twice (previously here AND in models/features.py).
    df = add_technical_features(df, features_cfg or {})

    # Datetime and Ticker normalization
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    elif df.index.name and str(df.index.name).lower() in ("datetime", "date", "timestamp"):
        df = df.reset_index().rename(columns={df.index.name: "Datetime"})
        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    elif "Date" in df.columns:
        df = df.rename(columns={"Date": "Datetime"})
        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    else:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "Datetime"})
            df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
        else:
            raise ValueError("No se encontró columna temporal (Datetime/Date/Timestamp).")

    if "Ticker" not in df.columns or df["Ticker"].isna().any():
        df["Ticker"] = ticker

    df = df.dropna(subset=["Datetime"]).copy()
    df = df.sort_values("Datetime")
    df = df.drop_duplicates(subset=["Datetime", "Ticker"], keep="last").reset_index(drop=True)

    # Numeric cleanup: cast to float, remove inf, light winsorization
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        df[num_cols] = df[num_cols].astype(float)
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
        for c in num_cols:
            s = df[c]
            if s.notna().sum() > 100:
                q_low, q_hi = s.quantile(0.001), s.quantile(0.999)
                df[c] = s.clip(lower=q_low, upper=q_hi)

    if df[["Datetime", "Ticker"]].isna().any().any():
        raise ValueError("NaN en claves después de transform")
    if not df["Datetime"].is_monotonic_increasing:
        raise ValueError("Datetime no ascendente después de transform")
    return df
