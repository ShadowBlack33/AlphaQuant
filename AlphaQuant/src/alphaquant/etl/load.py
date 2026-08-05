from __future__ import annotations
from pathlib import Path
import logging
from typing import List
import pandas as pd

def _strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); df.columns = [str(c).strip() for c in df.columns]; return df

def _coerce_keys(df: pd.DataFrame, out_path: Path, ticker_hint: str | None = None) -> pd.DataFrame:
    df = _strip_cols(df)
    lower = {c.lower(): c for c in df.columns}
    dt_src = None
    for cand in ("datetime","date","timestamp"):
        if cand in lower: dt_src = lower[cand]; break
    if dt_src is None and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index":"Datetime"}); dt_src = "Datetime"
    if dt_src and dt_src != "Datetime": df = df.rename(columns={dt_src:"Datetime"})
    if "Datetime" in df.columns: df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    tk_src = None
    for cand in ("ticker","symbol"):
        if cand in lower: tk_src = lower[cand]; break
    if tk_src and tk_src != "Ticker": df = df.rename(columns={tk_src:"Ticker"})
    if "Ticker" not in df.columns or df["Ticker"].isna().all():
        ticker = ticker_hint or out_path.stem.split("_")[0]
        df["Ticker"] = ticker
    return _strip_cols(df)

def save_csv_idempotent(df: pd.DataFrame, out_path: str | Path, dedupe_keys: List[str] = ["Datetime","Ticker"]) -> Path:
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    df = _coerce_keys(df.copy(), out_path)
    if out_path.exists():
        try:
            old = pd.read_csv(out_path)
            old = _coerce_keys(old, out_path)
            merged = pd.concat([old, df], ignore_index=True)
        except Exception as e:
            logging.warning(f"Problema leyendo {out_path}: {e}; guardo solo DF nuevo.")
            merged = df.copy()
    else:
        merged = df.copy()
    merged = _coerce_keys(merged, out_path)
    available_keys = [k for k in dedupe_keys if k in merged.columns]
    logging.info(f"Claves para dedupe: {available_keys}")
    if available_keys:
        merged = merged.drop_duplicates(subset=available_keys, keep="last")
    else:
        merged = merged.drop_duplicates(keep="last")
    if "Datetime" in merged.columns:
        merged = merged.sort_values(["Ticker","Datetime"] if "Ticker" in merged.columns else ["Datetime"])
    merged = merged.reset_index(drop=True)
    if "Datetime" in merged.columns and merged["Datetime"].isna().any():
        raise ValueError("NaT en 'Datetime' tras normalizar")
    if "Datetime" in merged.columns:
        if "Ticker" in merged.columns:
            for t, grp in merged.groupby("Ticker"):
                if not grp["Datetime"].is_monotonic_increasing:
                    raise ValueError(f"Datetime no ascendente para Ticker={t}")
        else:
            if not merged["Datetime"].is_monotonic_increasing:
                raise ValueError("Datetime no ascendente")
    merged.to_csv(out_path, index=False)
    logging.info(f"Saved: {out_path} ({len(merged)} rows)")
    return out_path
