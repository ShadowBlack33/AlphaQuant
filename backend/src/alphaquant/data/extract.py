from __future__ import annotations
import time
import pandas as pd

def _download_one(yf, ticker: str, start=None, end=None, interval="1d", retries=3, backoff=5):
    last_err = None
    for i in range(retries):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            if df is not None and len(df) > 0:
                return df
        except Exception as e:
            last_err = e
        time.sleep(backoff * (2 ** i))
    if last_err:
        raise last_err
    return pd.DataFrame()

def fetch_tickers(tickers, start=None, end=None, interval="1d") -> pd.DataFrame:
    import yfinance as yf
    frames = []
    for t in tickers:
        try:
            df = _download_one(yf, t, start=start, end=end, interval=interval, retries=3, backoff=5)
        except Exception:
            try:
                df = yf.download(t, period="max", interval=interval, progress=False, auto_adjust=False, threads=False)
            except Exception:
                df = pd.DataFrame()
        if df is None or len(df) == 0:
            continue
        df = df.copy().reset_index()
        # Aplana posible MultiIndex de columnas
        df.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in df.columns]
        df["Ticker"] = t
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
