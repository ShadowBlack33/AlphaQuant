"""Coinbase Exchange public daily candles (crypto, quoted in real USD).

Endpoint: https://api.exchange.coinbase.com/products/<product>/candles
No API key. Max 300 candles per request, so history is paged in windows.
Each row is [time, low, high, open, close, volume]; `time` is the bucket
start (00:00 UTC), so the close is the price at 24:00 UTC of that date.
"""
from __future__ import annotations

import time as _time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

from alphaquant.data.http import DEFAULT_TIMEOUT, build_session
from alphaquant.data.providers.base import DataProvider, ProviderError, RawPayload
from alphaquant.data.schema import CANONICAL_COLS

BASE_URL = "https://api.exchange.coinbase.com/products/{product}/candles"
GRANULARITY = 86_400
MAX_CANDLES = 300


class CoinbaseProvider(DataProvider):
    name = "coinbase"

    def __init__(self, session: requests.Session | None = None, pause_s: float = 0.35,
                 today: date | None = None):
        self._session = session or build_session()
        self._pause_s = pause_s
        self._today = today

    def _utc_today(self) -> date:
        return self._today or datetime.now(timezone.utc).date()

    def fetch(self, ticker: str, start: date, end: date | None = None) -> RawPayload:
        url = BASE_URL.format(product=ticker.upper())
        # Today's candle is still in progress: never request it.
        last_complete = self._utc_today() - timedelta(days=1)
        stop = min(end, last_complete) if end else last_complete

        rows: list[list[float]] = []
        windows: list[dict[str, str]] = []
        cursor = start
        while cursor <= stop:
            window_end = min(cursor + timedelta(days=MAX_CANDLES - 1), stop)
            params = {
                "granularity": str(GRANULARITY),
                "start": f"{cursor.isoformat()}T00:00:00Z",
                "end": f"{window_end.isoformat()}T23:59:59Z",
            }
            resp = self._session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
            if resp.status_code == 404:
                raise ProviderError(f"Coinbase no tiene el producto {ticker}")
            if resp.status_code != 200:
                raise ProviderError(
                    f"Coinbase respondio {resp.status_code} para {ticker}: {resp.text[:200]}")
            batch = resp.json()
            if not isinstance(batch, list):
                raise ProviderError(f"Respuesta inesperada de Coinbase: {str(batch)[:200]}")
            rows.extend(batch)
            windows.append(params)
            cursor = window_end + timedelta(days=1)
            if self._pause_s:
                _time.sleep(self._pause_s)

        return RawPayload(provider=self.name, ticker=ticker,
                          request={"url": url, "windows": windows}, records=rows)

    def normalize(self, payload: RawPayload) -> pd.DataFrame:
        if not payload.records:
            raise ProviderError(f"Coinbase no devolvio velas para {payload.ticker}")
        df = pd.DataFrame(payload.records, columns=["time", "low", "high", "open", "close", "volume"])
        df["date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None).dt.normalize()
        # Windows can overlap and the API may return candles before `start`.
        df = df.drop_duplicates(subset="date", keep="last").sort_values("date")
        start = pd.Timestamp(payload.request["windows"][0]["start"][:10])
        df = df[df["date"] >= start]
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype("float64")
        # No corporate actions in crypto: adjusted == raw.
        for c in ["open", "high", "low", "close", "volume"]:
            df[f"adj_{c}"] = df[c]
        df["dividend"] = 0.0
        df["split_factor"] = 1.0
        df["ticker"] = payload.ticker
        df["source"] = self.name
        return df[CANONICAL_COLS].reset_index(drop=True)
