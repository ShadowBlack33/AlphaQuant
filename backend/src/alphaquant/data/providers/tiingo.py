"""Tiingo end-of-day prices (US equities and ETFs).

Endpoint: https://api.tiingo.com/tiingo/daily/<ticker>/prices
Returns raw AND adjusted OHLCV plus divCash and splitFactor per day.
Free plan: ~50 requests/hour, 1,000/day; data is for internal use only.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import requests

from alphaquant.core.settings import require_secret
from alphaquant.data.http import DEFAULT_TIMEOUT, build_session
from alphaquant.data.providers.base import DataProvider, ProviderError, RawPayload
from alphaquant.data.schema import CANONICAL_COLS

BASE_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"

_FIELD_MAP = {
    "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume",
    "adjOpen": "adj_open", "adjHigh": "adj_high", "adjLow": "adj_low",
    "adjClose": "adj_close", "adjVolume": "adj_volume",
    "divCash": "dividend", "splitFactor": "split_factor",
}


class TiingoProvider(DataProvider):
    name = "tiingo"

    def __init__(self, token: str | None = None, session: requests.Session | None = None):
        self._token = token
        self._session = session or build_session()

    def _auth_header(self) -> dict[str, str]:
        token = self._token or require_secret("TIINGO_API_KEY")
        return {"Authorization": f"Token {token}"}

    def fetch(self, ticker: str, start: date, end: date | None = None) -> RawPayload:
        url = BASE_URL.format(ticker=ticker.lower())
        params = {"startDate": start.isoformat(), "resampleFreq": "daily", "format": "json"}
        if end is not None:
            params["endDate"] = end.isoformat()

        resp = self._session.get(url, params=params, headers=self._auth_header(),
                                 timeout=DEFAULT_TIMEOUT)
        if resp.status_code in (401, 403):
            raise ProviderError("Tiingo rechazo el token (401/403). Revisa TIINGO_API_KEY en .env")
        if resp.status_code == 404:
            raise ProviderError(f"Tiingo no tiene el ticker {ticker}")
        if resp.status_code != 200:
            raise ProviderError(f"Tiingo respondio {resp.status_code} para {ticker}: {resp.text[:200]}")

        records = resp.json()
        if not isinstance(records, list):
            raise ProviderError(f"Respuesta inesperada de Tiingo para {ticker}: {str(records)[:200]}")
        # The token travels in a header, so the persisted request has no secret.
        return RawPayload(provider=self.name, ticker=ticker,
                          request={"url": url, "params": params}, records=records)

    def normalize(self, payload: RawPayload) -> pd.DataFrame:
        df = pd.DataFrame(payload.records)
        missing = [k for k in ["date", *_FIELD_MAP] if k not in df.columns]
        if missing:
            raise ProviderError(f"Faltan campos en la respuesta de Tiingo: {missing}")
        df = df.rename(columns=_FIELD_MAP)
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
        df["ticker"] = payload.ticker
        df["source"] = self.name
        num_cols = [c for c in CANONICAL_COLS if c not in ("date", "ticker", "source")]
        df[num_cols] = df[num_cols].astype("float64")
        return df[CANONICAL_COLS].sort_values("date").reset_index(drop=True)
