"""Synthetic provider responses for offline tests."""
from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from alphaquant.data.calendars import expected_dates
from alphaquant.data.providers.base import DataProvider, RawPayload
from alphaquant.data.providers.coinbase import CoinbaseProvider
from alphaquant.data.providers.tiingo import TiingoProvider


def tiingo_records(start="2023-01-03", end="2024-06-28", seed=0, ticker="AAA",
                   dividend=None, split=None) -> list[dict]:
    """Realistic Tiingo-shaped rows on NYSE sessions.

    dividend=("YYYY-MM-DD", cash); split=("YYYY-MM-DD", ratio). Adjusted
    prices are built the way Tiingo does: factors applied to dates BEFORE
    each event, so adj/raw only changes on event dates.
    """
    dates = expected_dates(ticker, start, end)
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
    factors = np.ones(len(dates))
    div_cash = np.zeros(len(dates))
    split_f = np.ones(len(dates))
    if split:
        i = dates.get_loc(pd.Timestamp(split[0]))
        close[i:] /= split[1]
        factors[:i] /= split[1]
        split_f[i] = split[1]
    if dividend:
        i = dates.get_loc(pd.Timestamp(dividend[0]))
        cash = dividend[1]
        factors[:i] *= 1 - cash / close[i - 1]
        close[i:] -= cash
        div_cash[i] = cash
    open_ = close * (1 + rng.normal(0, 0.002, len(dates)))
    high = np.maximum(open_, close) * 1.004
    low = np.minimum(open_, close) * 0.996
    recs = []
    for k, d in enumerate(dates):
        f = factors[k]
        recs.append({
            "date": f"{d.date()}T00:00:00.000Z",
            "close": float(close[k]), "high": float(high[k]), "low": float(low[k]),
            "open": float(open_[k]), "volume": 1_000_000,
            "adjClose": float(close[k] * f), "adjHigh": float(high[k] * f),
            "adjLow": float(low[k] * f), "adjOpen": float(open_[k] * f),
            "adjVolume": 1_000_000, "divCash": float(div_cash[k]),
            "splitFactor": float(split_f[k]),
        })
    return recs


def coinbase_candles(start: date, end: date, seed=0) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D")
    close = 20_000 * np.exp(np.cumsum(rng.normal(0, 0.03, len(days))))
    out = []
    for d, c in zip(days, close):
        t = int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())
        out.append([t, c * 0.98, c * 1.02, c * 0.995, c, 1234.5])
    return out[::-1]  # Coinbase returns newest first


def canonical_equity(**kwargs) -> pd.DataFrame:
    recs = tiingo_records(**kwargs)
    payload = RawPayload("tiingo", kwargs.get("ticker", "AAA"), {"url": "x", "params": {}}, recs)
    return TiingoProvider(token="test").normalize(payload)


class FakeResponse:
    def __init__(self, status: int, payload):
        self.status_code = status
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeSession:
    """Mimics requests.Session.get and records every call."""

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[dict] = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {}), "headers": dict(headers or {})})
        return self.handler(url, params or {})


class StaticProvider(DataProvider):
    """Provider returning prebuilt records; counts fetch calls."""

    def __init__(self, name: str, records_by_ticker: dict, normalizer: DataProvider):
        self.name = name
        self._records = records_by_ticker
        self._normalizer = normalizer
        self.fetch_calls = 0

    def fetch(self, ticker, start, end=None):
        self.fetch_calls += 1
        if ticker not in self._records:
            from alphaquant.data.providers.base import ProviderError
            raise ProviderError(f"sin datos para {ticker}")
        request = {"url": f"fake://{ticker}", "windows": [{"start": f"{start}T00:00:00Z"}],
                   "params": {"startDate": str(start)}}
        return RawPayload(self.name, ticker, request, self._records[ticker])

    def normalize(self, payload):
        return self._normalizer.normalize(payload)


def coinbase_normalizer():
    return CoinbaseProvider(session=FakeSession(lambda u, p: FakeResponse(200, [])), pause_s=0)
