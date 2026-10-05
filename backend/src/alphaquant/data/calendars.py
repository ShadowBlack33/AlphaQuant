"""Trading calendars and cross-asset alignment.

Equities trade on NYSE sessions (no weekends or holidays). Crypto trades
every day. Mixing them naively leaves NaN on weekends (which is what broke
the v0.2 average correlation).
"""
from __future__ import annotations

from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd


def is_crypto(ticker: str) -> bool:
    """Crypto pairs in the universe are quoted against USD: BTC-USD, ETH-USD."""
    return ticker.upper().endswith("-USD")


@lru_cache(maxsize=1)
def _nyse():
    return xcals.get_calendar("XNYS")


def expected_dates(ticker: str, start, end) -> pd.DatetimeIndex:
    """Dates on which `ticker` should have a daily bar between start and end."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if end < start:
        return pd.DatetimeIndex([])
    if is_crypto(ticker):
        return pd.date_range(start, end, freq="D")
    cal = _nyse()
    start = max(start, cal.first_session)
    end = min(end, cal.last_session)
    return pd.DatetimeIndex(cal.sessions_in_range(start, end)).tz_localize(None)


def last_expected_date(ticker: str, today) -> pd.Timestamp:
    """Most recent date whose bar should already be published: the previous
    NYSE session for equities, yesterday (UTC) for crypto."""
    yesterday = pd.Timestamp(today).normalize() - pd.Timedelta(days=1)
    if is_crypto(ticker):
        return yesterday
    cal = _nyse()
    return pd.Timestamp(cal.date_to_session(yesterday, direction="previous")).tz_localize(None)


def aggregate_crypto_to_equity_sessions(returns: pd.DataFrame) -> pd.DataFrame:
    """Re-express crypto LOG returns on the equity calendar.

    A Monday crypto return becomes Friday-close -> Monday-close (Saturday +
    Sunday + Monday summed), so weekend risk is kept instead of dropped.
    Works because log returns are additive. Equity columns are untouched.
    Caveat: the crypto daily close is 00:00 UTC, about 4 hours after the
    NYSE close.
    """
    crypto_cols = [c for c in returns.columns if is_crypto(c)]
    equity_cols = [c for c in returns.columns if c not in crypto_cols]
    if not crypto_cols or not equity_cols:
        return returns

    sessions = returns[equity_cols].dropna(how="all").index
    out = returns[equity_cols].loc[sessions].copy()
    for col in crypto_cols:
        s = returns[col]
        first = s.first_valid_index()
        if first is None:
            out[col] = float("nan")
            continue
        cum = s.loc[first:].fillna(0.0).cumsum()
        on_sessions = cum.reindex(cum.index.union(sessions)).ffill().reindex(sessions)
        aligned = on_sessions.diff()
        # Before the crypto series starts there is no return, not a zero.
        aligned[sessions <= first] = float("nan")
        out[col] = aligned
    return out[returns.columns]
