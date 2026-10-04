import numpy as np
import pandas as pd

from alphaquant.risk.portfolio_analytics import rolling_avg_correlation


def _mixed_calendar_returns(n=400, seed=1):
    """Two 'equities' (weekdays only, NaN on weekends) and two perfectly
    correlated 'crypto' assets (7 days a week)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    weekday = idx.dayofweek < 5

    c = rng.normal(0, 0.02, n)
    df = pd.DataFrame({"BTC": c, "ETH": 2 * c}, index=idx)
    for name in ("AAA", "BBB"):
        s = rng.normal(0, 0.01, n)
        df[name] = np.where(weekday, s, np.nan)
    return df


def test_avg_corr_not_inflated_by_crypto_pair():
    df = _mixed_calendar_returns()
    avg = rolling_avg_correlation(df, window=63).dropna()
    assert len(avg) > 0
    # Only 1 of 6 pairs (BTC-ETH) is perfectly correlated, the rest are
    # independent noise -> average must be far below 1.0.
    assert avg.mean() < 0.6


def test_avg_corr_uses_common_calendar():
    df = _mixed_calendar_returns()
    avg = rolling_avg_correlation(df, window=63)
    # Result index must be a subset of the weekday-only common calendar.
    assert avg.index.isin(df.dropna().index).all()
