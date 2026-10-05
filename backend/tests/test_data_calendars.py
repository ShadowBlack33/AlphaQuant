import numpy as np
import pandas as pd

from alphaquant.data.calendars import (
    aggregate_crypto_to_equity_sessions,
    expected_dates,
    is_crypto,
    last_expected_date,
)


def test_is_crypto():
    assert is_crypto("BTC-USD") and is_crypto("eth-usd")
    assert not is_crypto("SPY") and not is_crypto("BRK-B")


def test_equity_calendar_skips_weekends_and_holidays():
    d = expected_dates("SPY", "2024-12-20", "2024-12-31")
    assert pd.Timestamp("2024-12-25") not in d  # Christmas
    assert pd.Timestamp("2024-12-21") not in d  # Saturday
    assert len(d) == 7


def test_crypto_calendar_is_every_day():
    assert len(expected_dates("BTC-USD", "2024-12-20", "2024-12-31")) == 12


def test_last_expected_date():
    assert last_expected_date("SPY", "2024-12-23") == pd.Timestamp("2024-12-20")  # Mon -> Fri
    assert last_expected_date("BTC-USD", "2024-12-23") == pd.Timestamp("2024-12-22")


def test_weekend_crypto_returns_are_summed_into_monday():
    days = pd.date_range("2024-01-01", "2024-01-14", freq="D")  # Mon..Sun x2
    crypto = pd.Series(0.01, index=days)
    equity = pd.Series(np.where(days.dayofweek < 5, 0.002, np.nan), index=days)
    out = aggregate_crypto_to_equity_sessions(pd.DataFrame({"SPY": equity, "BTC-USD": crypto}))

    assert out.index.dayofweek.max() < 5
    monday = pd.Timestamp("2024-01-08")
    assert out.loc[monday, "BTC-USD"] == pytest_approx(0.03)  # Sat + Sun + Mon
    assert out.loc[pd.Timestamp("2024-01-09"), "BTC-USD"] == pytest_approx(0.01)
    assert np.isnan(out.loc[pd.Timestamp("2024-01-01"), "BTC-USD"])  # no prior close
    pd.testing.assert_series_equal(out["SPY"], equity.dropna(), check_names=False)


def test_alignment_is_noop_without_crypto():
    df = pd.DataFrame({"A": [0.1, 0.2]}, index=pd.date_range("2024-01-01", periods=2))
    assert aggregate_crypto_to_equity_sessions(df) is df


def pytest_approx(x):
    import pytest
    return pytest.approx(x, rel=1e-12)
