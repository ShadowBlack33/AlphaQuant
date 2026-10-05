from datetime import date

import pandas as pd
import pytest

from alphaquant.data.providers import CoinbaseProvider, ProviderError, TiingoProvider
from alphaquant.data.schema import CANONICAL_COLS
from tests.data_fixtures import FakeResponse, FakeSession, coinbase_candles, tiingo_records


def test_tiingo_fetch_and_normalize():
    recs = tiingo_records(dividend=("2024-03-15", 1.5))
    session = FakeSession(lambda url, params: FakeResponse(200, recs))
    prov = TiingoProvider(token="secret-token", session=session)

    payload = prov.fetch("SPY", date(2023, 1, 1))
    df = prov.normalize(payload)

    assert list(df.columns) == CANONICAL_COLS
    assert df["date"].is_monotonic_increasing and df["date"].dt.tz is None
    assert df.loc[df["date"] == "2024-03-15", "dividend"].item() == 1.5
    assert session.calls[0]["url"].endswith("/tiingo/daily/spy/prices")
    assert session.calls[0]["headers"]["Authorization"] == "Token secret-token"


def test_tiingo_never_persists_the_token():
    session = FakeSession(lambda url, params: FakeResponse(200, tiingo_records()))
    payload = TiingoProvider(token="secret-token", session=session).fetch("SPY", date(2023, 1, 1))
    assert "secret-token" not in str(payload.request)


@pytest.mark.parametrize("status,needle", [(401, "token"), (404, "no tiene el ticker"), (500, "500")])
def test_tiingo_errors_are_explicit(status, needle):
    session = FakeSession(lambda url, params: FakeResponse(status, {"detail": "x"}))
    with pytest.raises(ProviderError, match=needle):
        TiingoProvider(token="secret-token", session=session).fetch("ZZZ", date(2023, 1, 1))


def _coinbase_handler(url, params):
    start = pd.Timestamp(params["start"][:10]).date()
    end = pd.Timestamp(params["end"][:10]).date()
    # Like the real API: may include one candle BEFORE the requested start.
    return FakeResponse(200, coinbase_candles(start - pd.Timedelta(days=1), end))


def test_coinbase_pages_in_windows_of_at_most_300_days():
    session = FakeSession(_coinbase_handler)
    prov = CoinbaseProvider(session=session, pause_s=0, today=date(2025, 1, 1))
    payload = prov.fetch("BTC-USD", date(2023, 1, 1))

    for call in session.calls:
        s = pd.Timestamp(call["params"]["start"][:10])
        e = pd.Timestamp(call["params"]["end"][:10])
        assert (e - s).days + 1 <= 300
    assert len(session.calls) == 3  # 731 days -> 300 + 300 + 131

    df = prov.normalize(payload)
    assert df["date"].is_unique and df["date"].is_monotonic_increasing
    assert df["date"].iloc[0] == pd.Timestamp("2023-01-01")
    # today's candle is still in progress and must never be included
    assert df["date"].iloc[-1] == pd.Timestamp("2024-12-31")
    assert (df["adj_close"] == df["close"]).all()
    assert (df["dividend"] == 0).all() and (df["split_factor"] == 1).all()


def test_coinbase_error_status():
    session = FakeSession(lambda url, params: FakeResponse(404, {"message": "NotFound"}))
    with pytest.raises(ProviderError, match="no tiene el producto"):
        CoinbaseProvider(session=session, pause_s=0, today=date(2024, 1, 10)).fetch(
            "XXX-USD", date(2024, 1, 1))
