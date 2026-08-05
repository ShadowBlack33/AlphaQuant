import pandas as pd
from alphaquant.etl.transform import transform_frame


def test_transform_shapes():
    df = pd.DataFrame({
        "Datetime": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"], utc=True),
        "Open": [1, 2, 3],
        "High": [2, 3, 4],
        "Low": [0.5, 1.5, 2.5],
        "Close": [1.5, 2.5, 3.5],
        "Adj Close": [1.5, 2.5, 3.5],
        "Volume": [100, 200, 300],
        "Ticker": ["AAPL", "AAPL", "AAPL"],
        "Interval": ["1d", "1d", "1d"],
    })
    cfg = {"sma_windows": [2], "ema_windows": [2], "returns": "log"}
    out = transform_frame(df, cfg, ticker="AAPL")
    assert set(["sma_2", "ema_2", "ret"]).issubset(out.columns)
    assert out["Datetime"].is_monotonic_increasing
    assert out[["Datetime", "Ticker"]].isna().sum().sum() == 0
