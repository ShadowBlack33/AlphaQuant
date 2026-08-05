import pandas as pd
from alphaquant.etl.transform import transform_frame


def test_no_nans_in_key_cols():
    df = pd.DataFrame({
        "Datetime": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
        "Open": [1.0, 1.1],
        "High": [1.05, 1.15],
        "Low": [0.95, 1.05],
        "Close": [1.0, 1.1],
        "Volume": [100, 120],
        "Ticker": ["MSFT", "MSFT"],
    })
    out = transform_frame(df, {"sma_windows": [], "ema_windows": []}, ticker="MSFT")
    assert out[["Datetime", "Ticker"]].isna().sum().sum() == 0
