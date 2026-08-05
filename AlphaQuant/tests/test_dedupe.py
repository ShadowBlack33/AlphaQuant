import pandas as pd
from pathlib import Path
from alphaquant.etl.load import save_csv_idempotent


def test_save_csv_idempotent(tmp_path: Path):
    df1 = pd.DataFrame({
        "Datetime": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
        "Close": [10, 11],
        "Volume": [100, 200],
        "Ticker": ["AAPL", "AAPL"],
    })
    out = tmp_path / "AAPL_1d.csv"
    save_csv_idempotent(df1, out)

    df2 = pd.DataFrame({
        "Datetime": pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True),
        "Close": [11, 12],
        "Volume": [200, 300],
        "Ticker": ["AAPL", "AAPL"],
    })
    save_csv_idempotent(df2, out)
    merged = pd.read_csv(out, parse_dates=["Datetime"])
    assert len(merged) == 3
    assert merged["Datetime"].is_monotonic_increasing
