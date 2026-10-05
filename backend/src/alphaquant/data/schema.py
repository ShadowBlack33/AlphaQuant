"""Canonical daily price schema shared by every provider.

Every provider normalizes its response to these columns, so validation and
transformation never depend on where the data came from.

- open/high/low/close/volume: prices as traded (NOT adjusted)
- adj_*: adjusted for splits AND dividends (total-return basis). For crypto
  there are no corporate actions, so adj_* equals the raw columns.
- dividend: cash dividend per share on the ex-date (0 otherwise)
- split_factor: split ratio on the ex-date (1 otherwise)
"""
from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

PRICE_COLS = ["open", "high", "low", "close"]
ADJ_PRICE_COLS = ["adj_open", "adj_high", "adj_low", "adj_close"]
CANONICAL_COLS = [
    "date", "ticker",
    *PRICE_COLS, "volume",
    *ADJ_PRICE_COLS, "adj_volume",
    "dividend", "split_factor", "source",
]

_positive = pa.Check.gt(0)
_non_negative = pa.Check.ge(0)

CANONICAL_SCHEMA = pa.DataFrameSchema(
    columns={
        "date": pa.Column("datetime64[ns]", nullable=False, unique=True),
        "ticker": pa.Column(str, nullable=False),
        **{c: pa.Column(float, _positive, nullable=False) for c in PRICE_COLS + ADJ_PRICE_COLS},
        "volume": pa.Column(float, _non_negative, nullable=False),
        "adj_volume": pa.Column(float, _non_negative, nullable=False),
        "dividend": pa.Column(float, _non_negative, nullable=False),
        "split_factor": pa.Column(float, _positive, nullable=False),
        "source": pa.Column(str, nullable=False),
    },
    checks=[
        pa.Check(lambda df: df["date"].is_monotonic_increasing,
                 error="las fechas no estan ordenadas"),
        pa.Check(lambda df: df["ticker"].nunique() == 1,
                 error="mas de un ticker en el mismo archivo"),
    ],
    strict=True,
    coerce=True,
)


def empty_canonical() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in CANONICAL_COLS})
