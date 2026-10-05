"""Validated canonical frame -> processed, model-ready daily file.

- Rows on non-trading dates are excluded (they are flagged, not repaired).
- `ret` is the daily LOG TOTAL return from adjusted closes, so dividends
  are included (v0.2 used unadjusted closes and lost ~5 pp/year on HYG).
- No clipping, no winsorizing, no filling: flags travel with the data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alphaquant.data.validation import FLAG_COLS

PROCESSED_COLS = [
    "date", "ticker", "open", "high", "low", "close", "volume",
    "adj_close", "dividend", "split_factor", "ret",
    *FLAG_COLS, "flag_any", "source",
]


def to_processed(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.loc[~frame["flag_non_session"]].copy()
    df["ret"] = np.log(df["adj_close"]).diff()
    df = df.iloc[1:]  # first row has no previous close
    df["flag_any"] = df[FLAG_COLS].any(axis=1)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df[PROCESSED_COLS].reset_index(drop=True)
