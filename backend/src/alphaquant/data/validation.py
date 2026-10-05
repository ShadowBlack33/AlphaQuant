"""Data quality checks: validate, FLAG, never clip.

Two levels:
1. Contract (Pandera, `schema.CANONICAL_SCHEMA`): types, nulls, positive
   prices, unique and sorted dates. Any violation -> FAIL, the ticker is not
   published to data/processed.
2. Row flags (business rules with thresholds): suspicious rows are marked
   with `flag_*` columns and reported, but kept. Real extreme days (e.g.
   SPY -11.6% on 2020-03-16) are exactly what a risk model must see; v0.2
   clipped them away.

Outlier detection uses ONLY past data (rolling window shifted by one day),
so a flag on day t never depends on anything after t.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
import pandera.pandas as pa

from alphaquant.data.calendars import expected_dates, is_crypto, last_expected_date
from alphaquant.data.schema import CANONICAL_SCHEMA

FLAG_COLS = [
    "flag_non_session",
    "flag_ohlc",
    "flag_zero_volume",
    "flag_stale",
    "flag_outlier",
    "flag_jump_no_action",
    "flag_adj_inconsistent",
]


class Status(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class Thresholds:
    """Starting values (practitioner heuristics, not standards). Tune them by
    counting false positives on the real history."""
    stale_run_equity: int = 3          # same close on >= 3 consecutive sessions
    stale_run_crypto: int = 2          # crypto trades every day: 2 is already odd
    outlier_window: int = 252          # ~1 trading year of past returns
    outlier_min_periods: int = 60
    outlier_z: float = 8.0             # robust z-score (median / MAD)
    jump_no_action: float = 0.25       # |raw move| > 25% with no split/dividend
    adj_ratio_tol: float = 1e-4        # adj/raw ratio may only change on events
    missing_fail_pct: float = 0.01     # > 1% of expected dates missing -> FAIL
    missing_fail_run: int = 3          # > 3 consecutive missing dates -> FAIL
    coverage_grace_days: int = 10      # history may start a few days after start_date


@dataclass
class ValidationResult:
    ticker: str
    status: Status
    frame: pd.DataFrame | None
    schema_errors: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)


def _summarize_schema_errors(err: pa.errors.SchemaErrors, limit: int = 10) -> list[str]:
    fc = err.failure_cases
    out = []
    for (col, check), grp in fc.groupby(["column", "check"], dropna=False):
        examples = grp["failure_case"].astype(str).head(3).tolist()
        out.append(f"{col}: {check} ({len(grp)} casos, ej: {examples})")
    return out[:limit]


def _max_false_run(present: np.ndarray) -> int:
    best = cur = 0
    for ok in present:
        cur = 0 if ok else cur + 1
        best = max(best, cur)
    return best


def _run_lengths_of_equal(s: pd.Series) -> pd.Series:
    """For each row, how many consecutive rows (ending here) share its value."""
    changed = s.ne(s.shift())
    group = changed.cumsum()
    return s.groupby(group).cumcount() + 1


def robust_zscore_past_only(r: pd.Series, window: int, min_periods: int) -> pd.Series:
    past = r.shift(1)
    med = past.rolling(window, min_periods=min_periods).median()
    mad = past.rolling(window, min_periods=min_periods).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )
    scale = 1.4826 * mad
    z = (r - med) / scale.replace(0.0, np.nan)
    return z


def validate(df: pd.DataFrame, ticker: str, start, today,
             thresholds: Thresholds | None = None) -> ValidationResult:
    th = thresholds or Thresholds()
    crypto = is_crypto(ticker)

    if df is None or df.empty:
        return ValidationResult(ticker, Status.FAIL, None, ["sin filas"])

    # ---- 1. Contract
    try:
        frame = CANONICAL_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        return ValidationResult(ticker, Status.FAIL, None, _summarize_schema_errors(err))
    frame = frame.reset_index(drop=True).copy()

    # ---- 2. Calendar
    first, last = frame["date"].iloc[0], frame["date"].iloc[-1]
    expected = expected_dates(ticker, first, last)
    dates = pd.DatetimeIndex(frame["date"])
    frame["flag_non_session"] = False if crypto else ~dates.isin(expected)
    present = expected.isin(dates)
    n_missing = int((~present).sum())
    pct_missing = n_missing / len(expected) if len(expected) else 0.0
    max_missing_run = _max_false_run(present)

    # ---- 3. Row flags (computed on valid session rows only)
    ok = ~frame["flag_non_session"]
    s = frame.loc[ok]
    for col in FLAG_COLS[1:]:
        frame[col] = False

    lo_ok = s["low"] <= s[["open", "close"]].min(axis=1) * (1 + 1e-9)
    hi_ok = s[["open", "close"]].max(axis=1) <= s["high"] * (1 + 1e-9)
    frame.loc[s.index, "flag_ohlc"] = ~(lo_ok & hi_ok)

    frame.loc[s.index, "flag_zero_volume"] = s["volume"] == 0

    run_needed = th.stale_run_crypto if crypto else th.stale_run_equity
    same_close_run = _run_lengths_of_equal(s["close"])
    single_print = (s["high"] == s["low"]) & (s["volume"] > 0)
    frame.loc[s.index, "flag_stale"] = (same_close_run >= run_needed) | single_print

    r = np.log(s["adj_close"]).diff()
    z = robust_zscore_past_only(r, th.outlier_window, th.outlier_min_periods)
    frame.loc[s.index, "flag_outlier"] = z.abs() > th.outlier_z

    if not crypto:
        no_event = (s["dividend"] == 0) & (s["split_factor"] == 1)
        raw_move = (s["close"] / s["close"].shift(1) - 1).abs()
        frame.loc[s.index, "flag_jump_no_action"] = (raw_move > th.jump_no_action) & no_event
        ratio = s["adj_close"] / s["close"]
        ratio_change = (ratio / ratio.shift(1) - 1).abs()
        frame.loc[s.index, "flag_adj_inconsistent"] = (ratio_change > th.adj_ratio_tol) & no_event

    for col in FLAG_COLS:
        frame[col] = frame[col].astype(bool)

    # ---- 4. Coverage and freshness
    messages: list[str] = []
    start_ts = pd.Timestamp(start).normalize()
    if first > start_ts + pd.Timedelta(days=th.coverage_grace_days):
        messages.append(f"la historia empieza el {first.date()} (pedido desde {start_ts.date()})")
    expected_last = last_expected_date(ticker, today)
    stale_data = last < expected_last
    if stale_data:
        messages.append(f"datos desactualizados: ultimo {last.date()}, esperado {expected_last.date()}")
    if n_missing:
        messages.append(f"{n_missing} fechas faltantes ({pct_missing:.2%}), racha maxima {max_missing_run}")

    flag_counts = {c: int(frame[c].sum()) for c in FLAG_COLS}
    top = (z.abs().dropna().sort_values(ascending=False).head(5))
    outliers = [
        {"date": str(frame.loc[i, "date"].date()), "ret": round(float(r.loc[i]), 5),
         "z": round(float(z.loc[i]), 1)}
        for i in top.index if frame.loc[i, "flag_outlier"]
    ]
    stats = {
        "rows": int(len(frame)),
        "first_date": str(first.date()),
        "last_date": str(last.date()),
        "expected_dates": int(len(expected)),
        "missing_dates": n_missing,
        "missing_pct": round(pct_missing, 5),
        "max_missing_run": max_missing_run,
        "dividends": int((frame["dividend"] > 0).sum()),
        "splits": int((frame["split_factor"] != 1).sum()),
        "flags": flag_counts,
        "top_outliers": outliers,
        "worst_day": {"date": str(frame.loc[r.idxmin(), "date"].date()),
                      "ret": round(float(r.min()), 5)} if r.notna().any() else None,
    }

    if pct_missing > th.missing_fail_pct or max_missing_run > th.missing_fail_run:
        status = Status.FAIL
        messages.append("FAIL: demasiadas fechas faltantes")
    elif any(flag_counts.values()) or messages:
        status = Status.WARN
    else:
        status = Status.PASS

    return ValidationResult(ticker, status, frame, [], stats, messages)
