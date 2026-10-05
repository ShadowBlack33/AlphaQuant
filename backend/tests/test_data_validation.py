import numpy as np
import pandas as pd
import pytest

from alphaquant.data.transform import to_processed
from alphaquant.data.validation import Status, robust_zscore_past_only, validate
from tests.data_fixtures import canonical_equity

START, TODAY = "2023-01-03", "2024-07-01"  # TODAY = day after last bar -> data is fresh


def _run(df, ticker="AAA"):
    return validate(df, ticker, START, TODAY)


def test_clean_data_passes():
    res = _run(canonical_equity())
    assert res.status is Status.PASS, (res.messages, res.stats["flags"])
    assert res.stats["missing_dates"] == 0


def test_contract_violations_fail_and_are_listed():
    df = canonical_equity()
    df.loc[10, "close"] = -1.0
    df = pd.concat([df, df.iloc[[20]]]).sort_values("date").reset_index(drop=True)
    res = _run(df)
    assert res.status is Status.FAIL
    joined = " ".join(res.schema_errors)
    assert "close" in joined and "date" in joined


def test_extreme_day_is_flagged_not_clipped():
    df = canonical_equity()
    i = 300
    for c in ["close", "adj_close", "low", "adj_low"]:
        df.loc[i:, c] *= 0.7  # a -30% crash that persists
    df.loc[i, "open"] = df.loc[i, "high"] = df.loc[i - 1, "close"] * 0.99
    df.loc[i, "adj_open"] = df.loc[i, "adj_high"] = df.loc[i - 1, "adj_close"] * 0.99
    res = _run(df)
    assert res.frame.loc[i, "flag_outlier"]
    processed = to_processed(res.frame)
    crash = processed.loc[processed["date"] == str(df.loc[i, "date"].date()), "ret"].item()
    assert crash == pytest.approx(np.log(0.7 * df.loc[i, "close"] / 0.7 / df.loc[i - 1, "close"]), rel=1e-9)
    assert crash < -0.3  # the full move survives: nothing was clipped


def test_outlier_score_uses_only_the_past():
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 600))
    full = robust_zscore_past_only(r, 252, 60)
    truncated = robust_zscore_past_only(r.iloc[:400], 252, 60)
    pd.testing.assert_series_equal(full.iloc[:400], truncated)


def test_stale_prices_flagged():
    df = canonical_equity()
    df.loc[101:102, "close"] = df.loc[100, "close"]  # 3 sessions with the same close
    df.loc[101:102, ["open", "high", "low"]] = df.loc[100, "close"]
    df.loc[101:102, "high"] = df.loc[100, "close"] * 1.001
    res = _run(df)
    assert res.frame.loc[102, "flag_stale"]
    assert res.status is Status.WARN


def test_ohlc_inconsistency_flagged():
    df = canonical_equity()
    df.loc[50, "high"] = df.loc[50, "low"] * 0.9
    res = _run(df)
    assert res.frame.loc[50, "flag_ohlc"]


def test_missing_sessions_warn_then_fail():
    df = canonical_equity()
    one_gap = _run(df.drop(index=[200]).reset_index(drop=True))
    assert one_gap.status is Status.WARN and one_gap.stats["missing_dates"] == 1
    long_gap = _run(df.drop(index=range(200, 205)).reset_index(drop=True))
    assert long_gap.status is Status.FAIL and long_gap.stats["max_missing_run"] == 5


def test_weekend_row_is_flagged_and_excluded_for_equities():
    df = canonical_equity()
    extra = df.iloc[[3]].copy()
    extra["date"] = pd.Timestamp("2023-01-07")  # a Saturday
    df = pd.concat([df, extra]).sort_values("date").reset_index(drop=True)
    res = _run(df)
    assert res.frame["flag_non_session"].sum() == 1
    assert "2023-01-07" not in set(to_processed(res.frame)["date"])


def test_dividend_is_in_total_return_and_not_flagged():
    df = canonical_equity(dividend=("2024-03-15", 2.0))
    res = _run(df)
    assert res.status is Status.PASS, res.stats["flags"]
    assert res.stats["dividends"] == 1
    p = to_processed(res.frame).set_index("date")
    raw_ret = np.log(p.loc["2024-03-15", "close"] / p["close"].shift(1).loc["2024-03-15"])
    assert p.loc["2024-03-15", "ret"] > raw_ret + 0.015  # ~2/100 dividend added back


def test_split_is_not_a_jump():
    res = _run(canonical_equity(split=("2024-02-01", 4.0)))
    assert not res.frame["flag_jump_no_action"].any()
    assert not res.frame["flag_adj_inconsistent"].any()
    assert res.stats["splits"] == 1


def test_unexplained_jump_and_adjustment_are_flagged():
    df = canonical_equity()
    df.loc[150:, ["open", "high", "low", "close"]] *= 1.4   # raw +40% with no event
    df.loc[150:, ["adj_open", "adj_high", "adj_low", "adj_close"]] *= 1.4
    df.loc[90, "adj_close"] *= 1.01  # adj changes without dividend/split
    df.loc[90, "adj_high"] = max(df.loc[90, "adj_high"], df.loc[90, "adj_close"])
    res = _run(df)
    assert res.frame.loc[150, "flag_jump_no_action"]
    assert res.frame.loc[90, "flag_adj_inconsistent"]


def test_stale_feed_is_reported():
    res = validate(canonical_equity(), "AAA", START, "2024-08-15")
    assert res.status is Status.WARN
    assert any("desactualizados" in m for m in res.messages)
