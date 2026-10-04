import numpy as np
import pandas as pd

from alphaquant.evaluation.backtest import backtest_signals


def _write_fixture(tmp_path, n=80, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    ret = rng.normal(0, 0.01, n)

    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame({"Datetime": dates, "ret": ret}).to_csv(raw / "TICK_1d.csv", index=False)

    # Perfect-foresight predictions: prediction at day t == realised return at t+1.
    ret_next = np.append(ret[1:], 0.0)
    preds = tmp_path / "TICK_1d_preds.csv"
    pd.DataFrame({"Datetime": dates, "y_pred": ret_next}).to_csv(preds, index=False)
    return raw, preds


def test_signal_is_paired_with_next_day_return(tmp_path):
    raw, preds = _write_fixture(tmp_path)
    res = backtest_signals(str(preds), raw_dir=str(raw))
    # With perfect foresight and correct alignment, every day is a win.
    assert (res["strategy_ret"] >= 0).all()
    assert res["equity"].iloc[-1] > 1.0


def test_sparse_predictions_do_not_break_alignment(tmp_path):
    raw, preds = _write_fixture(tmp_path)
    sparse = pd.read_csv(preds, parse_dates=["Datetime"]).iloc[::7]
    sparse.to_csv(preds, index=False)
    res = backtest_signals(str(preds), raw_dir=str(raw))
    assert (res["strategy_ret"] >= 0).all()


def test_costs_reduce_returns(tmp_path):
    raw, preds = _write_fixture(tmp_path)
    free = backtest_signals(str(preds), raw_dir=str(raw), cost_bps=0.0)
    costly = backtest_signals(str(preds), raw_dir=str(raw), cost_bps=10.0)
    assert costly["equity"].iloc[-1] < free["equity"].iloc[-1]
