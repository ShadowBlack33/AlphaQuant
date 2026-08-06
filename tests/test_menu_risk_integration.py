import logging

import numpy as np
import pandas as pd

from alphaquant.cli.menu import run_risk_analytics
from alphaquant.utils.config import DEFAULT_CONFIG


def _two_regime_returns(n_low=150, n_high=150, low_sigma=0.004, high_sigma=0.03, seed=7) -> pd.Series:
    rng = np.random.default_rng(seed)
    low = rng.normal(0, low_sigma, n_low)
    high = rng.normal(0, high_sigma, n_high)
    vals = np.concatenate([low, high])
    idx = pd.date_range("2020-01-01", periods=len(vals), freq="D", tz="UTC")
    return pd.Series(vals, index=idx, name="ret")


def test_run_risk_analytics_end_to_end(tmp_path):
    """Simulates what run_everything_once does after ETL + training:
    a data_dir full of *_1d.csv files with a ret column, no yfinance or
    input() involved.
    """
    data_dir = tmp_path / "data_raw"
    data_dir.mkdir()
    output_dir = tmp_path / "models_risk"

    for ticker, seed in [("AAA", 1), ("BBB", 2)]:
        r = _two_regime_returns(seed=seed)
        df = pd.DataFrame({"Datetime": r.index, "ret": r.values, "Ticker": ticker})
        df.to_csv(data_dir / f"{ticker}_1d.csv", index=False)

    logger = logging.getLogger("test_risk_analytics")
    risk_cfg = DEFAULT_CONFIG["risk"]

    run_risk_analytics(
        data_dir=data_dir, interval="1d", risk_cfg=risk_cfg,
        seed=42, logger=logger, output_dir=output_dir,
    )

    garch_summary = pd.read_csv(output_dir / "garch_summary.csv")
    regime_summary = pd.read_csv(output_dir / "regime_summary.csv")

    assert set(garch_summary["ticker"]) == {"AAA", "BBB"}
    assert set(regime_summary["ticker"]) == {"AAA", "BBB"}
    assert (output_dir / "vol_series" / "AAA_1d_garch_vol.csv").exists()
    assert (output_dir / "regime_labels" / "AAA_1d_regimes.csv").exists()
    # both series ended in the high-vol half of the synthetic data
    assert (regime_summary.set_index("ticker")["current_regime"] == "high_vol").all()


def test_run_risk_analytics_survives_empty_folder(tmp_path):
    """No CSVs at all (e.g. every ticker failed ETL upstream) shouldn't
    raise -- it should just print/log and move on, since run_everything_once
    already returns early when ok == 0, but run_risk_analytics itself must
    not assume the folder is non-empty."""
    data_dir = tmp_path / "empty_raw"
    data_dir.mkdir()
    output_dir = tmp_path / "models_risk"
    logger = logging.getLogger("test_risk_analytics_empty")

    run_risk_analytics(
        data_dir=data_dir, interval="1d", risk_cfg=DEFAULT_CONFIG["risk"],
        seed=42, logger=logger, output_dir=output_dir,
    )
    # nothing to assert beyond "it didn't raise" -- no summary files expected
    assert not (output_dir / "garch_summary.csv").exists()