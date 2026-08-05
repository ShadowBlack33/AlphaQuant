import numpy as np
import pandas as pd
import pytest

from alphaquant.risk.regime import (
    build_regime_features,
    relabel_by_volatility,
    fit_kmeans_regimes,
    fit_hmm_regimes,
    detect_regimes_for_file,
    run_for_folder,
    HAS_HMMLEARN,
)


def _two_regime_returns(n_low=300, n_high=300, low_sigma=0.004, high_sigma=0.03, seed=7) -> pd.Series:
    """Low-vol segment followed by a high-vol segment -- a case where
    regime detection has an unambiguous right answer."""
    rng = np.random.default_rng(seed)
    low = rng.normal(0, low_sigma, n_low)
    high = rng.normal(0, high_sigma, n_high)
    vals = np.concatenate([low, high])
    idx = pd.date_range("2020-01-01", periods=len(vals), freq="D", tz="UTC")
    return pd.Series(vals, index=idx, name="ret")


# ---------- build_regime_features ----------

def test_build_regime_features_columns_and_alignment():
    ret = pd.Series([0.01, -0.01, 0.02, -0.02, 0.0] * 20,
                     index=pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC"))
    cond_vol = pd.Series(np.linspace(0.1, 0.2, 100), index=ret.index)
    feats = build_regime_features(ret, cond_vol, roll_window=10)
    assert set(["ret", "cond_vol", "roll_mean_ret", "roll_vol"]).issubset(feats.columns)
    # first (roll_window - 1) rows must be dropped (incomplete rolling window)
    assert len(feats) == 100 - 9
    assert feats.isna().sum().sum() == 0


# ---------- relabel_by_volatility ----------

def test_relabel_by_volatility_orders_low_high():
    idx = pd.RangeIndex(6)
    features = pd.DataFrame({"cond_vol": [0.5, 0.5, 0.5, 5.0, 5.0, 5.0]}, index=idx)
    # cluster '1' happens to be the low-vol group and cluster '0' the high-vol
    # group -- relabel_by_volatility must not just trust the numeric id.
    raw_labels = [1, 1, 1, 0, 0, 0]
    regimes = relabel_by_volatility(features, raw_labels, vol_col="cond_vol")
    assert list(regimes) == ["low_vol", "low_vol", "low_vol", "high_vol", "high_vol", "high_vol"]


def test_relabel_by_volatility_three_regimes():
    idx = pd.RangeIndex(3)
    features = pd.DataFrame({"cond_vol": [10.0, 0.1, 1.0]}, index=idx)
    raw_labels = ["c_high", "c_low", "c_mid"]
    regimes = relabel_by_volatility(features, raw_labels, vol_col="cond_vol")
    assert regimes.tolist() == ["high_vol", "low_vol", "mid_vol"]


def test_relabel_by_volatility_more_than_three_regimes_uses_numbered_names():
    idx = pd.RangeIndex(4)
    features = pd.DataFrame({"cond_vol": [0.1, 0.2, 0.3, 0.4]}, index=idx)
    raw_labels = [0, 1, 2, 3]
    regimes = relabel_by_volatility(features, raw_labels, vol_col="cond_vol")
    assert regimes.tolist() == ["regime_0", "regime_1", "regime_2", "regime_3"]


# ---------- fit_kmeans_regimes ----------

def test_fit_kmeans_regimes_returns_labels_for_every_row():
    n = 200
    features = pd.DataFrame({
        "cond_vol": np.concatenate([np.full(100, 0.1), np.full(100, 0.5)]),
        "roll_vol": np.concatenate([np.full(100, 0.01), np.full(100, 0.05)]),
        "roll_mean_ret": np.zeros(n),
    })
    raw_labels, model, scaler = fit_kmeans_regimes(features, n_regimes=2)
    assert len(raw_labels) == n
    assert len(set(raw_labels)) == 2


# ---------- end-to-end: detect_regimes_for_file ----------

def test_detect_regimes_for_file_separates_low_and_high_vol_segments(tmp_path):
    r = _two_regime_returns(n_low=300, n_high=300)
    df = pd.DataFrame({"Datetime": r.index, "ret": r.values, "Ticker": "SIM"})
    p = tmp_path / "SIM_1d.csv"
    df.to_csv(p, index=False)

    summary, regime_df = detect_regimes_for_file(p, n_regimes=2, roll_window=21, method="kmeans")

    assert summary["ticker"] == "SIM"
    assert summary["method"] == "kmeans"
    assert summary["current_regime"] == "high_vol"  # series ends in the high-vol segment
    assert "pct_time_low_vol" in summary and "pct_time_high_vol" in summary

    # majority of the first (low-vol) segment should be labeled low_vol,
    # majority of the second (high-vol) segment should be labeled high_vol
    first_half = regime_df.iloc[: len(regime_df) // 2]
    second_half = regime_df.iloc[len(regime_df) // 2:]
    assert (first_half["regime"] == "low_vol").mean() > 0.7
    assert (second_half["regime"] == "high_vol").mean() > 0.7


def test_detect_regimes_for_file_too_few_observations_raises(tmp_path):
    r = _two_regime_returns(n_low=20, n_high=20)  # way below MIN_OBS for GARCH
    df = pd.DataFrame({"Datetime": r.index, "ret": r.values, "Ticker": "TINY"})
    p = tmp_path / "TINY_1d.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError):
        detect_regimes_for_file(p)


def test_detect_regimes_for_file_unknown_method_raises(tmp_path):
    r = _two_regime_returns(n_low=100, n_high=100)
    df = pd.DataFrame({"Datetime": r.index, "ret": r.values, "Ticker": "SIM"})
    p = tmp_path / "SIM_1d.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError):
        detect_regimes_for_file(p, method="not_a_real_method")


# ---------- HMM path (only if hmmlearn is installed) ----------

@pytest.mark.skipif(not HAS_HMMLEARN, reason="hmmlearn no esta instalado")
def test_fit_hmm_regimes_matches_kmeans_row_count():
    n = 200
    features = pd.DataFrame({
        "cond_vol": np.concatenate([np.full(100, 0.1), np.full(100, 0.5)]),
        "roll_vol": np.concatenate([np.full(100, 0.01), np.full(100, 0.05)]),
        "roll_mean_ret": np.zeros(n),
    })
    raw_labels, model, scaler = fit_hmm_regimes(features, n_regimes=2)
    assert len(raw_labels) == n
    assert len(set(raw_labels)) <= 2


def test_fit_hmm_regimes_raises_helpful_error_when_missing(monkeypatch):
    import alphaquant.risk.regime as regime_mod
    monkeypatch.setattr(regime_mod, "HAS_HMMLEARN", False)
    features = pd.DataFrame({"cond_vol": [0.1, 0.2, 0.3], "roll_vol": [0.01, 0.02, 0.03],
                              "roll_mean_ret": [0.0, 0.0, 0.0]})
    with pytest.raises(ImportError, match="hmmlearn"):
        regime_mod.fit_hmm_regimes(features, n_regimes=2)


# ---------- run_for_folder ----------

def test_run_for_folder_skips_bad_files_and_saves_summary(tmp_path):
    good = _two_regime_returns(n_low=150, n_high=150, seed=3)
    bad = pd.Series(np.random.normal(0, 0.01, 5))  # too few obs -> skipped

    for name, s in [("GOOD_1d.csv", good), ("BAD_1d.csv", bad)]:
        df = pd.DataFrame({"Datetime": pd.date_range("2020-01-01", periods=len(s), freq="D", tz="UTC"),
                            "ret": s.values, "Ticker": name.split("_")[0]})
        df.to_csv(tmp_path / name, index=False)

    out_path = tmp_path / "regime_summary.csv"
    df_summary = run_for_folder(folder=tmp_path, pattern="*_1d.csv", n_regimes=2,
                                 save_summary=True, summary_path=out_path)

    assert df_summary is not None
    assert len(df_summary) == 1
    assert df_summary.iloc[0]["ticker"] == "GOOD"
    assert out_path.exists()