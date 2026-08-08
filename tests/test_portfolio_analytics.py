import numpy as np
import pandas as pd
import pytest

from alphaquant.risk.portfolio_analytics import (
    historical_var,
    historical_cvar,
    parametric_var,
    drawdown_series,
    max_drawdown,
    sharpe_ratio,
    calmar_ratio,
    risk_by_regime,
    analyze_file,
    load_returns_matrix,
    rolling_avg_correlation,
    portfolio_returns,
    run_for_folder,
)


def _two_regime_returns(n_low=200, n_high=200, low_sigma=0.004, high_sigma=0.03, seed=7) -> pd.Series:
    rng = np.random.default_rng(seed)
    low = rng.normal(0, low_sigma, n_low)
    high = rng.normal(0, high_sigma, n_high)
    vals = np.concatenate([low, high])
    idx = pd.date_range("2020-01-01", periods=len(vals), freq="D", tz="UTC")
    return pd.Series(vals, index=idx, name="ret")


# ---------- VaR / CVaR ----------

def test_historical_var_cvar_cross_checked_against_numpy():
    r = pd.Series([-0.10, -0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.01, 0.02, 0.03])
    alpha = 0.1
    expected_var = -np.quantile(r.values, alpha)
    threshold = np.quantile(r.values, alpha)
    expected_cvar = -r[r <= threshold].mean()

    assert historical_var(r, alpha) == pytest.approx(expected_var)
    assert historical_cvar(r, alpha) == pytest.approx(expected_cvar)


def test_cvar_is_never_smaller_than_var():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.standard_t(df=3, size=2000) * 0.02)  # fat-tailed
    var = historical_var(r, alpha=0.05)
    cvar = historical_cvar(r, alpha=0.05)
    assert cvar >= var


def test_parametric_var_matches_normal_theory():
    rng = np.random.default_rng(1)
    sigma = 0.02
    r = pd.Series(rng.normal(0, sigma, 200_000))
    var = parametric_var(r, alpha=0.05)
    expected = 1.6448536269514722 * sigma  # -z_0.05 * sigma, mean ~ 0
    assert var == pytest.approx(expected, rel=0.02)


# ---------- Drawdown ----------

def test_drawdown_series_known_example():
    r = pd.Series([0.10, -0.20, 0.10, 0.05])
    dd = drawdown_series(r)
    assert dd.iloc[0] == pytest.approx(0.0)
    assert dd.iloc[1] == pytest.approx(-0.20)
    assert dd.iloc[2] == pytest.approx(-0.12)
    assert dd.iloc[3] == pytest.approx(-0.076)
    assert max_drawdown(r) == pytest.approx(0.20)


def test_max_drawdown_all_positive_returns_is_zero():
    r = pd.Series([0.01, 0.02, 0.005, 0.03])
    assert max_drawdown(r) == pytest.approx(0.0)


# ---------- Sharpe / Calmar ----------

def test_sharpe_ratio_zero_vol_is_nan():
    r = pd.Series([0.001] * 50)
    assert np.isnan(sharpe_ratio(r))


def test_sharpe_ratio_higher_mean_gives_higher_sharpe():
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 0.01, 500)
    low_mean = pd.Series(noise + 0.0001)
    high_mean = pd.Series(noise + 0.001)
    assert sharpe_ratio(high_mean) > sharpe_ratio(low_mean)


def test_calmar_ratio_higher_return_same_drawdown_gives_higher_calmar():
    # two series with an identical drawdown shape but different overall drift
    base_dd = np.array([0.10, -0.20, 0.10, 0.05])
    low = pd.Series(base_dd)
    high = pd.Series(base_dd + 0.01)  # uniformly better returns, same relative dip shape
    assert calmar_ratio(high) > calmar_ratio(low)


# ---------- risk_by_regime ----------

def test_risk_by_regime_high_vol_worse_than_low_vol():
    r = _two_regime_returns(n_low=200, n_high=200)
    regime_labels = ["low_vol"] * 200 + ["high_vol"] * 200
    regime_df = pd.DataFrame({"ret": r.values, "regime": regime_labels}, index=r.index)

    table = risk_by_regime(regime_df, alpha=0.05)
    assert set(table.index) == {"low_vol", "high_vol"}
    assert table.loc["low_vol", "n_obs"] == 200
    assert table.loc["high_vol", "n_obs"] == 200
    assert table.loc["high_vol", "var_hist"] > table.loc["low_vol", "var_hist"]
    assert table.loc["high_vol", "cvar_hist"] > table.loc["low_vol", "cvar_hist"]
    assert table.loc["high_vol", "worst_drawdown"] > table.loc["low_vol", "worst_drawdown"]


# ---------- analyze_file (integration with risk.regime) ----------

def test_analyze_file_end_to_end(tmp_path):
    r = _two_regime_returns(n_low=250, n_high=250)
    df = pd.DataFrame({"Datetime": r.index, "ret": r.values, "Ticker": "SIM"})
    p = tmp_path / "SIM_1d.csv"
    df.to_csv(p, index=False)

    overall, risk_table = analyze_file(p, n_regimes=2, roll_window=21)

    assert overall["ticker"] == "SIM"
    assert overall["current_regime"] == "high_vol"
    for key in ("var_hist", "cvar_hist", "max_drawdown", "sharpe", "calmar"):
        assert key in overall
    assert set(risk_table.index) == {"low_vol", "high_vol"}
    assert risk_table.loc["high_vol", "cvar_hist"] > risk_table.loc["low_vol", "cvar_hist"]


def test_analyze_file_too_few_observations_raises(tmp_path):
    r = _two_regime_returns(n_low=10, n_high=10)
    df = pd.DataFrame({"Datetime": r.index, "ret": r.values, "Ticker": "TINY"})
    p = tmp_path / "TINY_1d.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError):
        analyze_file(p)


# ---------- multi-ticker: load_returns_matrix / correlation / portfolio ----------

def test_load_returns_matrix_aligns_and_skips_bad_files(tmp_path):
    idx = pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC")
    good1 = pd.DataFrame({"Datetime": idx, "ret": np.linspace(0, 0.01, 50), "Ticker": "AAA"})
    good2 = pd.DataFrame({"Datetime": idx, "ret": np.linspace(0.01, 0, 50), "Ticker": "BBB"})
    bad = pd.DataFrame({"Datetime": idx, "Close": np.arange(50), "Ticker": "CCC"})  # no 'ret' column

    good1.to_csv(tmp_path / "AAA_1d.csv", index=False)
    good2.to_csv(tmp_path / "BBB_1d.csv", index=False)
    bad.to_csv(tmp_path / "CCC_1d.csv", index=False)

    matrix = load_returns_matrix(tmp_path, pattern="*_1d.csv")
    assert set(matrix.columns) == {"AAA", "BBB"}
    assert len(matrix) == 50


def test_rolling_avg_correlation_perfectly_correlated_and_anticorrelated():
    idx = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    rng = np.random.default_rng(3)
    a = rng.normal(0, 0.01, 100)

    corr_pos = rolling_avg_correlation(pd.DataFrame({"A": a, "B": a}, index=idx), window=20)
    corr_neg = rolling_avg_correlation(pd.DataFrame({"A": a, "B": -a}, index=idx), window=20)

    assert corr_pos.dropna().iloc[-1] == pytest.approx(1.0, abs=1e-9)
    assert corr_neg.dropna().iloc[-1] == pytest.approx(-1.0, abs=1e-9)


def test_rolling_avg_correlation_requires_two_columns():
    idx = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
    df = pd.DataFrame({"A": np.zeros(30)}, index=idx)
    with pytest.raises(ValueError):
        rolling_avg_correlation(df)


def test_portfolio_returns_equal_weight():
    df = pd.DataFrame({"A": [0.02, 0.04], "B": [0.00, 0.02]})
    port = portfolio_returns(df)
    assert port.tolist() == pytest.approx([0.01, 0.03])


def test_portfolio_returns_custom_weights_get_normalized():
    df = pd.DataFrame({"A": [0.10], "B": [0.00]})
    # weights sum to 2 (not 1) -- function should normalize internally
    port = portfolio_returns(df, weights={"A": 1.0, "B": 1.0})
    assert port.iloc[0] == pytest.approx(0.05)

    port_skewed = portfolio_returns(df, weights={"A": 3.0, "B": 1.0})
    assert port_skewed.iloc[0] == pytest.approx(0.075)  # 0.75*0.10 + 0.25*0.00


# ---------- run_for_folder ----------

def test_run_for_folder_end_to_end(tmp_path):
    output_dir = tmp_path / "out"
    for ticker, seed in [("AAA", 1), ("BBB", 2), ("CCC", 3)]:
        r = _two_regime_returns(n_low=150, n_high=150, seed=seed)
        df = pd.DataFrame({"Datetime": r.index, "ret": r.values, "Ticker": ticker})
        df.to_csv(tmp_path / f"{ticker}_1d.csv", index=False)
    # one file that will fail GARCH (too few obs) -- must be skipped, not crash the batch
    bad = pd.Series(np.random.normal(0, 0.01, 5))
    pd.DataFrame({"Datetime": pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC"),
                  "ret": bad.values, "Ticker": "BAD"}).to_csv(tmp_path / "BAD_1d.csv", index=False)

    results = run_for_folder(folder=tmp_path, pattern="*_1d.csv", save_outputs=True, output_dir=output_dir)

    assert "portfolio_summary" in results
    assert results["portfolio_summary"]["n_tickers"] == 4  # matrix load succeeds even for BAD (just short)
    assert "per_ticker_summary" in results
    surviving_tickers = set(results["per_ticker_summary"]["ticker"])
    assert surviving_tickers == {"AAA", "BBB", "CCC"}  # BAD skipped (too few obs for GARCH)

    assert (output_dir / "rolling_corr.csv").exists()
    assert (output_dir / "portfolio_summary.csv").exists()
    assert (output_dir / "risk_overall_summary.csv").exists()
    assert (output_dir / "risk_by_regime" / "AAA_1d_risk_by_regime.csv").exists()


def test_run_for_folder_single_ticker_skips_portfolio_level(tmp_path):
    r = _two_regime_returns(n_low=150, n_high=150)
    df = pd.DataFrame({"Datetime": r.index, "ret": r.values, "Ticker": "ONLY"})
    df.to_csv(tmp_path / "ONLY_1d.csv", index=False)

    results = run_for_folder(folder=tmp_path, pattern="*_1d.csv", save_outputs=False)

    assert "portfolio_summary" not in results
    assert "per_ticker_summary" in results
    assert results["per_ticker_summary"].iloc[0]["ticker"] == "ONLY"