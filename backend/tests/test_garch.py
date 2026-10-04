import numpy as np
import pandas as pd
import pytest

from alphaquant.risk.volatility import (
    fit_garch,
    conditional_volatility,
    forecast_volatility,
    load_returns,
    analyze_file,
    run_for_folder,
)


def _simulate_garch11(n=600, omega=1e-6, alpha=0.08, beta=0.90, seed=42) -> pd.Series:
    """Simulate a GARCH(1,1) return series so tests exercise real ARCH/GARCH
    dynamics rather than plain white noise."""
    rng = np.random.default_rng(seed)
    eps = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = rng.standard_normal() * np.sqrt(sigma2[t])
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.Series(eps, index=idx, name="ret")


def test_fit_garch_recovers_reasonable_params():
    r = _simulate_garch11()
    res = fit_garch(r, p=1, q=1)
    alpha = res.params["alpha[1]"]
    beta = res.params["beta[1]"]
    assert 0.0 <= alpha <= 1.0
    assert 0.0 <= beta <= 1.0
    # persistence should be well below explosive/non-stationary territory
    assert alpha + beta < 1.0


def test_fit_garch_raises_on_too_few_observations():
    r = pd.Series(np.random.normal(0, 0.01, 10))
    with pytest.raises(ValueError):
        fit_garch(r)


def test_conditional_volatility_same_length_as_input():
    r = _simulate_garch11(n=300)
    res = fit_garch(r)
    cv = conditional_volatility(res)
    assert len(cv) == len(r)
    assert (cv > 0).all()


def test_conditional_volatility_annualized_scales_up():
    r = _simulate_garch11(n=300)
    res = fit_garch(r)
    daily = conditional_volatility(res, annualize=False)
    annual = conditional_volatility(res, annualize=True, periods_per_year=252)
    # annualized vol = daily * sqrt(252), so ratio should be sqrt(252) everywhere
    ratio = (annual / daily).dropna()
    assert np.allclose(ratio, np.sqrt(252), rtol=1e-6)


def test_forecast_volatility_shape_and_positivity():
    r = _simulate_garch11(n=300)
    res = fit_garch(r)
    fc = forecast_volatility(res, horizon=5)
    assert list(fc.index) == ["h1", "h2", "h3", "h4", "h5"]
    assert (fc > 0).all()


def test_load_returns_reads_ret_column(tmp_path):
    df = pd.DataFrame({
        "Datetime": pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"),
        "ret": [0.01, -0.02, 0.005, 0.0, -0.01],
        "Ticker": ["AAPL"] * 5,
    })
    p = tmp_path / "AAPL_1d.csv"
    df.to_csv(p, index=False)
    s = load_returns(p)
    assert len(s) == 5
    assert s.name == "ret"


def test_load_returns_missing_column_raises(tmp_path):
    df = pd.DataFrame({
        "Datetime": pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"),
        "Close": [1, 2, 3, 4, 5],
    })
    p = tmp_path / "AAPL_1d.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_returns(p)


def test_analyze_file_summary_keys(tmp_path):
    r = _simulate_garch11(n=300)
    df = pd.DataFrame({"Datetime": r.index, "ret": r.values, "Ticker": "SIM"})
    p = tmp_path / "SIM_1d.csv"
    df.to_csv(p, index=False)

    summary, cv, fc = analyze_file(p, horizon=5)
    assert summary["ticker"] == "SIM"
    assert summary["n_obs"] == len(r)
    assert "aic" in summary and "bic" in summary
    assert "forecast_vol_h5" in summary
    assert len(cv) == len(r)
    assert len(fc) == 5


def test_run_for_folder_skips_bad_files_and_saves_summary(tmp_path):
    good = _simulate_garch11(n=300, seed=1)
    bad = pd.Series(np.random.normal(0, 0.01, 5))  # too few obs -> should be skipped

    for name, s in [("GOOD_1d.csv", good), ("BAD_1d.csv", bad)]:
        df = pd.DataFrame({"Datetime": pd.date_range("2020-01-01", periods=len(s), freq="D", tz="UTC"),
                            "ret": s.values, "Ticker": name.split("_")[0]})
        df.to_csv(tmp_path / name, index=False)

    out_path = tmp_path / "garch_summary.csv"
    df_summary = run_for_folder(folder=tmp_path, pattern="*_1d.csv", horizon=3,
                                 save_summary=True, summary_path=out_path)

    assert df_summary is not None
    assert len(df_summary) == 1  # only GOOD_1d.csv survives
    assert df_summary.iloc[0]["ticker"] == "GOOD"
    assert out_path.exists()