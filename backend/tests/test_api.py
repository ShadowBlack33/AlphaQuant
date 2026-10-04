import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from alphaquant.api.main import app
from alphaquant.risk.volatility import run_for_folder as run_garch_folder
from alphaquant.risk.regime import run_for_folder as run_regime_folder
from alphaquant.risk.portfolio_analytics import run_for_folder as run_portfolio_folder


def _two_regime_returns(n_low=150, n_high=150, low_sigma=0.004, high_sigma=0.03, seed=7) -> pd.Series:
    rng = np.random.default_rng(seed)
    low = rng.normal(0, low_sigma, n_low)
    high = rng.normal(0, high_sigma, n_high)
    vals = np.concatenate([low, high])
    idx = pd.date_range("2020-01-01", periods=len(vals), freq="D", tz="UTC")
    return pd.Series(vals, index=idx, name="ret")


@pytest.fixture
def pipeline_workspace(tmp_path, monkeypatch):
    """Builds a real (not mocked) data/raw + artifacts/risk tree using the
    actual pipeline functions, and points ALPHAQUANT_ROOT at it -- so the
    API is exercised against genuine output shapes.
    """
    monkeypatch.setenv("ALPHAQUANT_ROOT", str(tmp_path))

    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    for ticker, seed in [("AAA", 1), ("BBB", 2)]:
        r = _two_regime_returns(seed=seed)
        pd.DataFrame({"Datetime": r.index, "ret": r.values, "Ticker": ticker}).to_csv(
            data_dir / f"{ticker}_1d.csv", index=False
        )

    risk_dir = tmp_path / "artifacts" / "risk"
    run_garch_folder(folder=data_dir, pattern="*_1d.csv", save_summary=True,
                     summary_path=risk_dir / "garch_summary.csv",
                     save_series=True, series_dir=risk_dir / "vol_series")
    run_regime_folder(folder=data_dir, pattern="*_1d.csv", save_summary=True,
                      summary_path=risk_dir / "regime_summary.csv",
                      save_labels=True, labels_dir=risk_dir / "regime_labels")
    run_portfolio_folder(folder=data_dir, pattern="*_1d.csv", save_outputs=True, output_dir=risk_dir)
    return tmp_path


def test_health():
    client = TestClient(app)
    assert client.get("/api/health").json() == {"status": "ok"}


def test_endpoints_empty_when_no_pipeline_output_yet(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPHAQUANT_ROOT", str(tmp_path))
    client = TestClient(app)
    assert client.get("/api/tickers").json() == []
    assert client.get("/api/garch-summary").json() == []
    assert client.get("/api/regime-summary").json() == []
    assert client.get("/api/risk-overall").json() == []
    assert client.get("/api/portfolio-summary").json() == {}
    assert client.get("/api/rolling-corr").json() == []


def test_per_ticker_endpoints_404_for_unknown_ticker(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPHAQUANT_ROOT", str(tmp_path))
    client = TestClient(app)
    assert client.get("/api/vol-series/NOPE").status_code == 404
    assert client.get("/api/regime-labels/NOPE").status_code == 404
    assert client.get("/api/risk-by-regime/NOPE").status_code == 404


def test_tickers(pipeline_workspace):
    client = TestClient(app)
    assert client.get("/api/tickers").json() == ["AAA", "BBB"]


def test_invalid_ticker_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPHAQUANT_ROOT", str(tmp_path))
    client = TestClient(app)
    assert client.get("/api/vol-series/..%2F..%2Fetc").status_code in (404, 422)
    assert client.get("/api/regime-labels/bad$ticker").status_code == 422


def test_removed_classifier_endpoints_are_gone():
    client = TestClient(app)
    assert client.get("/api/signals").status_code == 404
    assert client.get("/api/metrics-full").status_code == 404


def test_garch_and_regime_summaries(pipeline_workspace):
    client = TestClient(app)
    garch = client.get("/api/garch-summary").json()
    regime = client.get("/api/regime-summary").json()
    assert {row["ticker"] for row in garch} == {"AAA", "BBB"}
    assert {row["ticker"] for row in regime} == {"AAA", "BBB"}
    assert {row["current_regime"] for row in regime} == {"high_vol"}


def test_risk_overall_and_portfolio_summary(pipeline_workspace):
    client = TestClient(app)
    risk_overall = client.get("/api/risk-overall").json()
    assert {row["ticker"] for row in risk_overall} == {"AAA", "BBB"}
    for row in risk_overall:
        assert row["var_hist"] > 0
        assert row["cvar_hist"] >= row["var_hist"]
    portfolio = client.get("/api/portfolio-summary").json()
    assert portfolio["n_tickers"] == 2
    assert "portfolio_var_hist" in portfolio


def test_rolling_corr(pipeline_workspace):
    client = TestClient(app)
    rows = client.get("/api/rolling-corr").json()
    assert len(rows) > 0
    assert "avg_rolling_corr" in rows[0]


def test_vol_series_and_regime_labels_per_ticker(pipeline_workspace):
    client = TestClient(app)
    vol = client.get("/api/vol-series/AAA").json()
    assert len(vol) > 0
    assert "cond_vol_annualized" in vol[0]
    labels = client.get("/api/regime-labels/AAA").json()
    assert len(labels) > 0
    assert "regime" in labels[0]
    risk_by_regime = client.get("/api/risk-by-regime/AAA").json()
    regimes_present = {row["regime"] for row in risk_by_regime}
    assert regimes_present == {"low_vol", "high_vol"}