import json
from datetime import date

import pandas as pd
import pytest

from alphaquant.core.settings import MissingSecretError, require_secret
from alphaquant.data.pipeline import run_etl
from alphaquant.data.providers import ProviderRegistry, RawPayload, TiingoProvider
from alphaquant.data.raw_store import RawStore
from alphaquant.risk.volatility import load_returns
from tests.data_fixtures import StaticProvider, coinbase_candles, coinbase_normalizer, tiingo_records

TODAY = date(2024, 7, 1)


def _registry():
    equity = StaticProvider("tiingo", {
        "SPY": tiingo_records(seed=1, ticker="SPY", dividend=("2024-03-15", 1.6)),
        "HYG": tiingo_records(seed=2, ticker="HYG"),
    }, TiingoProvider(token="test"))
    crypto = StaticProvider("coinbase", {
        "BTC-USD": coinbase_candles(date(2023, 1, 3), date(2024, 6, 30)),
    }, coinbase_normalizer())
    return ProviderRegistry(equity=equity, crypto=crypto)


@pytest.fixture
def workspace(tmp_path):
    return {
        "store": RawStore(tmp_path / "data" / "raw"),
        "processed": tmp_path / "data" / "processed",
        "reports": tmp_path / "artifacts" / "data_quality",
    }


def _run(ws, registry, tickers, offline=False):
    return run_etl(tickers=tickers, start=date(2023, 1, 3), end=None, registry=registry,
                   store=ws["store"], processed_dir=ws["processed"], report_dir=ws["reports"],
                   offline=offline, today=TODAY)


def test_end_to_end_publishes_valid_tickers_and_reports_failures(workspace):
    report = _run(workspace, _registry(), ["SPY", "HYG", "BTC-USD", "NOPE"])
    status = {t.ticker: t for t in report.tickers}

    assert status["SPY"].published and status["HYG"].published and status["BTC-USD"].published
    assert status["NOPE"].status == "FAIL" and not status["NOPE"].published
    assert report.overall == "FAIL"

    spy = pd.read_csv(workspace["processed"] / "SPY_1d.csv")
    assert {"date", "ret", "flag_any", "adj_close", "dividend"} <= set(spy.columns)
    assert spy["ret"].notna().all()
    assert not (workspace["processed"] / "NOPE_1d.csv").exists()

    # risk layer reads the processed schema directly
    s = load_returns(workspace["processed"] / "BTC-USD_1d.csv")
    assert s.index.is_monotonic_increasing and len(s) > 500

    manifest = json.loads((workspace["processed"] / "_manifest.json").read_text())
    assert set(manifest["tickers"]) == {"SPY", "HYG", "BTC-USD"}
    assert len(manifest["tickers"]["SPY"]["sha256"]) == 64

    md = (workspace["reports"] / "latest.md").read_text(encoding="utf-8")
    assert "SPY" in md and "NOPE" in md and "FAIL" in md


def test_raw_layer_is_written_and_offline_mode_rebuilds_without_network(workspace):
    _run(workspace, _registry(), ["SPY"])
    raw_dir = workspace["store"].root / "tiingo" / "SPY"
    assert len(list(raw_dir.glob("*.json.gz"))) == 1
    meta = json.loads(next(raw_dir.glob("*.meta.json")).read_text())
    assert meta["n_records"] > 300 and len(meta["sha256"]) == 64

    registry = _registry()
    report = _run(workspace, registry, ["SPY"], offline=True)
    assert registry.equity.fetch_calls == 0
    assert report.tickers[0].published


def test_raw_store_never_overwrites(workspace):
    payload = RawPayload("tiingo", "SPY", {"url": "x"}, [{"a": 1}], fetched_at="2024-01-01T00:00:00Z")
    workspace["store"].write(payload)
    with pytest.raises(FileExistsError):
        workspace["store"].write(payload)


def test_raw_store_detects_tampering(workspace):
    payload = RawPayload("tiingo", "SPY", {"url": "x"}, [{"a": 1}], fetched_at="2024-01-01T00:00:00Z")
    path = workspace["store"].write(payload)
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="Checksum"):
        workspace["store"].latest("tiingo", "SPY")


def test_failed_ticker_keeps_last_good_file(workspace):
    _run(workspace, _registry(), ["HYG"])
    good = (workspace["processed"] / "HYG_1d.csv").read_text()

    broken = _registry()
    broken.equity._records["HYG"] = broken.equity._records["HYG"][:50] + broken.equity._records["HYG"][60:]
    report = _run(workspace, broken, ["HYG"])
    assert report.tickers[0].status == "FAIL"
    assert (workspace["processed"] / "HYG_1d.csv").read_text() == good


def test_missing_secret_has_actionable_message(monkeypatch, tmp_path):
    monkeypatch.setenv("ALPHAQUANT_ROOT", str(tmp_path))
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    with pytest.raises(MissingSecretError, match=r"\.env"):
        require_secret("TIINGO_API_KEY")


def test_secret_read_from_dotenv(monkeypatch, tmp_path):
    monkeypatch.setenv("ALPHAQUANT_ROOT", str(tmp_path))
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    (tmp_path / ".env").write_text("TIINGO_API_KEY=abc123\n", encoding="utf-8")
    assert require_secret("TIINGO_API_KEY") == "abc123"
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
