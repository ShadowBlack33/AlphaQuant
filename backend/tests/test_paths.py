import pytest

from alphaquant.core import paths
from alphaquant.core.config import load_config


def test_env_var_overrides_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPHAQUANT_ROOT", str(tmp_path))
    assert paths.project_root() == tmp_path.resolve()
    assert paths.raw_dir() == tmp_path.resolve() / "data" / "raw"
    assert paths.risk_dir() == tmp_path.resolve() / "artifacts" / "risk"


def test_env_var_must_be_a_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPHAQUANT_ROOT", str(tmp_path / "does-not-exist"))
    with pytest.raises(paths.ProjectRootNotFoundError):
        paths.project_root()


def test_root_found_from_package_location(monkeypatch, tmp_path):
    """Without the env var, the root is found by walking up from the
    package, independent of the current working directory."""
    monkeypatch.delenv("ALPHAQUANT_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    root = paths.project_root()
    assert (root / "config" / "config.yaml").is_file()


def test_missing_config_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPHAQUANT_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        load_config()


def test_config_matches_tickers_on_disk():
    """Single source of truth: the configured universe must match the raw
    files present (skipped when data/raw has not been downloaded yet)."""
    cfg = load_config()
    raw = paths.raw_dir()
    on_disk = {f.stem.split("_")[0] for f in raw.glob("*_1d.csv")} if raw.exists() else set()
    if not on_disk:
        pytest.skip("data/raw vacio: corre el ETL primero")
    assert set(cfg["tickers"]) == on_disk
