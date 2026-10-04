"""Project configuration loader.

v0.2 silently fell back to a hard-coded default config (4 tickers) when
config/config.yaml was not found from the current working directory. That
hid wrong-folder runs. Now the file is located through core.paths and a
missing file is an error.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from alphaquant.core import paths

REQUIRED_KEYS = ("tickers", "start_date", "interval", "features", "seed", "risk")


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path is not None else paths.config_file()
    if not p.is_file():
        raise FileNotFoundError(f"No existe el archivo de configuracion: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"Faltan claves en {p.name}: {missing}")
    return cfg
