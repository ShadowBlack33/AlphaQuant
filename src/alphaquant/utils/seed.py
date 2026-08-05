"""Reproducibility utilities: set global seeds across libraries used in the project.

FIX (Fase 1): previously `set_global_seed` existed but was never called anywhere,
and every model had `random_state=42` hardcoded regardless of `config.yaml`.
Now `get_seed()` reads the seed from config and every training entrypoint calls
`set_global_seed(get_seed(cfg))` once at startup.
"""
from __future__ import annotations
import random
import numpy as np


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_seed(cfg: dict | None, default: int = 42) -> int:
    if not cfg:
        return default
    return int(cfg.get("seed", default))
