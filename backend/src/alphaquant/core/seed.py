"""Reproducibility: set global seeds for every library in use."""
from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # torch is optional until the deep learning backbone lands
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_seed(cfg: dict | None, default: int = 42) -> int:
    if not cfg:
        return default
    return int(cfg.get("seed", default))
