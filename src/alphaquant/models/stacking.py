
"""Stacking & ensembling utilities for ML models (no leakage)."""
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple

def inverse_rmse_weights(rmses: Dict[str, float], eps: float = 1e-8) -> Dict[str, float]:
    inv = {k: 1.0 / (v + eps) for k, v in rmses.items() if np.isfinite(v) and v >= 0}
    s = sum(inv.values())
    if s == 0:
        # fallback to uniform
        n = len(inv) if inv else 1
        return {k: 1.0/n for k in inv}
    return {k: v/s for k, v in inv.items()}

def weighted_average(preds: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    keys = [k for k in preds.keys() if k in weights]
    if not keys:
        # uniform mean if weights don't match
        arr = np.vstack([preds[k] for k in preds])
        return arr.mean(axis=0)
    arr = np.vstack([preds[k]*weights[k] for k in keys])
    return arr.sum(axis=0)

def simple_average(preds: Dict[str, np.ndarray]) -> np.ndarray:
    arr = np.vstack([v for v in preds.values()])
    return arr.mean(axis=0)
