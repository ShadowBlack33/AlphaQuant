"""Evaluation metrics for regression on price/returns."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rmse(y_true, y_pred) -> float:
    # FIX (Fase 0): `squared=False` in mean_squared_error was removed in
    # scikit-learn >= 1.6. Compute RMSE manually so this never breaks on upgrade.
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)
    try:
        acc = (
            np.sign(np.diff(np.r_[y_true[0], y_true]))
            == np.sign(np.diff(np.r_[y_pred[0], y_pred]))
        ).mean()
    except Exception:
        acc = np.nan
    return {"mae": float(mae), "rmse": rmse_val, "directional_acc": float(acc)}
