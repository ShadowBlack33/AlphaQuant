"""Optuna hyperparameter tuning for selected models with walk-forward CV.

Requires the optional `tuning` extra: pip install -e ".[tuning]"
"""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from alphaquant.features.technical import add_technical_features
from alphaquant.models.cv import ExpandingWindowSplit
from alphaquant.utils.config import load_config

DEFAULT_FEATURES_CFG = load_config().get("features", {})


def load_df(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, parse_dates=["Datetime"]).sort_values("Datetime").reset_index(drop=True)


def _make_supervised(df: pd.DataFrame, target: str, horizon: int, lags: int) -> tuple[pd.DataFrame, str]:
    df = df.copy()
    y_col = f"y_{target}_t+{horizon}"
    df[y_col] = df[target].shift(-horizon)
    df = df.dropna().reset_index(drop=True)
    return df, y_col


def objective_svr(trial, df: pd.DataFrame, target: str = "ret", horizon: int = 1,
                   lags: int = 5, features_cfg: dict | None = None) -> float:
    C = trial.suggest_float("C", 0.1, 100.0, log=True)
    eps = trial.suggest_float("epsilon", 1e-4, 0.5, log=True)
    gamma = trial.suggest_categorical("gamma", ["scale", "auto"])

    # FIX (Fase 0): previously called add_technical_features(df) with NO cfg,
    # which raised TypeError since cfg is a required positional argument.
    df_f = add_technical_features(df, features_cfg or DEFAULT_FEATURES_CFG)
    df_f, y_col = _make_supervised(df_f, target=target, horizon=horizon, lags=lags)

    X = df_f.drop(columns=["Datetime", "Ticker", "Interval", y_col], errors="ignore").values
    y = df_f[y_col].values
    n = len(df_f)
    initial_train = max(300, int(0.6 * n))
    splitter = ExpandingWindowSplit(n_splits=3, test_size=int(0.2 * n), initial_train_size=initial_train)

    pipe = Pipeline([("scaler", StandardScaler()), ("model", SVR(C=C, epsilon=eps, gamma=gamma))])
    rmses = []
    for tr, te in splitter.split(n):
        pipe.fit(X[tr], y[tr])
        y_hat = pipe.predict(X[te])
        rmses.append(float(np.sqrt(((y_hat - y[te]) ** 2).mean())))
    return float(np.mean(rmses))


def objective_xgb(trial, df: pd.DataFrame, target: str = "ret", horizon: int = 1,
                   lags: int = 5, features_cfg: dict | None = None) -> float:
    if not HAS_XGB:
        raise optuna.TrialPruned()
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 1000, step=100),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.0, 2.0),
        random_state=42,
        tree_method="hist",
    )
    df_f = add_technical_features(df, features_cfg or DEFAULT_FEATURES_CFG)
    df_f, y_col = _make_supervised(df_f, target=target, horizon=horizon, lags=lags)

    X = df_f.drop(columns=["Datetime", "Ticker", "Interval", y_col], errors="ignore").values
    y = df_f[y_col].values
    n = len(df_f)
    initial_train = max(300, int(0.6 * n))
    splitter = ExpandingWindowSplit(n_splits=3, test_size=int(0.2 * n), initial_train_size=initial_train)

    rmses = []
    for tr, te in splitter.split(n):
        model = XGBRegressor(**params)
        model.fit(X[tr], y[tr])
        y_hat = model.predict(X[te])
        rmses.append(float(np.sqrt(((y_hat - y[te]) ** 2).mean())))
    return float(np.mean(rmses))


def tune_file(csv_path: str | Path, model: str = "svr", n_trials: int = 30,
              target: str = "ret", horizon: int = 1, lags: int = 5,
              features_cfg: dict | None = None) -> Path:
    if not HAS_OPTUNA:
        raise RuntimeError("optuna no esta instalado. Instala el extra: pip install -e '.[tuning]'")
    p = Path(csv_path)
    df = load_df(p)
    study = optuna.create_study(direction="minimize")
    if model == "svr":
        study.optimize(lambda tr: objective_svr(tr, df, target, horizon, lags, features_cfg), n_trials=n_trials)
    elif model == "xgb":
        if not HAS_XGB:
            raise RuntimeError("xgboost not available")
        study.optimize(lambda tr: objective_xgb(tr, df, target, horizon, lags, features_cfg), n_trials=n_trials)
    else:
        raise ValueError("Unknown model: choose 'svr' or 'xgb'")
    out = Path("models") / f"best_params_{model}_{p.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(study.best_trial.params.__repr__())
    logging.info(f"Saved best params to {out}")
    return out
