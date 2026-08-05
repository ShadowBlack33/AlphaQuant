"""Market regime detection (clustering / HMM), built on top of GARCH
conditional volatility.

Pipeline: fit GARCH per ticker (alphaquant.risk.volatility) -> build a
feature matrix combining conditional volatility with rolling return
statistics -> cluster into regimes with KMeans (default, no extra
dependency) or a Gaussian HMM (optional, requires `hmmlearn`).

Cluster ids are arbitrary/unordered by construction, so every clustering
method here is followed by `relabel_by_volatility`, which renames clusters
by their mean conditional volatility (ascending) so "low_vol"/"high_vol"
(or "regime_0".."regime_{k-1}" for k > 3) means the same thing run to run,
regardless of which numeric label KMeans/HMM happened to assign.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from alphaquant.risk.volatility import fit_garch, conditional_volatility, load_returns

try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMMLEARN = True
except Exception:
    HAS_HMMLEARN = False

DEFAULT_FEATURE_COLS = ("cond_vol", "roll_vol", "roll_mean_ret")
MIN_OBS_FOR_REGIMES = 30


def build_regime_features(
    returns: pd.Series,
    cond_vol: pd.Series,
    roll_window: int = 21,
) -> pd.DataFrame:
    """Combine returns, GARCH conditional volatility, and rolling stats
    into a feature matrix for regime clustering. Rows with an incomplete
    rolling window are dropped.
    """
    df = pd.DataFrame({"ret": returns})
    df["cond_vol"] = cond_vol.reindex(df.index)
    df["roll_mean_ret"] = df["ret"].rolling(roll_window, min_periods=roll_window).mean()
    df["roll_vol"] = df["ret"].rolling(roll_window, min_periods=roll_window).std(ddof=0)
    df = df.dropna()
    return df


def _regime_name(rank: int, n_regimes: int) -> str:
    if n_regimes == 2:
        return ["low_vol", "high_vol"][rank]
    if n_regimes == 3:
        return ["low_vol", "mid_vol", "high_vol"][rank]
    return f"regime_{rank}"


def relabel_by_volatility(
    features: pd.DataFrame,
    raw_labels: Sequence[int],
    vol_col: str = "cond_vol",
) -> pd.Series:
    """Rename arbitrary cluster ids to ordered, human-readable regime names
    based on each cluster's mean `vol_col` (ascending)."""
    tmp = pd.DataFrame({"_cluster": raw_labels}, index=features.index)
    tmp[vol_col] = features[vol_col].values
    order = tmp.groupby("_cluster")[vol_col].mean().sort_values().index.tolist()
    n_regimes = len(order)
    mapping = {cluster_id: _regime_name(rank, n_regimes) for rank, cluster_id in enumerate(order)}
    return pd.Series([mapping[c] for c in raw_labels], index=features.index, name="regime")


def fit_kmeans_regimes(
    features: pd.DataFrame,
    feature_cols: Sequence[str] = DEFAULT_FEATURE_COLS,
    n_regimes: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, KMeans, StandardScaler]:
    X = features[list(feature_cols)].to_numpy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = KMeans(n_clusters=n_regimes, random_state=random_state, n_init=10)
    raw_labels = model.fit_predict(Xs)
    return raw_labels, model, scaler


def fit_hmm_regimes(
    features: pd.DataFrame,
    feature_cols: Sequence[str] = DEFAULT_FEATURE_COLS,
    n_regimes: int = 2,
    random_state: int = 42,
    n_iter: int = 200,
):
    """Gaussian HMM regime detection (captures regime persistence via a
    transition matrix, unlike KMeans which treats each row independently).
    Requires `hmmlearn` (`pip install hmmlearn`); raises ImportError with
    that instruction if it isn't installed.
    """
    if not HAS_HMMLEARN:
        raise ImportError(
            "hmmlearn no esta instalado. Instala con `pip install hmmlearn` "
            "para usar method='hmm' en la deteccion de regimenes."
        )
    X = features[list(feature_cols)].to_numpy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = GaussianHMM(
        n_components=n_regimes, covariance_type="diag",
        random_state=random_state, n_iter=n_iter,
    )
    model.fit(Xs)
    raw_labels = model.predict(Xs)
    return raw_labels, model, scaler


def detect_regimes_for_file(
    p: Path,
    p_order: int = 1,
    q_order: int = 1,
    n_regimes: int = 2,
    roll_window: int = 21,
    ret_col: str = "ret",
    method: str = "kmeans",
    random_state: int = 42,
) -> Tuple[dict, pd.DataFrame]:
    """Fit GARCH, build regime features, cluster, and relabel by volatility.

    Returns (summary_dict, regime_df) where regime_df has one row per
    Datetime with columns [ret, cond_vol, roll_mean_ret, roll_vol, regime].
    """
    r = load_returns(p, ret_col=ret_col)
    res = fit_garch(r, p=p_order, q=q_order)
    cv = conditional_volatility(res, annualize=True)
    features = build_regime_features(r, cv, roll_window=roll_window)

    if len(features) < MIN_OBS_FOR_REGIMES:
        raise ValueError(
            f"Se necesitan al menos {MIN_OBS_FOR_REGIMES} observaciones tras el "
            f"rolling window ({roll_window}) para detectar regimenes, hay {len(features)}"
        )

    if method == "kmeans":
        raw_labels, _model, _scaler = fit_kmeans_regimes(features, n_regimes=n_regimes, random_state=random_state)
    elif method == "hmm":
        raw_labels, _model, _scaler = fit_hmm_regimes(features, n_regimes=n_regimes, random_state=random_state)
    else:
        raise ValueError(f"method desconocido: '{method}' (usa 'kmeans' o 'hmm')")

    regimes = relabel_by_volatility(features, raw_labels, vol_col="cond_vol")
    regime_df = features.copy()
    regime_df["regime"] = regimes.values

    summary = {
        "file": p.name,
        "ticker": p.stem.split("_")[0],
        "method": method,
        "n_regimes": n_regimes,
        "n_obs": len(regime_df),
        "current_regime": regime_df["regime"].iloc[-1],
    }
    time_in_regime = regime_df["regime"].value_counts(normalize=True)
    for name, frac in time_in_regime.items():
        summary[f"pct_time_{name}"] = float(frac)

    return summary, regime_df


def run_for_folder(
    folder: str | Path = "data/raw",
    pattern: str = "*_1d.csv",
    p_order: int = 1,
    q_order: int = 1,
    n_regimes: int = 2,
    roll_window: int = 21,
    ret_col: str = "ret",
    method: str = "kmeans",
    random_state: int = 42,
    save_summary: bool = True,
    summary_path: Path = Path("models/risk/regime_summary.csv"),
    save_labels: bool = False,
    labels_dir: Path = Path("models/risk/regime_labels"),
) -> pd.DataFrame | None:
    """Detect regimes per ticker across a folder of `*_1d.csv` files.

    Mirrors the run_for_folder pattern used in
    alphaquant.risk.volatility / alphaquant.training.*: skips files that
    fail (too few observations, bad columns, non-convergence) with a
    warning instead of aborting the whole batch.
    """
    folder = Path(folder)
    rows = []
    for f in sorted(folder.glob(pattern)):
        try:
            summary, regime_df = detect_regimes_for_file(
                f, p_order=p_order, q_order=q_order, n_regimes=n_regimes,
                roll_window=roll_window, ret_col=ret_col, method=method,
                random_state=random_state,
            )
            rows.append(summary)
            if save_labels:
                labels_dir.mkdir(parents=True, exist_ok=True)
                regime_df.to_csv(labels_dir / f"{f.stem}_regimes.csv")
        except Exception as e:
            print(f"[WARN] Deteccion de regimenes fallo en {f.name}: {e}")

    if not rows:
        print("Sin resultados de regimenes.")
        return None

    df = pd.DataFrame(rows)
    if save_summary:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(summary_path, index=False)
    return df