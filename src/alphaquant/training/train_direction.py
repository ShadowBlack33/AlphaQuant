"""Directional (UP/DOWN) classifier.

NOTE: per project direction discussion, this module is kept as a *secondary,
tactical signal* rather than the core deliverable of the project. The core
deliverable is shifting toward risk analytics (see alphaquant.risk, Fase 4).
"""
from __future__ import annotations
from pathlib import Path
import logging
import argparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from alphaquant.utils.seed import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

ANSI_GREEN = "\033[92m"; ANSI_RED = "\033[91m"; ANSI_BOLD = "\033[1m"; ANSI_RESET = "\033[0m"


def load_df(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p, parse_dates=["Datetime"])
    return df.sort_values("Datetime").reset_index(drop=True)


def make_label(df: pd.DataFrame, target_col: str = "ret", horizon: int = 1) -> pd.Series:
    return (df[target_col].shift(-horizon) > 0).astype(int)


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop_cols = {"Datetime", "Ticker", "Interval"}
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop_cols]


def walk_forward_indices(n: int, initial_train: int, test_size: int):
    start = initial_train
    while start < n:
        tr = np.arange(0, start)
        te_end = min(n, start + test_size)
        te = np.arange(start, te_end)
        if len(te) == 0:
            break
        yield tr, te
        start = te_end


def _fmt_pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{100.0 * float(x):5.1f}%"


def _color_pred(label: str) -> str:
    s = str(label).upper()
    if s == "UP":
        return f"{ANSI_GREEN}{label}{ANSI_RESET}"
    if s == "DOWN":
        return f"{ANSI_RED}{label}{ANSI_RESET}"
    return label


def train_one_file(p: Path, horizon: int = 1, initial_train: int | None = None, test_size: int = 200,
                    save_trace: bool = False, trace_dir: Path | None = None, random_state: int = 42) -> dict:
    df = load_df(p)
    if df.empty or "ret" not in df.columns:
        raise ValueError("DataFrame vacío o sin columna 'ret'")

    y = make_label(df, "ret", horizon=horizon)
    X_cols = feature_columns(df)
    X_df = df[X_cols].copy().replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    if X_df.shape[1] == 0:
        raise ValueError("Todas las columnas de features están vacías (NaN).")
    X = X_df.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=int)
    idx = np.arange(len(df)); n = len(df)

    if initial_train is None:
        initial_train = max(500, int(n * 0.6))

    models = {
        "logreg": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "rf": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=400, n_jobs=-1,
                                            class_weight="balanced_subsample", random_state=random_state)),
        ]),
    }

    last_proba = {}
    last_date = None
    trace_rows = []

    for tr, te in walk_forward_indices(n, initial_train=initial_train, test_size=test_size):
        X_tr, X_te = X[tr], X[te]
        y_tr = yv[tr]
        te_idx = idx[te]
        if len(np.unique(y_tr)) < 2:
            continue

        row = {"Datetime": df.loc[te_idx[-1], "Datetime"], "ticker": p.stem.split("_")[0]}
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_te)[:, 1]
            row[f"proba_{name}"] = float(proba[-1])
            last_proba[name] = float(proba[-1])

        proba_ens = float(np.clip(0.5 * (row.get("proba_logreg", 0.5) + row.get("proba_rf", 0.5)), 0, 1))
        row["proba_ens"] = proba_ens
        row["pred"] = "UP" if proba_ens >= 0.5 else "DOWN"
        row["y_true_next"] = int(yv[te][-1])
        trace_rows.append(row)
        last_date = row["Datetime"]

    if save_trace and trace_rows:
        outdir = (trace_dir or Path("models/traces"))
        outdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(trace_rows).to_csv(outdir / f"{p.stem}_trace.csv", index=False)

    proba_logreg = last_proba.get("logreg", 0.5)
    proba_rf = last_proba.get("rf", 0.5)
    proba_ens = float(np.clip(0.5 * (proba_logreg + proba_rf), 0, 1))
    pred_label = "UP" if proba_ens >= 0.5 else "DOWN"
    confidence = abs(proba_ens - 0.5)

    return {
        "file": p.name, "ticker": p.stem.split("_")[0], "last_date": last_date,
        "proba_logreg": proba_logreg, "proba_rf": proba_rf, "proba_ens": proba_ens,
        "pred": pred_label, "confidence": confidence,
    }


def _print_table(df: pd.DataFrame, color_pred: bool = True):
    df_print = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_print["last_date"]):
        df_print["last_date"] = pd.to_datetime(df_print["last_date"], errors="coerce")
    df_print["FECHA"] = df_print["last_date"].dt.date.astype(str)
    df_print["PROBA_UP (LOGREG)"] = df_print["proba_logreg"].map(_fmt_pct)
    df_print["PROBA_UP (RF)"] = df_print["proba_rf"].map(_fmt_pct)
    df_print["PROBA_UP (ENS)"] = df_print["proba_ens"].map(_fmt_pct)
    df_print["pred_col"] = df_print["pred"].map(_color_pred) if color_pred else df_print["pred"]
    df_print = df_print[["ticker", "FECHA", "PROBA_UP (LOGREG)", "PROBA_UP (RF)", "PROBA_UP (ENS)", "pred_col"]]
    df_print = df_print.rename(columns={"pred_col": "pred"})
    rows = [list(df_print.columns)] + df_print.astype(str).values.tolist()
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for i, r in enumerate(rows):
        print("  ".join(r[j].ljust(widths[j]) for j in range(len(r))))


def run_folder(folder: Path = Path("data/raw"), pattern: str = "*_1d.csv", horizon: int = 1,
               initial_train: int | None = None, test_size: int = 200, top_n: int | None = 10,
               save_summary: bool = False, summary_path: Path = Path("models/prob_summary.csv"),
               print_summary: bool = True, save_trace: bool = False, trace_dir: Path | None = None,
               seed: int = 42) -> Path | None:
    set_global_seed(seed)  # FIX (Fase 1): seed wired through
    folder = Path(folder)
    summaries = []
    for p in sorted(folder.glob(pattern)):
        try:
            summaries.append(train_one_file(p, horizon=horizon, initial_train=initial_train, test_size=test_size,
                                             save_trace=save_trace, trace_dir=trace_dir, random_state=seed))
        except Exception as e:
            print(f"[train_direction] Error con {p.name}: {e}")

    if not summaries:
        print("Sin resultados para clasificador.")
        return None

    df = pd.DataFrame(summaries)
    if "confidence" in df.columns:
        df = df.sort_values("confidence", ascending=False).reset_index(drop=True)

    if print_summary:
        top_up = df.sort_values("proba_ens", ascending=False).head(top_n) if top_n else df
        candidatos_down = df.sort_values("proba_ens", ascending=True)
        top_down = candidatos_down[~candidatos_down["ticker"].isin(top_up["ticker"])].head(top_n) if top_n else candidatos_down

        print(f"\n{ANSI_BOLD}=== TOP-{top_n} ALCISTAS por PROBA_UP (ENS) ==={ANSI_RESET}")
        _print_table(top_up)
        print(f"\n{ANSI_BOLD}=== TOP-{top_n} BAJISTAS por PROBA_UP (ENS) ==={ANSI_RESET}")
        _print_table(top_down)
        print(f"\n{ANSI_BOLD}=== PROBABILIDADES (sube) — ordenadas por confianza ==={ANSI_RESET}")
        df_show = df.head(top_n) if (top_n is not None) else df
        _print_table(df_show)

    if save_summary:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        df.drop(columns=["confidence"]).to_csv(summary_path, index=False)
        print(f"\nResumen guardado en: {summary_path}")

    return summary_path if save_summary else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=str, default="data/raw")
    ap.add_argument("--pattern", type=str, default="*_1d.csv")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--initial-train", type=int, default=None)
    ap.add_argument("--test-size", type=int, default=200)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--save-summary", action="store_true")
    ap.add_argument("--summary-path", type=str, default="models/prob_summary.csv")
    ap.add_argument("--save-trace", action="store_true")
    ap.add_argument("--trace-dir", type=str, default="models/traces")
    args = ap.parse_args()
    run_folder(
        folder=Path(args.folder), pattern=args.pattern, horizon=args.horizon,
        initial_train=args.initial_train, test_size=args.test_size,
        top_n=args.top_n, save_summary=args.save_summary, summary_path=Path(args.summary_path),
        print_summary=True, save_trace=bool(args.save_trace), trace_dir=Path(args.trace_dir),
    )
