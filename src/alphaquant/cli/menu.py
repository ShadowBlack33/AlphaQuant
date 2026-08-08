from __future__ import annotations
import sys
import logging
from pathlib import Path

from alphaquant.utils.config import load_config
from alphaquant.utils.logging_cfg import setup_logging
from alphaquant.utils.log_cleanup import cleanup_logs
from alphaquant.utils.seed import set_global_seed, get_seed
from alphaquant.etl.extract import fetch_tickers
from alphaquant.etl.transform import transform_frame
from alphaquant.etl.load import save_csv_idempotent
from alphaquant.training.train_regression import run_for_folder as run_regression_folder
from alphaquant.training.train_direction import run_folder as run_classif_folder
from alphaquant.risk.volatility import run_for_folder as run_garch_folder
from alphaquant.risk.regime import run_for_folder as run_regime_folder
from alphaquant.risk.portfolio_analytics import run_for_folder as run_portfolio_folder

VALID_INTERVALS = {
    "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "4h",
    "1d", "5d", "1wk", "1mo", "3mo",
}

PRESETS = {
    "Principales (SPY, QQQ, indices y mega-caps)": [
        "SPY", "QQQ", "DIA", "IWM", "AAPL", "MSFT", "TSLA", "NVDA",
        "AMZN", "GOOGL", "META", "GLD", "TLT", "UUP", "USO",
    ],
    "Mega-cap Tech": ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL", "META"],
    "ETFs Macro": ["SPY", "QQQ", "DIA", "IWM", "GLD", "TLT", "UUP", "USO", "HYG"],
    "Cripto (YF)": ["BTC-USD", "ETH-USD"],
}


def _input(prompt: str, default: str | None = None) -> str:
    s = input(f"{prompt}{' [' + default + ']' if default else ''}: ").strip()
    return s if s else (default or "")


def _yesno(prompt: str, default: bool = False) -> bool:
    d = "s" if default else "n"
    s = input(f"{prompt} (s/n) [{d}]: ").strip().lower()
    if not s:
        s = d
    return s == "s"


def _all_tickers_from_presets() -> list[str]:
    seen = set()
    out: list[str] = []
    for lst in PRESETS.values():
        for t in lst:
            u = t.upper().strip()
            if u and u not in seen:
                seen.add(u)
                out.append(u)
    return out


def run_risk_analytics(
    data_dir: Path,
    interval: str,
    risk_cfg: dict,
    seed: int,
    logger: logging.Logger,
    output_dir: Path = Path("models/risk"),
) -> None:
    """GARCH volatility + regime detection + portfolio risk (VaR/CVaR,
    rolling correlation, drawdown by regime) over every ticker CSV in
    data_dir. Kept as a standalone function (rather than inline in
    run_everything_once) so it can be unit-tested against synthetic data
    without needing yfinance or the interactive input() prompts.

    Each stage is wrapped independently: a failure at the batch level
    (e.g. a missing optional dependency) is logged and printed, but
    doesn't take down a pipeline run that already produced ETL + training
    output. Note that portfolio_analytics.run_for_folder re-fits GARCH +
    regimes per ticker internally (via risk.regime.detect_regimes_for_file)
    rather than reusing the summaries computed above -- some duplicated
    computation, traded for each module staying independently runnable.
    """
    pattern = f"*_{interval}.csv"

    try:
        garch_summary = run_garch_folder(
            folder=str(data_dir), pattern=pattern,
            p_order=int(risk_cfg.get("garch_p", 1)),
            q_order=int(risk_cfg.get("garch_q", 1)),
            horizon=int(risk_cfg.get("garch_horizon", 10)),
            save_summary=True, summary_path=output_dir / "garch_summary.csv",
            save_series=True, series_dir=output_dir / "vol_series",
        )
        if garch_summary is not None:
            print(f"  OK GARCH: {output_dir / 'garch_summary.csv'}")
    except Exception as e:
        logger.exception("GARCH fallo")
        print(f"  ERROR GARCH: {e}")

    try:
        regime_summary = run_regime_folder(
            folder=str(data_dir), pattern=pattern,
            n_regimes=int(risk_cfg.get("n_regimes", 2)),
            roll_window=int(risk_cfg.get("roll_window", 21)),
            method=str(risk_cfg.get("regime_method", "kmeans")),
            random_state=seed,
            save_summary=True, summary_path=output_dir / "regime_summary.csv",
            save_labels=True, labels_dir=output_dir / "regime_labels",
        )
        if regime_summary is not None:
            print(f"  OK Regimenes: {output_dir / 'regime_summary.csv'}")
    except Exception as e:
        logger.exception("Deteccion de regimenes fallo")
        print(f"  ERROR Regimenes: {e}")

    try:
        run_portfolio_folder(
            folder=str(data_dir), pattern=pattern,
            n_regimes=int(risk_cfg.get("n_regimes", 2)),
            roll_window=int(risk_cfg.get("roll_window", 21)),
            method=str(risk_cfg.get("regime_method", "kmeans")),
            random_state=seed,
            save_outputs=True, output_dir=output_dir,
        )
        print(f"  OK Riesgo de portafolio: {output_dir / 'risk_overall_summary.csv'}")
    except Exception as e:
        logger.exception("Analisis de riesgo de portafolio fallo")
        print(f"  ERROR Riesgo de portafolio: {e}")


def run_everything_once(cfg, logger):
    print("\n======== ALPHAQUANT - EJECUCION AUTOMATICA ========")

    start = _input("Fecha inicio YYYY-MM-DD", cfg.get("start_date", "2015-01-01"))
    end = _input("Fecha fin YYYY-MM-DD (Enter=hoy)", cfg.get("end_date", "")) or None
    save_csv = _yesno("Guardar CSV resumen de probabilidades?", default=True)

    interval = (cfg.get("interval", "1d") or "1d").lower()
    if interval not in VALID_INTERVALS:
        interval = "1d"

    top_n = int(cfg.get("default_top_n", 10))
    data_dir = Path(cfg.get("data_dir", "data/raw"))
    data_dir.mkdir(parents=True, exist_ok=True)
    features = cfg.get("features", {})
    seed = get_seed(cfg)  # FIX (Fase 1): seed now read from config end-to-end

    tickers = _all_tickers_from_presets()
    print(f"\nMercados a procesar ({len(tickers)}): {', '.join(tickers)}")

    failed, ok = [], 0
    for t in tickers:
        try:
            logger.info(f"[ETL] {t} {start}->{end} @ {interval}")
            df_raw = fetch_tickers([t], start=start, end=end, interval=interval)
            if df_raw is None or len(df_raw) == 0:
                print(f"  Sin datos {t}")
                failed.append(t)
                continue
            df_t = df_raw if "Ticker" not in df_raw.columns else df_raw[df_raw["Ticker"] == t].copy()
            df_tf = transform_frame(df_t, features_cfg=features, ticker=t)
            if "Interval" not in df_tf.columns:
                df_tf["Interval"] = interval
            out_path = data_dir / f"{t}_{interval}.csv"
            save_csv_idempotent(df_tf, out_path, dedupe_keys=["Datetime", "Ticker"])
            print(f"  OK {out_path}")
            ok += 1
        except Exception as e:
            logger.exception(f"ETL fallo {t}")
            print(f"  ERROR {t}: {e}")
            failed.append(t)

    if failed:
        print(f"\nReintentando tickers fallidos ({len(failed)}): {', '.join(failed)}")
        still = []
        for t in failed:
            try:
                df_raw = fetch_tickers([t], start=None, end=None, interval=interval)
                if df_raw is None or len(df_raw) == 0:
                    print(f"  Sin datos tras reintento {t}")
                    still.append(t)
                    continue
                df_t = df_raw if "Ticker" not in df_raw.columns else df_raw[df_raw["Ticker"] == t].copy()
                df_tf = transform_frame(df_t, features_cfg=features, ticker=t)
                if "Interval" not in df_tf.columns:
                    df_tf["Interval"] = interval
                out_path = data_dir / f"{t}_{interval}.csv"
                save_csv_idempotent(df_tf, out_path, dedupe_keys=["Datetime", "Ticker"])
                print(f"  OK {out_path} (reintento)")
                ok += 1
            except Exception as e:
                logger.exception(f"ETL reintento fallo {t}")
                print(f"  ERROR {t} reintento: {e}")
                still.append(t)
        if still:
            print(f"  Tickers sin datos tras reintentos: {', '.join(still)}")

    if ok == 0:
        print("No hubo CSVs transformados. Abortando.")
        return

    print("\n-> Entrenando regresion...")
    run_regression_folder(folder=str(data_dir), target="ret", horizon=1, embargo=5, save_preds=True, seed=seed)
    print("  OK Metricas regresion: models/metrics_full.csv")

    print("\n-> Entrenando clasificacion (Top-N + CSV opcional + trazas)...")
    run_classif_folder(
        folder=data_dir, pattern=f"*_{interval}.csv", horizon=1, initial_train=None,
        test_size=200, top_n=top_n, save_summary=save_csv,
        summary_path=Path("models/prob_summary.csv"), print_summary=True,
        save_trace=True, trace_dir=Path("models/traces"), seed=seed,
    )
    if save_csv:
        print("  OK Resumen: models/prob_summary.csv")

    risk_cfg = cfg.get("risk", {}) or {}
    run_risk = _yesno("Ejecutar analisis de riesgo (GARCH + regimenes)?", default=bool(risk_cfg.get("enabled", True)))
    if run_risk:
        print("\n-> Analizando riesgo (GARCH + regimenes)...")
        run_risk_analytics(data_dir=data_dir, interval=interval, risk_cfg=risk_cfg, seed=seed, logger=logger)


def main():
    setup_logging()
    logger = logging.getLogger("alphaquant")
    cleanup_logs(keep=5)

    try:
        cfg = load_config("config/config.yaml")
    except Exception as e:
        print(f"No pude cargar config/config.yaml: {e}")
        return

    set_global_seed(get_seed(cfg))
    run_everything_once(cfg, logger)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrumpido")
        sys.exit(1)