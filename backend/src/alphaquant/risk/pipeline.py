"""Risk analytics pipeline: GARCH + regimes + portfolio risk.

Replaces the risk section of the v0.2 interactive CLI (cli/menu.py).
Reads every `*_{interval}.csv` in data/processed/ and writes to artifacts/risk/.

Run: python -m alphaquant.risk.pipeline
"""
from __future__ import annotations

import logging
from pathlib import Path

from alphaquant.core import paths
from alphaquant.core.config import load_config
from alphaquant.core.logging import setup_logging
from alphaquant.core.seed import set_global_seed
from alphaquant.risk.portfolio_analytics import run_for_folder as run_portfolio_folder
from alphaquant.risk.regime import run_for_folder as run_regime_folder
from alphaquant.risk.volatility import run_for_folder as run_garch_folder

logger = logging.getLogger(__name__)


def run_risk_analytics(
    data_dir: Path,
    output_dir: Path,
    interval: str,
    risk_cfg: dict,
    seed: int,
) -> dict[str, bool]:
    """Run the three risk stages independently.

    A failure in one stage is logged and reported, but does not stop the
    others. Returns {stage: succeeded}.
    """
    pattern = f"*_{interval}.csv"
    n_regimes = int(risk_cfg.get("n_regimes", 2))
    roll_window = int(risk_cfg.get("roll_window", 21))
    method = str(risk_cfg.get("regime_method", "kmeans"))
    status: dict[str, bool] = {}

    try:
        run_garch_folder(
            folder=data_dir, pattern=pattern,
            p_order=int(risk_cfg.get("garch_p", 1)),
            q_order=int(risk_cfg.get("garch_q", 1)),
            horizon=int(risk_cfg.get("garch_horizon", 10)),
            save_summary=True, summary_path=output_dir / "garch_summary.csv",
            save_series=True, series_dir=output_dir / "vol_series",
        )
        status["garch"] = True
    except Exception:
        logger.exception("GARCH fallo")
        status["garch"] = False

    try:
        run_regime_folder(
            folder=data_dir, pattern=pattern,
            n_regimes=n_regimes, roll_window=roll_window, method=method,
            random_state=seed,
            save_summary=True, summary_path=output_dir / "regime_summary.csv",
            save_labels=True, labels_dir=output_dir / "regime_labels",
        )
        status["regimes"] = True
    except Exception:
        logger.exception("Deteccion de regimenes fallo")
        status["regimes"] = False

    try:
        run_portfolio_folder(
            folder=data_dir, pattern=pattern,
            n_regimes=n_regimes, roll_window=roll_window, method=method,
            random_state=seed,
            save_outputs=True, output_dir=output_dir,
        )
        status["portfolio"] = True
    except Exception:
        logger.exception("Analisis de riesgo de portafolio fallo")
        status["portfolio"] = False

    return status


def main() -> int:
    setup_logging()
    cfg = load_config()
    set_global_seed(int(cfg["seed"]))
    status = run_risk_analytics(
        data_dir=paths.processed_dir(),
        output_dir=paths.risk_dir(),
        interval=str(cfg.get("interval", "1d")),
        risk_cfg=cfg["risk"],
        seed=int(cfg["seed"]),
    )
    for stage, ok in status.items():
        logger.info("%s: %s", stage, "OK" if ok else "FALLO")
    return 0 if all(status.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
