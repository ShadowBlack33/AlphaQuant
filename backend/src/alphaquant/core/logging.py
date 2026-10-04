from __future__ import annotations

import logging
import sys
from pathlib import Path

from alphaquant.core import paths


def setup_logging(logs_dir: str | Path | None = None, level: int = logging.INFO) -> Path:
    log_dir = Path(logs_dir) if logs_dir is not None else paths.logs_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "app.log"
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return log_path
