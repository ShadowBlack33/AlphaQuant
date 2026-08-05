from __future__ import annotations
import logging
import sys
from pathlib import Path


def setup_logging(logs_dir: str = "logs", level: int = logging.INFO) -> None:
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(logs_dir) / "app.log"
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,  # avoid duplicate handlers if setup_logging is called twice
    )
