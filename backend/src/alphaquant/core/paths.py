"""Single source of truth for filesystem locations.

Every path is resolved from the repository root, never from the current
working directory. In v0.2 all paths were relative ("data/raw",
"models/risk"), so running a script from the wrong folder silently created
duplicate data/, models/ and logs/ trees outside the repo.

Root resolution order:
1. The ALPHAQUANT_ROOT environment variable (Docker, tests).
2. Walk up from this file until a folder containing config/config.yaml
   is found (editable install inside the repo).
If neither works, fail loudly instead of guessing.
"""
from __future__ import annotations

import os
from pathlib import Path

CONFIG_MARKER = Path("config") / "config.yaml"


class ProjectRootNotFoundError(RuntimeError):
    """Raised when the repository root cannot be located."""


def project_root() -> Path:
    env = os.environ.get("ALPHAQUANT_ROOT")
    if env:
        root = Path(env).expanduser().resolve()
        if not root.is_dir():
            raise ProjectRootNotFoundError(f"ALPHAQUANT_ROOT no es un directorio: {root}")
        return root
    for parent in Path(__file__).resolve().parents:
        if (parent / CONFIG_MARKER).is_file():
            return parent
    raise ProjectRootNotFoundError(
        "No se encontro la raiz del proyecto (config/config.yaml). "
        "Instala el paquete en modo editable dentro del repo o define ALPHAQUANT_ROOT."
    )


def config_file() -> Path:
    return project_root() / CONFIG_MARKER


def data_dir() -> Path:
    return project_root() / "data"


def raw_dir() -> Path:
    return data_dir() / "raw"


def processed_dir() -> Path:
    return data_dir() / "processed"


def artifacts_dir() -> Path:
    """Generated outputs (trained models, risk tables, logs). Never committed."""
    return project_root() / "artifacts"


def risk_dir() -> Path:
    return artifacts_dir() / "risk"


def logs_dir() -> Path:
    return artifacts_dir() / "logs"
