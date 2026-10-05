"""Secrets and environment settings.

Secrets live in a `.env` file at the repository root (ignored by git) or in
real environment variables, which take precedence. Values are never logged.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

from alphaquant.core import paths


class MissingSecretError(RuntimeError):
    """Raised when a required secret is not configured."""


def load_env() -> None:
    """Load <repo>/.env into os.environ without overriding real env vars."""
    try:
        env_file = paths.project_root() / ".env"
    except paths.ProjectRootNotFoundError:
        return
    if env_file.is_file():
        load_dotenv(env_file, override=False)


def require_secret(name: str) -> str:
    load_env()
    value = os.environ.get(name, "").strip()
    if not value:
        raise MissingSecretError(
            f"Falta {name}. Agregalo al archivo .env en la raiz del repo: {name}=tu_valor"
        )
    return value
