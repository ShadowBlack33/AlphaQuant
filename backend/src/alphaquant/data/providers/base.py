"""Provider interface: every data source fetches a raw payload and
normalizes it to the canonical schema (alphaquant.data.schema)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd


class ProviderError(RuntimeError):
    """A provider request failed in a way retries cannot fix."""


@dataclass
class RawPayload:
    """Exactly what a provider returned, plus how it was requested.

    `request` must never contain secrets: it is persisted next to the data.
    """
    provider: str
    ticker: str
    request: dict[str, Any]
    records: list[Any]
    fetched_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )


class DataProvider(ABC):
    name: str = "base"

    @abstractmethod
    def fetch(self, ticker: str, start: date, end: date | None = None) -> RawPayload:
        """Download daily bars for one ticker."""

    @abstractmethod
    def normalize(self, payload: RawPayload) -> pd.DataFrame:
        """Convert a raw payload to the canonical schema (unvalidated)."""
