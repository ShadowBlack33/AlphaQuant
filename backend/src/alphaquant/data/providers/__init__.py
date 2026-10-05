"""Provider registry: which source serves which ticker."""
from __future__ import annotations

from alphaquant.data.calendars import is_crypto
from alphaquant.data.providers.base import DataProvider, ProviderError, RawPayload
from alphaquant.data.providers.coinbase import CoinbaseProvider
from alphaquant.data.providers.tiingo import TiingoProvider

__all__ = ["DataProvider", "ProviderError", "RawPayload", "CoinbaseProvider",
           "TiingoProvider", "ProviderRegistry"]

_FACTORIES = {"tiingo": TiingoProvider, "coinbase": CoinbaseProvider}


class ProviderRegistry:
    """Maps tickers to providers. Crypto (`*-USD`) and everything else can
    use different sources, configured in config.yaml under `data:`."""

    def __init__(self, equity: DataProvider, crypto: DataProvider):
        self.equity = equity
        self.crypto = crypto

    @classmethod
    def from_config(cls, cfg: dict) -> "ProviderRegistry":
        data_cfg = cfg.get("data", {}) or {}
        eq_name = data_cfg.get("equity_provider", "tiingo")
        cr_name = data_cfg.get("crypto_provider", "coinbase")
        for name in (eq_name, cr_name):
            if name not in _FACTORIES:
                raise ValueError(f"Proveedor desconocido: {name}. Opciones: {sorted(_FACTORIES)}")
        return cls(equity=_FACTORIES[eq_name](), crypto=_FACTORIES[cr_name]())

    def for_ticker(self, ticker: str) -> DataProvider:
        return self.crypto if is_crypto(ticker) else self.equity
