"""Immutable raw layer.

Each download is stored exactly as the provider returned it:

    data/raw/<provider>/<ticker>/<fetched_at>.json.gz   (payload)
    data/raw/<provider>/<ticker>/<fetched_at>.meta.json (request, sha256, counts)

Files are created with exclusive mode, so an existing download can never be
overwritten. Anything downstream can be rebuilt from here.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from alphaquant.core import paths
from alphaquant.data.providers.base import RawPayload

SCHEMA_VERSION = 1


class RawStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else paths.raw_dir()

    def _dir(self, provider: str, ticker: str) -> Path:
        return self.root / provider / ticker.upper()

    def write(self, payload: RawPayload) -> Path:
        folder = self._dir(payload.provider, payload.ticker)
        folder.mkdir(parents=True, exist_ok=True)
        stamp = payload.fetched_at.replace(":", "").replace("-", "")
        data_path = folder / f"{stamp}.json.gz"
        meta_path = folder / f"{stamp}.meta.json"

        blob = gzip.compress(json.dumps(payload.records, separators=(",", ":")).encode("utf-8"),
                             mtime=0)
        with open(data_path, "xb") as f:  # "x": fail if it already exists
            f.write(blob)
        meta = {
            "schema_version": SCHEMA_VERSION,
            "provider": payload.provider,
            "ticker": payload.ticker,
            "fetched_at": payload.fetched_at,
            "request": payload.request,
            "n_records": len(payload.records),
            "sha256": hashlib.sha256(blob).hexdigest(),
            "file": data_path.name,
        }
        with open(meta_path, "x", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return data_path

    def latest(self, provider: str, ticker: str) -> RawPayload | None:
        folder = self._dir(provider, ticker)
        metas = sorted(folder.glob("*.meta.json")) if folder.exists() else []
        if not metas:
            return None
        meta = json.loads(metas[-1].read_text(encoding="utf-8"))
        blob = (folder / meta["file"]).read_bytes()
        if hashlib.sha256(blob).hexdigest() != meta["sha256"]:
            raise ValueError(f"Checksum invalido en {folder / meta['file']}")
        records = json.loads(gzip.decompress(blob))
        payload = RawPayload(provider=meta["provider"], ticker=meta["ticker"],
                             request=meta["request"], records=records,
                             fetched_at=meta["fetched_at"])
        return payload

    @staticmethod
    def describe(payload: RawPayload) -> dict:
        d = asdict(payload)
        d.pop("records")
        return d
