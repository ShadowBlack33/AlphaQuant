"""ETL pipeline: extract -> raw (immutable) -> validate -> processed -> report.

Usage:
    python -m alphaquant.data.pipeline                 # all tickers in config.yaml
    python -m alphaquant.data.pipeline --tickers SPY HYG
    python -m alphaquant.data.pipeline --offline       # rebuild from latest raw, no network

Every run downloads the FULL history (one request per equity, ~15 per
crypto). Adjusted prices are re-based backwards on every new dividend, so
incremental appends would mix old and new adjustment bases.

A ticker that FAILs validation is not written to data/processed (its last
good file, if any, is kept) and the run exits with code 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from alphaquant import __version__
from alphaquant.core import paths
from alphaquant.core.config import load_config
from alphaquant.core.logging import setup_logging
from alphaquant.data.providers import ProviderError, ProviderRegistry
from alphaquant.data.raw_store import RawStore
from alphaquant.data.report import RunReport, TickerReport
from alphaquant.data.transform import to_processed
from alphaquant.data.validation import Status, Thresholds, validate

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_atomic_csv(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    tmp.replace(path)
    return digest


def run_etl(
    tickers: list[str],
    start: date,
    end: date | None,
    registry: ProviderRegistry,
    store: RawStore,
    processed_dir: Path,
    report_dir: Path,
    offline: bool = False,
    today: date | None = None,
    thresholds: Thresholds | None = None,
) -> RunReport:
    today = today or datetime.now(timezone.utc).date()
    started = _now()
    run_id = started.replace(":", "").replace("-", "")
    results: list[TickerReport] = []
    manifest_entries: dict[str, dict] = {}

    for ticker in tickers:
        provider = registry.for_ticker(ticker)
        rep = TickerReport(ticker=ticker, provider=provider.name, status=Status.FAIL.value,
                           published=False)
        try:
            if offline:
                payload = store.latest(provider.name, ticker)
                if payload is None:
                    raise ProviderError("modo offline sin descargas previas en data/raw")
                rep.raw_file = "(ultima descarga existente)"
            else:
                payload = provider.fetch(ticker, start, end)
                rep.raw_file = str(store.write(payload).relative_to(store.root.parent.parent))
            canonical = provider.normalize(payload)
            result = validate(canonical, ticker, start, today, thresholds)
            rep.status = result.status.value
            rep.stats, rep.messages, rep.errors = result.stats, result.messages, result.schema_errors

            if result.status is not Status.FAIL:
                processed = to_processed(result.frame)
                out = processed_dir / f"{ticker}_1d.csv"
                digest = _write_atomic_csv(processed, out)
                rep.published = True
                manifest_entries[ticker] = {
                    "file": out.name, "sha256": digest, "rows": int(len(processed)),
                    "first_date": processed["date"].iloc[0], "last_date": processed["date"].iloc[-1],
                    "source": provider.name, "status": rep.status, "fetched_at": payload.fetched_at,
                }
        except Exception as exc:  # one ticker must never take down the run
            logger.exception("ETL fallo para %s", ticker)
            rep.status = Status.FAIL.value
            rep.errors.append(f"{type(exc).__name__}: {exc}")
        logger.info("%-8s %-8s %s publicado=%s", ticker, provider.name, rep.status, rep.published)
        results.append(rep)

    report = RunReport(run_id=run_id, started_at=started, finished_at=_now(),
                       mode="offline" if offline else "online", tickers=results)
    report.write(report_dir)
    _update_manifest(processed_dir, run_id, manifest_entries)
    return report


def _update_manifest(processed_dir: Path, run_id: str, entries: dict[str, dict]) -> None:
    path = processed_dir / "_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"tickers": {}}
    manifest["tickers"].update(entries)
    manifest.update({"last_run_id": run_id, "code_version": __version__, "updated_at": _now()})
    processed_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AlphaQuant ETL")
    parser.add_argument("--tickers", nargs="+", help="subconjunto de tickers (por defecto: config.yaml)")
    parser.add_argument("--offline", action="store_true", help="reprocesar la ultima descarga, sin red")
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config()
    tickers = [t.upper() for t in (args.tickers or cfg["tickers"])]
    unknown = sorted(set(tickers) - {t.upper() for t in cfg["tickers"]})
    if unknown:
        parser.error(f"tickers fuera de config.yaml: {unknown}")

    end = cfg.get("end_date")
    report = run_etl(
        tickers=tickers,
        start=pd.Timestamp(cfg["start_date"]).date(),
        end=pd.Timestamp(end).date() if end else None,
        registry=ProviderRegistry.from_config(cfg),
        store=RawStore(),
        processed_dir=paths.processed_dir(),
        report_dir=paths.artifacts_dir() / "data_quality",
        offline=args.offline,
    )
    counts = pd.Series([t.status for t in report.tickers]).value_counts().to_dict()
    logger.info("Resultado global: %s %s", report.overall, counts)
    logger.info("Informe: %s", paths.artifacts_dir() / "data_quality" / "latest.md")
    return 1 if report.overall == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
