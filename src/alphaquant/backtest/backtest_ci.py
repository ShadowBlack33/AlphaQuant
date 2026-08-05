"""Deterministic CI backtest sanity check over a fixed sample.

FIX (Fase 3): the accuracy threshold check was commented out, meaning CI
could never fail on model quality regressions. It's now active by default
(configurable via env var for flexibility).
"""
import csv
import os
import sys
from pathlib import Path

SRC = Path("data/samples/prob_summary_sample.csv")
OUT = Path("reports")
OUT.mkdir(parents=True, exist_ok=True)

MIN_ACCURACY = float(os.environ.get("BACKTEST_CI_MIN_ACCURACY", "0.60"))

if not SRC.exists():
    print(f"[ERROR] Missing sample file: {SRC}", file=sys.stderr)
    sys.exit(2)

total = 0
hits = 0
rows = []

with SRC.open(newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        total += 1
        hit = 1 if row["pred"].strip().upper() == row["actual"].strip().upper() else 0
        hits += hit
        rows.append({**row, "hit": hit})

accuracy = hits / total if total else 0.0

rep_path = OUT / "backtest_ci_report.csv"
with rep_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)

print(f"[OK] Samples: {total} | Hits: {hits} | Accuracy: {accuracy:.2%}")

if accuracy < MIN_ACCURACY:
    print(f"[FAIL] Accuracy {accuracy:.2%} under threshold ({MIN_ACCURACY:.0%}).", file=sys.stderr)
    sys.exit(3)

sys.exit(0)
