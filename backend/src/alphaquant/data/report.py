"""Data quality report written on every ETL run (JSON for machines, Markdown
for humans). It is also a citable document for the future RAG assistant."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from alphaquant.data.validation import FLAG_COLS


@dataclass
class TickerReport:
    ticker: str
    provider: str
    status: str
    published: bool
    raw_file: str | None = None
    stats: dict = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class RunReport:
    run_id: str
    started_at: str
    finished_at: str
    mode: str
    tickers: list[TickerReport]

    @property
    def overall(self) -> str:
        statuses = {t.status for t in self.tickers}
        for s in ("FAIL", "WARN"):
            if s in statuses:
                return s
        return "PASS"

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id, "started_at": self.started_at,
            "finished_at": self.finished_at, "mode": self.mode,
            "overall": self.overall,
            "tickers": [t.__dict__ for t in self.tickers],
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Informe de calidad de datos - {self.run_id}",
            "",
            f"Estado global: **{self.overall}** | modo: {self.mode} | "
            f"inicio {self.started_at} | fin {self.finished_at}",
            "",
            "| Ticker | Fuente | Estado | Publicado | Filas | Desde | Hasta | Faltantes | "
            "Flags | Peor dia |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for t in self.tickers:
            st = t.stats or {}
            n_flags = sum((st.get("flags") or {}).values())
            worst = st.get("worst_day") or {}
            worst_txt = f"{worst['ret']:.2%} ({worst['date']})" if worst else "-"
            lines.append(
                f"| {t.ticker} | {t.provider} | {t.status} | {'si' if t.published else 'no'} | "
                f"{st.get('rows', '-')} | {st.get('first_date', '-')} | {st.get('last_date', '-')} | "
                f"{st.get('missing_dates', '-')} | {n_flags} | {worst_txt} |"
            )
        lines += ["", "## Detalle", ""]
        for t in self.tickers:
            lines.append(f"### {t.ticker} - {t.status}")
            for e in t.errors:
                lines.append(f"- ERROR: {e}")
            for m in t.messages:
                lines.append(f"- {m}")
            flags = (t.stats or {}).get("flags") or {}
            active = {k: v for k, v in flags.items() if v}
            if active:
                lines.append("- flags: " + ", ".join(f"{k}={v}" for k, v in active.items()))
            for o in (t.stats or {}).get("top_outliers", []):
                lines.append(f"- outlier {o['date']}: ret={o['ret']:.2%}, z={o['z']}")
            lines.append("")
        lines += ["## Leyenda de flags", ""]
        lines += [f"- `{c}`" for c in FLAG_COLS]
        return "\n".join(lines) + "\n"

    def write(self, out_dir: Path) -> tuple[Path, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"dq_report_{self.run_id}.json"
        md_path = out_dir / f"dq_report_{self.run_id}.md"
        json_path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(self.to_markdown(), encoding="utf-8")
        (out_dir / "latest.json").write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
        (out_dir / "latest.md").write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
        return json_path, md_path
