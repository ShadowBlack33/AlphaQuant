from __future__ import annotations
import os
from pathlib import Path


def cleanup_logs(log_dir: Path = Path("logs"), keep: int = 5) -> None:
    if not log_dir.exists():
        return
    logs = sorted(log_dir.glob("*.log"), key=os.path.getmtime, reverse=True)
    if len(logs) <= keep:
        return
    for f in logs[keep:]:
        try:
            f.unlink()
            print(f"Log eliminado: {f.name}")
        except Exception as e:
            print(f"Error eliminando {f.name}: {e}")
