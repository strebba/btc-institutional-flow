"""Snapshot giornaliero delle barriere attive per il backtest storico.

Esegue ``StructuredNotesDB.snapshot_active_barriers()`` per salvare lo stato
corrente di tutte le barriere attive nella tabella ``barrier_snapshots``.
Lo storico accumulato consente al pilastro Barrier di usare le barriere
corrette per-data durante il backtest.

Da eseguire giornalmente (cron, GitHub Actions, o scheduler):
    python scripts/cron_snapshot_barriers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import setup_logging
from src.edgar.structured_notes_db import StructuredNotesDB

_log = setup_logging("cron.snapshot_barriers")


def main() -> None:
    db = StructuredNotesDB()
    n = db.snapshot_active_barriers()
    _log.info("Snapshot barriere completato: %d barriere salvate", n)
    if n == 0:
        _log.warning(
            "Nessuna barriera attiva trovata — eseguire scripts/run_edgar.py "
            "per popolare il DB di note strutturate."
        )


if __name__ == "__main__":
    main()
