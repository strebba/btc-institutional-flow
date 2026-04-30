"""Cron script: aggiorna l'Institutional Flow Index nel DB.

Uso:
    python3 scripts/cron_ifi.py              # aggiorna il giorno corrente
    python3 scripts/cron_ifi.py --backfill   # backfill tutta la storia disponibile
    python3 scripts/cron_ifi.py --days 7     # aggiorna ultimi N giorni

Scheduling suggerito (crontab, ogni giorno alle 22:00 UTC):
    0 22 * * *  /path/venv/bin/python3 /path/scripts/cron_ifi.py

Exit code:
    0 — IFI aggiornato con successo
    1 — fetch dati fallito
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analytics.ifi_updater import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggiorna Institutional Flow Index DB")
    parser.add_argument("--backfill", action="store_true", help="Ricalcola tutta la storia disponibile")
    parser.add_argument("--days", type=int, default=1, help="Giorni da aggiornare (default 1)")
    args = parser.parse_args()

    sys.exit(run(backfill=args.backfill, days=args.days))
