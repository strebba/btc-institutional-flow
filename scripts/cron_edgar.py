"""Cron script: refresh incrementale delle note strutturate IBIT da SEC EDGAR.

Cerca i filing 424B2/424B3 degli ultimi N giorni (finestra di lookback con
overlap), li parsa e fa upsert nel DB. L'upsert è idempotente su `filing_url`,
quindi l'overlap fra run successive non crea duplicati.

Progettato per essere chiamato da uno scheduler (cron, DO App Platform,
GitHub Actions) senza argomenti interattivi.

Uso:
    python3 scripts/cron_edgar.py

Variabili d'ambiente:
    EDGAR_USER_AGENT   Obbligatoria di fatto: "app/1.0 (email-reale)". Senza una
                       email valida la SEC può bannare l'IP (cfr. ToS EDGAR).
    EDGAR_LOOKBACK_DAYS  Giorni di finestra all'indietro (default 14).

Scheduling suggerito (crontab, una volta al giorno):
    30 6 * * *  EDGAR_USER_AGENT="ibit-gamma-tracker/1.0 (you@email)" \
                /path/venv/bin/python3 /path/scripts/cron_edgar.py

Exit code:
    0 — refresh completato (anche con 0 nuovi filing)
    1 — ricerca EDGAR fallita (nessun dato scritto)
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import setup_logging
from src.edgar.parser import ProspectusParser
from src.edgar.search import EdgarEftsSearcher
from src.edgar.structured_notes_db import StructuredNotesDB

_log = setup_logging("cron_edgar")

_DEFAULT_LOOKBACK_DAYS = 14


def main() -> None:
    lookback = int(os.getenv("EDGAR_LOOKBACK_DAYS", _DEFAULT_LOOKBACK_DAYS))
    start    = (date.today() - timedelta(days=lookback)).isoformat()

    _log.info("Refresh EDGAR incrementale: finestra da %s (lookback %dg)", start, lookback)

    searcher = EdgarEftsSearcher()
    try:
        filings = searcher.collect_all_filings(start_date=start)
    except Exception as exc:
        _log.error("Ricerca EDGAR fallita: %s", exc)
        sys.exit(1)

    if not filings:
        _log.info("Nessun filing nella finestra: niente da aggiornare.")
        print(f"[OK] 0 filing da {start} — nessun aggiornamento")
        return

    _log.info("Trovati %d filing unici: parsing...", len(filings))
    notes = ProspectusParser().parse_batch(filings)

    db  = StructuredNotesDB()
    ids = db.upsert_notes(notes)
    stats = db.summary()

    print(
        f"[OK] finestra da {start}: {len(filings)} filing → "
        f"{len(notes)} note parsate, {len(ids)} upsert. "
        f"DB: {stats['total_notes']} note / {stats['total_barriers']} barriere"
    )


if __name__ == "__main__":
    main()
