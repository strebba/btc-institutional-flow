#!/usr/bin/env python3
"""Ricostruisce a mano la coorte dealer-flow di un giorno passato.

Non è nel cron e non si collega a Postgres (si ferma se ``DATABASE_URL`` è
impostata). ``--dry-run`` legge gli SQLite in sola lettura e stampa le
predizioni che verrebbero scritte, con la provenienza di ogni input.

Esempio sul Mac, senza toccare il database di produzione:

    DATABASE_URL= .venv/bin/python scripts/reconstruct_cohort.py \\
        --date 2026-11-03 --dry-run \\
        --predictions-db data/runtime.db \\
        --gex-db data/structured_notes.db \\
        --notes-db data/structured_notes.db

``--write`` inserisce solo quella coorte (source ``dealer_flow``,
``created_at`` = slot, ``metadata.reconstructed`` = true). Se la coorte
esiste già, non la modifica. ``--score`` verifica soltanto gli id appena
inseriti.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.forecast.reconstruct import main

if __name__ == "__main__":
    sys.exit(main())
