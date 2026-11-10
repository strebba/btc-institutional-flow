#!/usr/bin/env python3
"""Riscarica l'OHLC giornaliero completo e sostituisce le barre parziali.

Non è nel cron. Con ``DATABASE_URL`` impostata si ferma, a meno di
``--allow-postgres``: sul Mac il file è lo SQLite locale.

    DATABASE_URL= .venv/bin/python scripts/backfill_prices.py \\
        --dry-run --db data/runtime.db

    DATABASE_URL= .venv/bin/python scripts/backfill_prices.py \\
        --write --db data/runtime.db

``--dry-run`` scarica e stampa le diff (close/high/low vecchi contro quelli
della sessione chiusa) senza scrivere. ``--write`` sovrascrive solo le date
che differiscono. Il giorno UTC corrente non viene persistito come close.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.flows.price_backfill import main

if __name__ == "__main__":
    sys.exit(main())
