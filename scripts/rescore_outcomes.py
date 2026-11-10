#!/usr/bin/env python3
"""Ricalcola gli esiti già verificati sulle barre giornaliere complete.

Non modifica predictions, weights, e non crea predizioni. Legge i prezzi
già nel DB: lanciare prima ``backfill_prices.py --write``.

Si ferma se ``DATABASE_URL`` punta a Postgres, salvo ``--allow-postgres``.

    DATABASE_URL= .venv/bin/python scripts/rescore_outcomes.py \\
        --dry-run --db data/runtime.db

    DATABASE_URL= .venv/bin/python scripts/rescore_outcomes.py \\
        --write --db data/runtime.db

``--write`` copia la riga corrente in ``outcomes_archive`` (valori + timestamp
+ reason) e poi aggiorna ``outcomes``. Ripeterlo non ri-archivia se i numeri
non cambiano.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.forecast.rescore import main

if __name__ == "__main__":
    sys.exit(main())
