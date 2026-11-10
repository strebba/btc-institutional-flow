#!/usr/bin/env python3
"""Confronta dealer_flow e dealer_flow_v2 sulle cohort maturate dopo la pre-registrazione.

Il verdetto usa solo date >= --start (default 2026-11-10). ``--replay`` aggiunge
in fondo un blocco IN-SAMPLE / INDICATIVE ONLY che non entra nel verdetto.

Non scrive sul DB. Con DATABASE_URL impostata si ferma, salvo --allow-postgres.

    DATABASE_URL= .venv/bin/python scripts/compare_v1_v2.py --db data/runtime.db
    DATABASE_URL= .venv/bin/python scripts/compare_v1_v2.py --db data/runtime.db --replay
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.forecast.shadow_eval import main

if __name__ == "__main__":
    sys.exit(main())
