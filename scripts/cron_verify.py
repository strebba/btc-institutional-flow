"""Cron script: verifica le previsioni mature e ne salva gli esiti.

Recupera i prezzi reali da PriceFetcher e usa il verifier per assegnare gli Outcome
alle previsioni il cui orizzonte è scaduto. Idempotente: una previsione già verificata
non viene rivalutata (UNIQUE su outcomes.prediction_id).

Uso:
    python3 scripts/cron_verify.py

Scheduling suggerito (una volta al giorno, dopo cron_predict):
    30 7 * * * /path/venv/bin/python3 /path/scripts/cron_verify.py

Exit code: 0 sempre (le mancanze dati lasciano la previsione open).
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import setup_logging

_log = setup_logging("cron_verify")

_TICKER = {"BTC": "BTC-USD"}


def main() -> int:
    import pandas as pd  # noqa: F401  (usato indirettamente dal verifier)

    from src.flows.price_fetcher import PriceFetcher
    from src.forecast.prediction_db import PredictionDB
    from src.forecast.verifier import score_due_predictions

    db = PredictionDB()
    fetcher = PriceFetcher()

    def price_provider(asset, start, end):
        ticker = _TICKER.get(asset, asset)
        # buffer di 1 giorno sul fondo per includere il giorno di maturazione
        return fetcher.fetch(
            ticker,
            start_date=start.date(),
            end_date=(end + timedelta(days=1)).date(),
        )

    due = db.get_due()
    outcomes = score_due_predictions(db, price_provider, datetime.utcnow())

    hits = sum(1 for o in outcomes if o.hit)
    print(
        f"[OK] mature={len(due)} verificate={len(outcomes)} hit={hits} "
        f"miss={len(outcomes) - hits}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
