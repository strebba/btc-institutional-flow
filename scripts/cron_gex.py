"""Cron script: salva uno snapshot GEX giornaliero nel DB.

Progettato per essere chiamato da un job scheduler (cron, DO App Platform,
GitHub Actions) senza argomenti interattivi.

Uso:
    python3 scripts/cron_gex.py

Scheduling suggerito (crontab, lunedì-venerdì):
    0 10,14,18,22 * * 1-5  /path/venv/bin/python3 /path/scripts/cron_gex.py

Exit code:
    0 — snapshot salvato con successo
    1 — fetch Deribit fallito (nessun dato scritto)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import setup_logging
from src.gex.deribit_client import DeribitClient
from src.gex.gex_calculator import GexCalculator
from src.gex.gex_db import GexDB
from src.gex.regime_detector import RegimeDetector

_log = setup_logging("cron_gex")


def main() -> None:
    db = GexDB()

    # Pre-popola storico per percentile GEX corretto
    detector = RegimeDetector()
    history  = db.get_latest_n(90)
    detector.load_history_from_db(history)

    _log.info("Fetch opzioni Deribit...")
    try:
        client  = DeribitClient()
        spot    = client.get_spot_price()
        options = client.fetch_all_options("BTC")
    except Exception as exc:
        _log.error("Fetch Deribit fallito: %s", exc)
        sys.exit(1)

    if not options:
        _log.error("Nessuna opzione ricevuta da Deribit")
        sys.exit(1)

    calc     = GexCalculator()
    snapshot = calc.calculate_gex(options, spot)
    state    = detector.detect(snapshot)

    db.insert_snapshot(snapshot, state.regime)

    print(
        f"[OK] spot={snapshot.spot_price:,.0f} "
        f"gex={snapshot.total_net_gex/1e6:+.1f}M "
        f"regime={state.regime} "
        f"total_snapshots={db.count()}"
    )


if __name__ == "__main__":
    main()
