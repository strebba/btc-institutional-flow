"""Cron script: salva uno snapshot GEX giornaliero nel DB.

Progettato per essere chiamato da un job scheduler (cron, DO App Platform,
GitHub Actions) senza argomenti interattivi.

Uso:
    python3 scripts/cron_gex.py               # BTC ed ETH
    python3 scripts/cron_gex.py --asset ETH   # un solo asset

Scheduling suggerito (crontab, lunedì-venerdì):
    0 10,14,18,22 * * 1-5  /path/venv/bin/python3 /path/scripts/cron_gex.py

Exit code:
    0 — snapshot salvati per tutti gli asset richiesti
    1 — fetch Deribit fallito per almeno un asset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.assets import ASSETS
from src.config import setup_logging
from src.gex.deribit_client import DeribitClient
from src.gex.gex_calculator import GexCalculator
from src.gex.gex_db import GexDB
from src.gex.regime_detector import RegimeDetector

_log = setup_logging("cron_gex")


def snapshot_asset(asset: str, client: DeribitClient, db: GexDB) -> bool:
    """Calcola e salva lo snapshot GEX di un asset. Vero se è stato scritto."""
    # Pre-popola storico per percentile GEX corretto
    detector = RegimeDetector(asset=asset)
    detector.load_history_from_db(db.get_latest_n(90, asset=asset))

    _log.info("Fetch opzioni Deribit %s...", asset)
    try:
        spot    = client.get_spot_price(asset)
        options = client.fetch_all_options(ASSETS[asset].deribit_currency)
    except Exception as exc:
        _log.error("Fetch Deribit %s fallito: %s", asset, exc)
        return False

    if not options:
        _log.error("Nessuna opzione %s ricevuta da Deribit", asset)
        return False

    calc     = GexCalculator()
    snapshot = calc.calculate_gex(options, spot)
    state    = detector.detect(snapshot)

    db.insert_snapshot(snapshot, state.regime, asset=asset)

    print(
        f"[OK] {asset} spot={snapshot.spot_price:,.0f} "
        f"gex={snapshot.total_net_gex/1e6:+.1f}M "
        f"regime={state.regime} "
        f"total_snapshots={db.count(asset=asset)}"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot GEX giornaliero")
    parser.add_argument("--asset", choices=[*ASSETS, "all"], default="all")
    args = parser.parse_args()

    assets = list(ASSETS) if args.asset == "all" else [args.asset]
    db = GexDB()
    client = DeribitClient()
    esiti = [snapshot_asset(a, client, db) for a in assets]
    if not all(esiti):
        sys.exit(1)


if __name__ == "__main__":
    main()
