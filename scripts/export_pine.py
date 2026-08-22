"""Esporta l'indicatore TradingView (Pine v6) con i livelli GEX correnti.

Pine Script non può interrogare API esterne: i livelli vengono incorporati nel
sorgente, quindi lo script va rigenerato quando i dati cambiano.

Uso:
    python3 scripts/export_pine.py                      # fetch live da Deribit
    python3 scripts/export_pine.py --from-db            # ultimo snapshot dal DB (no rete)
    python3 scripts/export_pine.py -o /tmp/gex.pine --history-days 0
    make export-pine

Exit code:
    0 — file scritto
    1 — nessun dato disponibile (fetch fallito o DB vuoto)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import setup_logging
from src.gex.gex_db import GexDB
from src.gex.pine_exporter import MAX_PROFILE_STRIKES, build_pine_script
from src.gex.regime_detector import RegimeDetector

_log = setup_logging("export_pine")

_DEFAULT_OUT = Path(__file__).resolve().parent.parent / "exports" / "btc_gex_tradingview.pine"


def _snapshot_live():
    """Snapshot GEX fresco da Deribit (None se il fetch fallisce)."""
    from src.gex.deribit_client import DeribitClient
    from src.gex.gex_calculator import GexCalculator

    client = DeribitClient()
    spot = client.get_spot_price()
    options = client.fetch_all_options("BTC")
    if not options:
        return None
    return GexCalculator().calculate_gex(options, spot)


def _snapshot_from_db(db: GexDB):
    """Ultimo snapshot persistito (senza profilo per strike, non è nel DB)."""
    rows = db.get_latest_n(1)
    return rows[-1] if rows else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Export indicatore GEX per TradingView")
    parser.add_argument("-o", "--output", type=Path, default=_DEFAULT_OUT, help="File .pine di destinazione")
    parser.add_argument("--from-db", action="store_true", help="Usa l'ultimo snapshot del DB invece di Deribit")
    parser.add_argument("--history-days", type=int, default=120, help="Giorni di storico livelli (0 = nessuno)")
    parser.add_argument("--max-strikes", type=int, default=MAX_PROFILE_STRIKES, help="Strike nel profilo call/put")
    parser.add_argument("--range-pct", type=float, default=0.15, help="Finestra strike attorno allo spot")
    parser.add_argument("--stdout", action="store_true", help="Stampa lo script invece di scriverlo su file")
    args = parser.parse_args()

    db = GexDB()

    if args.from_db:
        snapshot = _snapshot_from_db(db)
        source = "DB"
    else:
        try:
            snapshot = _snapshot_live()
        except Exception as exc:
            _log.warning("Fetch Deribit fallito (%s), ripiego sul DB", exc)
            snapshot = None
        source = "Deribit"
        if snapshot is None:
            snapshot = _snapshot_from_db(db)
            source = "DB (fallback)"

    if snapshot is None:
        _log.error("Nessuno snapshot GEX disponibile: né Deribit né DB hanno dati.")
        sys.exit(1)

    detector = RegimeDetector()
    detector.load_history_from_db(db.get_latest_n(90))
    state = detector.detect(snapshot)

    history = db.get_walls_series(days=args.history_days) if args.history_days > 0 else None

    ibit_ratio = None
    try:
        from src.flows.price_fetcher import PriceFetcher

        ibit_ratio = PriceFetcher().get_ibit_btc_ratio()
    except Exception as exc:
        _log.debug("Ratio IBIT/BTC non disponibile: %s", exc)

    script = build_pine_script(
        snapshot,
        regime=state.regime,
        gex_percentile=state.gex_percentile,
        history=history,
        max_strikes=args.max_strikes,
        range_pct=args.range_pct,
        ibit_ratio=ibit_ratio,
    )

    if args.stdout:
        print(script)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(script, encoding="utf-8")
    _log.info(
        "Indicatore scritto in %s (fonte: %s, flip=%s, put_wall=%s, call_wall=%s)",
        args.output, source, snapshot.gamma_flip_price, snapshot.put_wall, snapshot.call_wall,
    )
    print(f"✓ {args.output}  —  fonte {source}, {len(script.splitlines())} righe Pine")


if __name__ == "__main__":
    main()
