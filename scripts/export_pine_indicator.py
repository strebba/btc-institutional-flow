"""Genera un indicatore TradingView (Pine Script v6) con i livelli GEX correnti.

Pine Script non può fare chiamate HTTP: un indicatore non può leggere
`/api/gex` in tempo reale. Questo script tira lo snapshot GEX corrente
in-process (stesso path di `export_desk_note.py`, nessun server richiesto) e
lo congela in un file `.pine` con gamma flip, put/call wall e max pain come
livelli fissi. Per aggiornarli: rilancia lo script e incolla di nuovo il
contenuto nel Pine Editor di TradingView, sovrascrivendo la versione precedente.

La stessa generazione è disponibile anche da un tasto nella tab GEX della
dashboard Streamlit (`src/dashboard/tabs/gex.py`) — entrambi usano
`src.gex.pine_export.build_pine_indicator()`.

Uso:
    python3 scripts/export_pine_indicator.py                # -> out/tradingview/gex_levels.pine
    python3 scripts/export_pine_indicator.py --out /tmp/x.pine

Exit code:
    0 — file scritto
    2 — lo snapshot GEX non è disponibile (Deribit giù o dati insufficienti)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import setup_logging
from src.gex.pine_export import build_pine_indicator

_log = setup_logging("export_pine_indicator")

_DEFAULT_OUT = Path("out/tradingview/gex_levels.pine")


def _fetch_snapshot() -> dict:
    """Richiama l'endpoint GEX in-process e appiattisce la risposta.

    Stesso pattern di ``report._payload``, ma riporta la forma attesa da
    ``build_pine_indicator``: campi di ``snapshot`` + ``regime`` stringa.
    """
    from src.api.routers import gex as r_gex

    response = r_gex.get_gex()
    body = json.loads(bytes(response.body))
    data = body.get("data") or {}
    snapshot = dict(data.get("snapshot") or {})
    snapshot["regime"] = (data.get("regime") or {}).get("label")
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args()

    snapshot = _fetch_snapshot()
    if not snapshot.get("spot_price"):
        _log.error("Snapshot GEX non disponibile: nessun dato da congelare nell'indicatore.")
        return 2

    pine = build_pine_indicator(snapshot)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(pine, encoding="utf-8")
    _log.info("Indicatore scritto in %s (regime=%s, net_gex=%.1fM)",
               args.out, snapshot.get("regime"), snapshot.get("total_net_gex_m", 0))
    print(f"\nFatto: {args.out}")
    print("Apri il Pine Editor su TradingView, incolla il contenuto e clicca 'Add to chart'.")
    print("Per aggiornare i livelli: rilancia questo script e incolla di nuovo, sovrascrivendo tutto.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
