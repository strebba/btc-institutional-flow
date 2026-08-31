"""Cron script: fotografia giornaliera di funding rate e open interest da CoinGecko.

Esiste per un motivo solo: CoinGecko espone l'open interest come **livello
istantaneo**, non come serie. Senza uno storico accumulato da noi la variazione a
7 giorni — che pesa 0,20 del pilastro macro — resterebbe scoperta per sempre.

Lo snapshot va nel DB **versionato** (`data/structured_notes.db`), non in
`runtime.db`: il filesystem di DO App Platform è effimero, quindi una serie
salvata lì sparirebbe a ogni redeploy e la finestra a 7 giorni non maturerebbe
mai. È lo stesso meccanismo che `cron_edgar.py` usa per le note SEC.

Uso:
    python3 scripts/cron_macro.py

Variabili d'ambiente:
    COINGECKO_API_KEY  Opzionale: l'endpoint /derivatives risponde anche senza,
                       la chiave del tier Demo alza solo i limiti di frequenza.

Exit code:
    0 — snapshot scritto
    1 — CoinGecko non ha restituito nulla di utilizzabile (niente scritto)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import setup_logging
from src.edgar.structured_notes_db import StructuredNotesDB
from src.flows.coingecko_client import CoinGeckoClient

_log = setup_logging("cron_macro")


def main() -> None:
    funding, oi_usd, n_contratti = CoinGeckoClient().fetch_funding_and_oi()

    if funding is None or oi_usd is None:
        # Mai scrivere una riga a metà: uno snapshot con oi_usd NULL non aiuta la
        # finestra a 7 giorni e sporca la serie con un buco che sembra un dato.
        _log.error("CoinGecko non ha restituito funding/OI utilizzabili: niente snapshot.")
        sys.exit(1)

    db = StructuredNotesDB()
    db.record_macro_snapshot(
        funding_ann_pct=funding, oi_usd=oi_usd, n_contracts=n_contratti
    )
    db.checkpoint()   # forza il WAL nel .db prima del commit git nel workflow

    variazione = db.get_oi_change_pct(7)
    testo_var = "n/d (storico ancora corto)" if variazione is None else f"{variazione:+.2f}%"

    print(
        f"[OK] snapshot macro: funding {funding:.2f}% ann · "
        f"OI ${oi_usd / 1e9:.1f}B su {n_contratti} perpetui · variazione 7g {testo_var}"
    )


if __name__ == "__main__":
    main()
