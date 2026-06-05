"""Cron script: calcola il segnale multi-fattore e lo salva nel DB.

Usa gli stessi data source dell'endpoint /api/signals ma in modo standalone,
senza richiedere che il server API sia in esecuzione.

Uso:
    python3 scripts/cron_signal.py

Scheduling suggerito (crontab, ogni ora nei giorni lavorativi):
    0 * * * 1-5  /path/venv/bin/python3 /path/scripts/cron_signal.py

Exit code:
    0 — segnale calcolato e salvato con successo
    1 — fetch dati critico fallito (GEX o flussi non disponibili)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import setup_logging

_log = setup_logging("cron_signal")


def main() -> int:
    from src.analytics.signal_db import SignalDB
    from src.forecast.context import DataUnavailable, gather_dealer_flow_context

    try:
        ctx = gather_dealer_flow_context()
    except DataUnavailable as exc:
        _log.error("%s", exc)
        return 1

    inserted = SignalDB().insert(
        ctx.result,
        spot_price_usd=ctx.spot,
        total_gex_usd=ctx.snapshot.total_net_gex,
        ibit_flow_3d_usd=ctx.ibit_flow_3d,
        funding_rate_pct=ctx.funding_rate_ann,
        oi_change_7d_pct=ctx.oi_change_7d_pct,
        long_short_ratio=ctx.long_short_ratio,
        put_call_ratio=ctx.snapshot.put_call_ratio,
        liq_long_usd=ctx.liquidations_long,
        liq_short_usd=ctx.liquidations_short,
        near_active_barrier=ctx.near_barrier,
    )

    status = "salvato" if inserted else "già presente (dup)"
    print(
        f"[OK] {ctx.result.signal} score={ctx.result.score:.1f} "
        f"spot={ctx.spot:,.0f} gex={ctx.snapshot.total_net_gex/1e6:+.1f}M "
        f"flow3d={ctx.ibit_flow_3d/1e6:+.1f}M — {status}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
