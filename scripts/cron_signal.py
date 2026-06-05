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
    from src.gex.deribit_client import DeribitClient
    from src.gex.gex_calculator import GexCalculator
    from src.flows.scraper import FarsideScraper
    from src.flows.price_fetcher import PriceFetcher
    from src.flows.correlation import FlowCorrelation
    from src.edgar.structured_notes_db import StructuredNotesDB
    from src.analytics.signal_model import SignalModel, SignalInputs
    from src.analytics.signal_db import SignalDB

    # ── GEX ───────────────────────────────────────────────────────────────────
    _log.info("Fetch GEX da Deribit...")
    try:
        client = DeribitClient()
        spot = client.get_spot_price()
        options = client.fetch_all_options("BTC")
    except Exception as exc:
        _log.error("Fetch Deribit fallito: %s", exc)
        return 1

    if not options:
        _log.error("Nessuna opzione Deribit ricevuta")
        return 1

    calc = GexCalculator()
    snapshot = calc.calculate_gex(options, spot)
    total_gex = snapshot.total_net_gex

    # ── Flussi ETF ────────────────────────────────────────────────────────────
    _log.info("Fetch flussi ETF da Farside...")
    try:
        scraper = FarsideScraper()
        raw_flows = scraper.fetch()
        agg_flows = scraper.aggregate(raw_flows)
        prices = PriceFetcher().get_all_prices()
        merged = FlowCorrelation().merge(agg_flows, prices)
    except Exception as exc:
        _log.error("Fetch flussi fallito: %s", exc)
        return 1

    ibit_flow_3d = 0.0
    if not merged.empty and "ibit_flow_3d" in merged.columns:
        last_val = merged["ibit_flow_3d"].dropna()
        if not last_val.empty:
            ibit_flow_3d = float(last_val.iloc[-1])

    # ── Barriere ──────────────────────────────────────────────────────────────
    try:
        active_barriers = StructuredNotesDB().get_active_barriers()
    except Exception:
        active_barriers = []

    barrier_exclusion_pct = 0.05
    near_barrier = False
    if spot > 0:
        for b in active_barriers:
            bp = b.get("level_price_btc") or 0.0
            if bp > 0 and abs(spot - bp) / spot < barrier_exclusion_pct:
                near_barrier = True
                break

    # ── Macro CoinGlass (opzionali — failure non blocca) ─────────────────────
    funding_rate_ann: float | None = None
    oi_change_7d_pct: float | None = None
    long_short_ratio: float | None = None
    liquidations_long: float | None = None
    liquidations_short: float | None = None

    try:
        from src.flows.coinglass_client import CoinGlassClient
        cg = CoinGlassClient()

        try:
            fr = cg.fetch_funding_rate_history(days=14)
            if not fr.empty:
                funding_rate_ann = float(fr.iloc[-1]) * 3 * 365 * 100
        except Exception as exc:
            _log.warning("Funding rate non disponibile: %s", exc)

        try:
            oi = cg.fetch_aggregated_oi_history(days=14)
            if len(oi) >= 7:
                oi_7d_ago = float(oi.iloc[-8])
                oi_now = float(oi.iloc[-1])
                if oi_7d_ago > 0:
                    oi_change_7d_pct = (oi_now - oi_7d_ago) / oi_7d_ago * 100
        except Exception as exc:
            _log.warning("OI change non disponibile: %s", exc)

        try:
            ls = cg.fetch_long_short_ratio(days=3)
            if not ls.empty:
                long_short_ratio = float(ls.iloc[-1])
        except Exception as exc:
            _log.warning("Long/short ratio non disponibile: %s", exc)

        try:
            liq = cg.fetch_liquidations(days=2)
            if not liq.empty:
                liquidations_long = float(liq["long_usd"].iloc[-1])
                liquidations_short = float(liq["short_usd"].iloc[-1])
        except Exception as exc:
            _log.warning("Liquidazioni non disponibili: %s", exc)

    except Exception as exc:
        _log.warning("CoinGlass client non disponibile: %s", exc)

    # ── Calcolo segnale ───────────────────────────────────────────────────────
    inputs = SignalInputs(
        gex_usd=total_gex,
        etf_flow_3d_usd=ibit_flow_3d,
        funding_rate_annualized_pct=funding_rate_ann,
        oi_change_7d_pct=oi_change_7d_pct,
        long_short_ratio=long_short_ratio,
        put_call_ratio=snapshot.put_call_ratio,
        liquidations_long_24h_usd=liquidations_long,
        liquidations_short_24h_usd=liquidations_short,
        near_active_barrier=near_barrier,
    )
    result = SignalModel().compute(inputs)

    # ── Salvataggio ───────────────────────────────────────────────────────────
    inserted = SignalDB().insert(
        result,
        spot_price_usd=spot,
        total_gex_usd=total_gex,
        ibit_flow_3d_usd=ibit_flow_3d,
        funding_rate_pct=funding_rate_ann,
        oi_change_7d_pct=oi_change_7d_pct,
        long_short_ratio=long_short_ratio,
        put_call_ratio=snapshot.put_call_ratio,
        liq_long_usd=liquidations_long,
        liq_short_usd=liquidations_short,
        near_active_barrier=near_barrier,
    )

    status = "salvato" if inserted else "già presente (dup)"
    print(
        f"[OK] {result.signal} score={result.score:.1f} "
        f"spot={spot:,.0f} gex={total_gex/1e6:+.1f}M "
        f"flow3d={ibit_flow_3d/1e6:+.1f}M — {status}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
