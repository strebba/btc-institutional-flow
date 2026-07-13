"""Raccolta del contesto dealer-flow live (GEX + flussi + macro) → SignalResult.

Single source of truth dell'assemblaggio dati del segnale multi-fattore: usata sia da
`scripts/cron_signal.py` (persistenza storica) sia da `scripts/cron_predict.py` (predizioni).
Richiede rete (Deribit, Farside, CoinGlass). Solleva `DataUnavailable` se i dati critici
(GEX o flussi) non sono disponibili; i fattori macro CoinGlass sono opzionali.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.analytics.factor_scorers import SignalInputs, SignalModel, SignalResult
from src.config import setup_logging
from src.gex.models import GexSnapshot

_log = setup_logging("forecast.context")

_BARRIER_EXCLUSION_PCT = 0.05


class DataUnavailable(RuntimeError):
    """Dati critici (GEX o flussi) non disponibili: il segnale non è calcolabile."""


@dataclass
class DealerFlowContext:
    """Contesto completo per costruire segnale + predizioni dealer-flow."""

    result: SignalResult
    snapshot: GexSnapshot
    spot: float
    inputs: SignalInputs
    ibit_flow_3d: float
    near_barrier: bool
    funding_rate_ann: Optional[float] = None
    oi_change_7d_pct: Optional[float] = None
    long_short_ratio: Optional[float] = None
    liquidations_long: Optional[float] = None
    liquidations_short: Optional[float] = None
    near_call_wall: bool = False
    ibit_flow_5d_ago: Optional[float] = None


def gather_dealer_flow_context(weights: Optional[dict[str, float]] = None) -> DealerFlowContext:
    """Recupera GEX/flussi/macro e calcola il segnale. Solleva DataUnavailable sui critici."""
    from src.gex.deribit_client import DeribitClient
    from src.gex.gex_calculator import GexCalculator
    from src.flows.scraper import FarsideScraper
    from src.flows.price_fetcher import PriceFetcher
    from src.flows.correlation import FlowCorrelation
    from src.edgar.structured_notes_db import StructuredNotesDB

    # ── GEX ───────────────────────────────────────────────────────────────────
    _log.info("Fetch GEX da Deribit...")
    try:
        client = DeribitClient()
        spot = client.get_spot_price()
        options = client.fetch_all_options("BTC")
    except Exception as exc:
        raise DataUnavailable(f"Fetch Deribit fallito: {exc}") from exc
    if not options:
        raise DataUnavailable("Nessuna opzione Deribit ricevuta")

    snapshot = GexCalculator().calculate_gex(options, spot)
    total_gex = snapshot.total_net_gex

    # ── Flussi ETF ────────────────────────────────────────────────────────────
    _log.info("Fetch flussi ETF da Farside...")
    try:
        scraper = FarsideScraper()
        agg_flows = scraper.aggregate(scraper.fetch())
        prices = PriceFetcher().get_all_prices()
        merged = FlowCorrelation().merge(agg_flows, prices)
    except Exception as exc:
        raise DataUnavailable(f"Fetch flussi fallito: {exc}") from exc

    ibit_flow_3d = 0.0
    ibit_flow_5d_ago: Optional[float] = None
    if not merged.empty and "ibit_flow_3d" in merged.columns:
        vals = merged["ibit_flow_3d"].dropna()
        if not vals.empty:
            ibit_flow_3d = float(vals.iloc[-1])
    # Granger lead: flusso aggregato di 5 giorni fa (colonna ibit_flow se presente)
    if not merged.empty:
        flow_col = "ibit_flow" if "ibit_flow" in merged.columns else None
        if flow_col is None and "ibit_flow_3d" in merged.columns:
            flow_col = "ibit_flow_3d"
        if flow_col is not None:
            flow_vals = merged[flow_col].dropna()
            if len(flow_vals) >= 6:
                ibit_flow_5d_ago = float(flow_vals.iloc[-6])

    # ── Barriere EDGAR ────────────────────────────────────────────────────────
    try:
        active_barriers = StructuredNotesDB().get_active_barriers()
    except Exception:
        active_barriers = []
    near_barrier = False
    if spot > 0:
        for b in active_barriers:
            bp = b.get("level_price_btc") or 0.0
            if bp > 0 and abs(spot - bp) / spot < _BARRIER_EXCLUSION_PCT:
                near_barrier = True
                break

    # ── Call wall proximity (positive gamma) ──────────────────────────────────
    _CALL_WALL_PIN_PCT = 0.02  # entro 2% = pinning meccanico
    near_call_wall = False
    if (
        spot > 0
        and total_gex > 0
        and snapshot.call_wall is not None
        and abs(spot - snapshot.call_wall) / spot < _CALL_WALL_PIN_PCT
    ):
        near_call_wall = True

    # ── Macro CoinGlass (opzionali) ────────────────────────────────────────────
    funding_rate_ann = oi_change_7d_pct = long_short_ratio = None
    liquidations_long = liquidations_short = None
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
                oi_7d_ago, oi_now = float(oi.iloc[-8]), float(oi.iloc[-1])
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

    # ── Segnale ────────────────────────────────────────────────────────────────
    inputs = SignalInputs(
        gex_usd=total_gex,
        etf_flow_3d_usd=ibit_flow_3d,
        funding_rate_annualized_pct=funding_rate_ann,
        oi_change_7d_pct=oi_change_7d_pct,
        long_short_ratio=long_short_ratio,
        put_call_ratio=snapshot.put_call_ratio,
        liquidations_long_24h_usd=liquidations_long,
        liquidations_short_24h_usd=liquidations_short,
        granger_lead_flow_usd=ibit_flow_5d_ago,
        near_active_barrier=near_barrier,
        near_call_wall=near_call_wall,
    )
    result = SignalModel(weights=weights).compute(inputs)

    return DealerFlowContext(
        result=result, snapshot=snapshot, spot=spot, inputs=inputs,
        ibit_flow_3d=ibit_flow_3d, near_barrier=near_barrier,
        funding_rate_ann=funding_rate_ann, oi_change_7d_pct=oi_change_7d_pct,
        long_short_ratio=long_short_ratio,
        liquidations_long=liquidations_long, liquidations_short=liquidations_short,
        near_call_wall=near_call_wall, ibit_flow_5d_ago=ibit_flow_5d_ago,
    )
