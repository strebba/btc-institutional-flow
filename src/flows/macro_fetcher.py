"""Fetch unificato dei dati macro CoinGlass (funding/OI/L-S/liquidazioni).

Singola fonte di verità usata da /api/signals, /api/macro e dashboard data_loader.
Sostituisce i 3 blocchi duplicati di fetch macro che avevano caching inconsistente.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

_log = logging.getLogger(__name__)


@dataclass
class MacroData:
    funding_rate_annualized_pct: Optional[float] = None
    oi_change_7d_pct: Optional[float] = None
    long_short_ratio: Optional[float] = None
    liquidations_long_24h_usd: Optional[float] = None
    liquidations_short_24h_usd: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "funding_rate_annualized_pct": self.funding_rate_annualized_pct,
            "oi_change_7d_pct": self.oi_change_7d_pct,
            "long_short_ratio": self.long_short_ratio,
            "liquidations_long_24h_usd": self.liquidations_long_24h_usd,
            "liquidations_short_24h_usd": self.liquidations_short_24h_usd,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MacroData:
        return cls(
            funding_rate_annualized_pct=d.get("funding_rate_annualized_pct"),
            oi_change_7d_pct=d.get("oi_change_7d_pct"),
            long_short_ratio=d.get("long_short_ratio"),
            liquidations_long_24h_usd=d.get("liquidations_long_24h_usd"),
            liquidations_short_24h_usd=d.get("liquidations_short_24h_usd"),
        )


def fetch_macro_data(
    *,
    cg_client=None,
    cache_data: Optional[dict] = None,
) -> MacroData:
    """Fetch dati macro CoinGlass con fallback a cache opzionale.

    Args:
        cg_client: CoinGlassClient opzionale (se None, ne crea uno nuovo).
        cache_data: dict opzionale da cui leggere valori pre-esistenti
                    (es. cache_get("macro_data") nell'API).

    Returns:
        MacroData con i valori disponibili (None per quelli non fetchabili).
    """
    from src.flows.coinglass_client import CoinGlassClient

    cache_data = cache_data or {}
    out = MacroData.from_dict(cache_data)
    cg = cg_client or CoinGlassClient()

    if out.funding_rate_annualized_pct is None:
        try:
            fr = cg.fetch_funding_rate_history(days=14)
            if not fr.empty:
                out.funding_rate_annualized_pct = float(fr.iloc[-1]) * 3 * 365 * 100
        except Exception:
            pass

    if out.oi_change_7d_pct is None:
        try:
            oi = cg.fetch_aggregated_oi_history(days=14)
            if len(oi) >= 8 and float(oi.iloc[-8]) > 0:
                out.oi_change_7d_pct = (
                    (float(oi.iloc[-1]) - float(oi.iloc[-8]))
                    / float(oi.iloc[-8]) * 100
                )
        except Exception:
            pass

    if out.long_short_ratio is None:
        try:
            ls = cg.fetch_long_short_ratio(days=3)
            if not ls.empty:
                out.long_short_ratio = float(ls.iloc[-1])
        except Exception:
            pass

    if out.liquidations_long_24h_usd is None:
        try:
            liq = cg.fetch_liquidations(days=2)
            if not liq.empty:
                out.liquidations_long_24h_usd = float(liq["long_usd"].iloc[-1])
                out.liquidations_short_24h_usd = float(liq["short_usd"].iloc[-1])
        except Exception:
            pass

    return out
