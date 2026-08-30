"""Fetch unificato dei dati macro CoinGlass (funding/OI/L-S/liquidazioni).

Singola fonte di verità usata da /api/signals, /api/macro e dashboard data_loader.
Sostituisce i 3 blocchi duplicati di fetch macro che avevano caching inconsistente.

Ogni campo può essere ``None``, ma ``source_status`` dice **perché**: senza quella
distinzione "non lo sappiamo" e "il mercato è piatto" sono lo stesso valore, e il
pilastro macro finisce per pubblicare un giudizio che nessun dato sostiene.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

from src.flows.coinglass_client import CoinGlassClient, CoinGlassError

_log = logging.getLogger(__name__)

#: Almeno un fattore è stato letto davvero.
STATUS_OK = "ok"

#: Nessuna chiave CoinGlass configurata — è un problema di configurazione, non di
#: mercato, e si risolve valorizzando COINGLASS_API_KEY nella spec DO.
STATUS_NO_API_KEY = "no_api_key"

#: Chiave presente ma nessun dato ottenuto (API giù, tier insufficiente, rate limit).
STATUS_UNAVAILABLE = "unavailable"


@dataclass
class MacroData:
    funding_rate_annualized_pct: float | None = None
    oi_change_7d_pct: float | None = None
    long_short_ratio: float | None = None
    liquidations_long_24h_usd: float | None = None
    liquidations_short_24h_usd: float | None = None
    source_status: str = STATUS_UNAVAILABLE

    def to_dict(self) -> dict:
        return {
            "funding_rate_annualized_pct": self.funding_rate_annualized_pct,
            "oi_change_7d_pct": self.oi_change_7d_pct,
            "long_short_ratio": self.long_short_ratio,
            "liquidations_long_24h_usd": self.liquidations_long_24h_usd,
            "liquidations_short_24h_usd": self.liquidations_short_24h_usd,
            "source_status": self.source_status,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MacroData:
        return cls(
            funding_rate_annualized_pct=d.get("funding_rate_annualized_pct"),
            oi_change_7d_pct=d.get("oi_change_7d_pct"),
            long_short_ratio=d.get("long_short_ratio"),
            liquidations_long_24h_usd=d.get("liquidations_long_24h_usd"),
            liquidations_short_24h_usd=d.get("liquidations_short_24h_usd"),
            # una cache serializzata prima di questo campo resta leggibile
            source_status=d.get("source_status", STATUS_UNAVAILABLE),
        )

    def has_any_value(self) -> bool:
        """Vero se almeno un fattore è stato ottenuto."""
        return any(
            v is not None
            for v in (
                self.funding_rate_annualized_pct,
                self.oi_change_7d_pct,
                self.long_short_ratio,
                self.liquidations_long_24h_usd,
                self.liquidations_short_24h_usd,
            )
        )


def fetch_macro_data(
    *,
    cg_client=None,
    cache_data: dict | None = None,
) -> MacroData:
    """Fetch dati macro CoinGlass con fallback a cache opzionale.

    Args:
        cg_client: CoinGlassClient opzionale (se None, ne crea uno nuovo).
        cache_data: dict opzionale da cui leggere valori pre-esistenti
                    (es. cache_get("macro_data") nell'API).

    Returns:
        MacroData con i valori disponibili (None per quelli non fetchabili) e
        ``source_status`` che spiega il perché di eventuali None.
    """
    cache_data = cache_data or {}
    out = MacroData.from_dict(cache_data)
    cg = cg_client or CoinGlassClient()

    if not cg.has_api_key:
        # Senza chiave ogni chiamata fallirebbe: cinque richieste e cinque warning
        # per niente, a ogni giro di /api/signals. Meglio dirlo e fermarsi.
        _log.warning(
            "COINGLASS_API_KEY non configurata: pilastro macro senza dati. "
            "Impostala fra gli envs dell'app (o in config/settings.yaml) per abilitarlo."
        )
        out.source_status = STATUS_OK if out.has_any_value() else STATUS_NO_API_KEY
        return out

    if out.funding_rate_annualized_pct is None:
        try:
            fr = cg.fetch_funding_rate_history(days=14)
            if not fr.empty:
                out.funding_rate_annualized_pct = float(fr.iloc[-1]) * 3 * 365 * 100
        except (CoinGlassError, requests.RequestException) as exc:
            _log.warning("Funding rate fetch failed: %s", exc)

    if out.oi_change_7d_pct is None:
        try:
            oi = cg.fetch_aggregated_oi_history(days=14)
            if len(oi) >= 8 and float(oi.iloc[-8]) > 0:
                out.oi_change_7d_pct = (
                    (float(oi.iloc[-1]) - float(oi.iloc[-8]))
                    / float(oi.iloc[-8]) * 100
                )
        except (CoinGlassError, requests.RequestException, IndexError) as exc:
            _log.warning("OI history fetch failed: %s", exc)

    if out.long_short_ratio is None:
        try:
            ls = cg.fetch_long_short_ratio(days=3)
            if not ls.empty:
                out.long_short_ratio = float(ls.iloc[-1])
        except (CoinGlassError, requests.RequestException) as exc:
            _log.warning("Long/short ratio fetch failed: %s", exc)

    if out.liquidations_long_24h_usd is None:
        try:
            liq = cg.fetch_liquidations(days=2)
            if not liq.empty:
                out.liquidations_long_24h_usd = float(liq["long_usd"].iloc[-1])
                out.liquidations_short_24h_usd = float(liq["short_usd"].iloc[-1])
        except (CoinGlassError, requests.RequestException) as exc:
            _log.warning("Liquidations fetch failed: %s", exc)

    out.source_status = STATUS_OK if out.has_any_value() else STATUS_UNAVAILABLE
    return out
