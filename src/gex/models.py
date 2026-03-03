"""Dataclass per il Gamma Exposure (GEX) delle opzioni BTC su Deribit."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional


@dataclass
class GexByStrike:
    """GEX aggregato per uno specifico strike price.

    Attributes:
        strike: prezzo di esercizio in USD.
        call_gex: GEX totale delle call a questo strike.
        put_gex: GEX totale delle put a questo strike.
        net_gex: call_gex + put_gex.
        call_oi: open interest totale call.
        put_oi: open interest totale put.
    """

    strike: float
    call_gex: float = 0.0
    put_gex: float  = 0.0
    net_gex: float  = 0.0
    call_oi: float  = 0.0
    put_oi: float   = 0.0


@dataclass
class GexSnapshot:
    """Snapshot completo del GEX calcolato in un dato momento.

    Attributes:
        timestamp: istante del calcolo.
        spot_price: prezzo spot BTC al momento del calcolo.
        total_net_gex: somma di tutti i GEX → positivo = regime stabilizzante.
        gamma_flip_price: prezzo dove il GEX cumulativo (dal basso) cambia segno.
        put_wall: strike con il massimo |GEX| negativo (supporto meccanico).
        call_wall: strike con il massimo GEX positivo (resistenza meccanica).
        max_pain: strike che minimizza il payoff totale delle opzioni.
        gex_by_strike: lista di GexByStrike per ogni strike attivo.
        total_call_oi: open interest totale delle call.
        total_put_oi: open interest totale delle put.
        put_call_ratio: rapporto put OI / call OI.
        distance_to_put_wall_pct: distanza spot → put_wall in %.
        distance_to_call_wall_pct: distanza spot → call_wall in %.
    """

    timestamp: datetime
    spot_price: float
    total_net_gex: float
    gamma_flip_price: Optional[float]
    put_wall: Optional[float]
    call_wall: Optional[float]
    max_pain: Optional[float]
    gex_by_strike: list[GexByStrike] = field(default_factory=list)
    total_call_oi: float = 0.0
    total_put_oi:  float = 0.0
    put_call_ratio: Optional[float] = None
    distance_to_put_wall_pct:  Optional[float] = None
    distance_to_call_wall_pct: Optional[float] = None

    def __post_init__(self) -> None:
        if self.put_wall and self.spot_price:
            self.distance_to_put_wall_pct = (
                (self.put_wall - self.spot_price) / self.spot_price * 100
            )
        if self.call_wall and self.spot_price:
            self.distance_to_call_wall_pct = (
                (self.call_wall - self.spot_price) / self.spot_price * 100
            )
        if self.total_call_oi and self.total_put_oi:
            self.put_call_ratio = self.total_put_oi / self.total_call_oi


@dataclass
class RegimeState:
    """Stato del regime di gamma in un dato momento.

    Attributes:
        timestamp: istante di rilevazione.
        regime: 'positive_gamma' | 'negative_gamma' | 'neutral'.
        total_net_gex: GEX totale netto.
        spot_price: prezzo spot BTC.
        put_wall: prezzo put wall.
        call_wall: prezzo call wall.
        gamma_flip: prezzo gamma flip.
        alerts: lista di alert attivi.
        gex_percentile: percentile storico del GEX (0-100).
    """

    timestamp: datetime
    regime: str
    total_net_gex: float
    spot_price: float
    put_wall: Optional[float]
    call_wall: Optional[float]
    gamma_flip: Optional[float]
    alerts: list[str] = field(default_factory=list)
    gex_percentile: Optional[float] = None
