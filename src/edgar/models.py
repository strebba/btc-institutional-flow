"""Dataclass per le note strutturate e i barrier levels."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional


@dataclass
class BarrierLevel:
    """Singolo livello di barriera di una nota strutturata.

    Attributes:
        barrier_type: tipo di barriera (knock_in, autocall, buffer, knock_out).
        level_pct: percentuale rispetto al valore iniziale (es. 70.0 = 70%).
        level_price_ibit: prezzo IBIT corrispondente alla barriera.
        level_price_btc: prezzo BTC corrispondente (calcolato tramite ratio).
        observation_date: data di osservazione della barriera (None = continua).
        status: active | triggered | expired.
        note_id: FK verso StructuredNote.id (None prima del salvataggio).
        id: PK nel DB (None prima del salvataggio).
    """

    barrier_type: str
    level_pct: float
    level_price_ibit: Optional[float] = None
    level_price_btc: Optional[float] = None
    observation_date: Optional[date] = None
    status: str = "active"
    note_id: Optional[int] = None
    id: Optional[int] = None


@dataclass
class StructuredNote:
    """Nota strutturata emessa su IBIT, estratta da un filing SEC 424B2/424B3.

    Attributes:
        filing_url: URL diretto al documento EDGAR.
        issuer: banca emittente (es. "JPMorgan").
        issue_date: data di emissione.
        maturity_date: data di scadenza.
        notional_usd: valore nozionale in USD.
        product_type: tipo prodotto (autocallable, barrier, buffer, ...).
        underlying: sottostante (tipicamente "IBIT").
        initial_level: prezzo IBIT alla data di pricing.
        autocall_trigger_pct: livello di autocall in % (es. 100.0 = 100%).
        knockin_barrier_pct: livello knock-in in % (es. 70.0 = 70%).
        buffer_pct: buffer di protezione in % (es. 10.0 = 10%).
        participation_rate: tasso di partecipazione per note con leva.
        coupon_rate: cedola annua in % se prevista.
        observation_dates: lista di date di osservazione (JSON-serializzabile).
        barriers: lista di BarrierLevel associati.
        raw_text: testo grezzo del filing (troncato a 50k caratteri).
        created_at: timestamp di creazione del record.
        id: PK nel DB (None prima del salvataggio).
    """

    filing_url: str
    issuer: Optional[str] = None
    issue_date: Optional[date] = None
    maturity_date: Optional[date] = None
    notional_usd: Optional[float] = None
    product_type: Optional[str] = None
    underlying: str = "IBIT"
    initial_level: Optional[float] = None
    autocall_trigger_pct: Optional[float] = None
    knockin_barrier_pct: Optional[float] = None
    buffer_pct: Optional[float] = None
    participation_rate: Optional[float] = None
    coupon_rate: Optional[float] = None
    observation_dates: list[str] = field(default_factory=list)
    barriers: list[BarrierLevel] = field(default_factory=list)
    raw_text: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    id: Optional[int] = None
