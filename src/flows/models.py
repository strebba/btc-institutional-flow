"""Dataclass per i flussi ETF."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class EtfFlowData:
    """Flusso netto giornaliero per un singolo ETF Bitcoin.

    Attributes:
        date: data del flusso.
        ticker: ticker dell'ETF (es. "IBIT", "FBTC", "BITB").
        flow_usd: flusso netto in USD (positivo = inflow, negativo = outflow).
        source: sorgente del dato ("farside", "sosovalue").
    """

    date: date
    ticker: str
    flow_usd: float
    source: str = "farside"


@dataclass
class AggregateFlows:
    """Flussi aggregati per data (tutti gli ETF Bitcoin spot).

    Attributes:
        date: data.
        total_flow_usd: flusso totale aggregato.
        ibit_flow_usd: flusso IBIT.
        flows_by_ticker: flusso per ciascun ETF.
    """

    date: date
    total_flow_usd: float
    ibit_flow_usd: float
    flows_by_ticker: dict[str, float] = field(default_factory=dict)

