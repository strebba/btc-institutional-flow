"""Dataclass per i flussi ETF e i prezzi di mercato."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


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


@dataclass
class PriceData:
    """OHLCV giornaliero per un asset.

    Attributes:
        date: data.
        ticker: es. "BTC-USD", "IBIT".
        open: prezzo apertura.
        high: massimo.
        low: minimo.
        close: prezzo chiusura.
        volume: volume.
        daily_return: rendimento giornaliero logaritmico.
    """

    date: date
    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    daily_return: Optional[float] = None


@dataclass
class MergedRecord:
    """Record giornaliero con flussi + prezzi uniti.

    Attributes:
        date: data.
        ibit_flow: flusso IBIT in USD.
        total_flow: flusso totale ETF in USD.
        btc_close: prezzo chiusura BTC.
        btc_return: rendimento BTC (log).
        ibit_close: prezzo chiusura IBIT.
        ibit_btc_ratio: rapporto IBIT/BTC.
        btc_realized_vol_7d: volatilità realizzata 7gg di BTC.
    """

    date: date
    ibit_flow: Optional[float]
    total_flow: Optional[float]
    btc_close: Optional[float]
    btc_return: Optional[float]
    ibit_close: Optional[float]
    ibit_btc_ratio: Optional[float] = None
    btc_realized_vol_7d: Optional[float] = None
