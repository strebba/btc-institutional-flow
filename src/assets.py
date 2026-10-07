"""Registro degli asset tracciati: un solo punto di verità per le cablature per-asset.

Ogni sorgente (Deribit, Farside, CoinGlass, CoinGecko, yfinance) e ogni nome di
colonna del ``merged_df`` dipendono dall'asset. Invece di spargere ``"BTC"`` e
``"IBIT"`` nei moduli, i chiamanti ricevono un ``AssetSpec`` e leggono da lì.

BTC conserva i nomi legacy delle colonne (``btc_close``, ``ibit_flow``, …): sono
il contratto di analytics, API e PTF-Dashboard e non cambiano. ETH usa nomi con
prefisso (``eth_close``, ``etha_flow``, …).

``features`` dice cosa è disponibile per l'asset: ETH ha solo i dati (GEX, flussi,
macro); BTC anche barriere/EDGAR e ``analytics`` (regime analysis e alert flussi
tarati su storico e soglie IBIT).
"""
from __future__ import annotations

from dataclasses import dataclass

DATA_FEATURES = frozenset({"gex", "flows", "macro"})


@dataclass(frozen=True)
class AssetSpec:
    """Descrizione di un asset tracciato e delle sue sorgenti."""

    key: str
    label: str
    deribit_currency: str
    deribit_index: str
    spot_ticker: str
    lead_etf: str
    farside_url: str
    coinglass_etf_path: str
    coinglass_symbol: str
    coinglass_pair: str
    coingecko_index_id: str
    features: frozenset[str]

    @property
    def prefix(self) -> str:
        return self.key.lower()

    @property
    def price_col(self) -> str:
        return f"{self.prefix}_close"

    @property
    def return_col(self) -> str:
        return f"{self.prefix}_return"

    @property
    def vol_col(self) -> str:
        return f"{self.prefix}_vol_7d"

    @property
    def lead_prefix(self) -> str:
        return self.lead_etf.lower()

    @property
    def lead_flow_col(self) -> str:
        return f"{self.lead_prefix}_flow"

    def has(self, feature: str) -> bool:
        return feature in self.features


ASSETS: dict[str, AssetSpec] = {
    "BTC": AssetSpec(
        key="BTC",
        label="Bitcoin",
        deribit_currency="BTC",
        deribit_index="btc_usd",
        spot_ticker="BTC-USD",
        lead_etf="IBIT",
        farside_url="https://farside.co.uk/bitcoin-etf-flow-all-data/",
        coinglass_etf_path="bitcoin",
        coinglass_symbol="BTC",
        coinglass_pair="BTCUSDT",
        coingecko_index_id="BTC",
        features=DATA_FEATURES | {"analytics", "barriers", "edgar"},
    ),
    "ETH": AssetSpec(
        key="ETH",
        label="Ethereum",
        deribit_currency="ETH",
        deribit_index="eth_usd",
        spot_ticker="ETH-USD",
        lead_etf="ETHA",
        farside_url="https://farside.co.uk/ethereum-etf-flow-all-data/",
        coinglass_etf_path="ethereum",
        coinglass_symbol="ETH",
        coinglass_pair="ETHUSDT",
        coingecko_index_id="ETH",
        features=DATA_FEATURES,
    ),
}

DEFAULT_ASSET = "BTC"


def get_asset(key: str | None = None) -> AssetSpec:
    """Restituisce lo spec dell'asset (case-insensitive, default BTC).

    Raises:
        ValueError: se l'asset non è tracciato.
    """
    norm = (key or DEFAULT_ASSET).strip().upper()
    try:
        return ASSETS[norm]
    except KeyError:
        raise ValueError(f"Asset non tracciato: {key!r} (disponibili: {', '.join(ASSETS)})") from None
