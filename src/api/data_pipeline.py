"""Pipeline dati condivisa: Farside → PriceFetcher → FlowCorrelation.

Single source of truth per tutti gli endpoint API e l'alert monitor.
Riduce la duplicazione 7x del pattern:
    FarsideScraper → aggregate → PriceFetcher.get_all_prices → FlowCorrelation.merge
"""
from __future__ import annotations

from src.config import setup_logging
from src.flows.correlation import FlowCorrelation
from src.flows.price_fetcher import PriceFetcher
from src.flows.scraper import FarsideScraper

_log = setup_logging("api.data_pipeline")


def get_flow_context():
    """Fetch ETF flows + prezzi e restituisci il contesto completo.

    Returns:
        dict con chiavi:
            merged_df: pd.DataFrame (flows + prezzi uniti)
            agg: list[AggregateFlows]
            prices: pd.DataFrame
    """
    scraper = FarsideScraper()
    raw = scraper.fetch()
    agg = scraper.aggregate(raw)

    fetcher = PriceFetcher()
    prices = fetcher.get_all_prices()

    merged = FlowCorrelation().merge(agg, prices)

    return {
        "merged_df": merged,
        "agg": agg,
        "prices": prices,
    }
