"""Pipeline dati condivisa: Farside → PriceFetcher → FlowCorrelation.

Single source of truth per gli endpoint API, la dashboard e l'alert monitor:

    FarsideScraper → aggregate → PriceFetcher.get_all_prices → FlowCorrelation.merge

Gli import delle classi sono pigri (dentro ``get_flow_context``) così i test
possono patchare i simboli nei moduli di origine (``src.flows.scraper.FarsideScraper``,
``src.flows.price_fetcher.PriceFetcher``, …) e la patch ha effetto.
"""
from __future__ import annotations


def get_flow_context(*, price_fallback: bool = False) -> dict:
    """Fetch ETF flows + prezzi e restituisci il contesto completo.

    Args:
        price_fallback: se True e i prezzi risultano vuoti, forza un download
            yfinance di BTC-USD/IBIT e riprova (comportamento della dashboard
            al primo avvio, quando il DB prezzi è ancora vuoto).

    Returns:
        dict con chiavi:
            raw: list[EtfFlowData] (fetch grezzo Farside)
            agg: list[AggregateFlows]
            prices: pd.DataFrame
            merged_df: pd.DataFrame (flows + prezzi uniti)
    """
    from src.flows.correlation import FlowCorrelation
    from src.flows.price_fetcher import PriceFetcher
    from src.flows.scraper import FarsideScraper

    scraper = FarsideScraper()
    raw = scraper.fetch()
    agg = scraper.aggregate(raw)

    fetcher = PriceFetcher()
    prices = fetcher.get_all_prices()
    if price_fallback and prices.empty:
        fetcher.fetch("BTC-USD")
        fetcher.fetch("IBIT")
        prices = fetcher.get_all_prices()

    merged = FlowCorrelation().merge(agg, prices)

    return {"raw": raw, "agg": agg, "prices": prices, "merged_df": merged}
