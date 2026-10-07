"""Pipeline flussi condivisa: l'asset arriva a scraper, prezzi e merge."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from src.api.data_pipeline import get_flow_context


def test_eth_attraversa_tutta_la_catena():
    scraper = MagicMock()
    fetcher = MagicMock()
    fetcher.get_all_prices.return_value = pd.DataFrame()
    corr = MagicMock()

    with patch("src.flows.scraper.FarsideScraper", return_value=scraper) as scraper_cls, \
         patch("src.flows.price_fetcher.PriceFetcher", return_value=fetcher), \
         patch("src.flows.correlation.FlowCorrelation", return_value=corr):
        get_flow_context(price_fallback=True, asset="eth")

    assert scraper_cls.call_args.kwargs["asset"] == "ETH"
    assert [c.args[0] for c in fetcher.fetch.call_args_list] == ["ETH-USD", "ETHA"]
    assert all(c.kwargs["asset"] == "ETH" for c in fetcher.get_all_prices.call_args_list)
    assert corr.merge.call_args.kwargs["asset"] == "ETH"
