"""Test per forecast/context.py — gather_dealer_flow_context."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.forecast.context import DataUnavailable, DealerFlowContext, gather_dealer_flow_context


class TestGatherDealerFlowContext:
    def test_raises_data_unavailable_when_deribit_fails(self):
        with patch("src.gex.deribit_client.DeribitClient") as mock_deribit:
            mock_deribit.return_value.get_spot_price.side_effect = RuntimeError("down")
            with pytest.raises(DataUnavailable, match="Fetch Deribit fallito"):
                gather_dealer_flow_context()

    def test_raises_data_unavailable_when_no_options(self):
        with patch("src.gex.deribit_client.DeribitClient") as mock_deribit:
            mock_client = mock_deribit.return_value
            mock_client.get_spot_price.return_value = 85000.0
            mock_client.fetch_all_options.return_value = []
            with pytest.raises(DataUnavailable, match="Nessuna opzione"):
                gather_dealer_flow_context()

    def test_raises_data_unavailable_when_flows_fail(self):
        with patch("src.gex.deribit_client.DeribitClient") as mock_deribit, \
             patch("src.gex.gex_calculator.GexCalculator") as mock_calc, \
             patch("src.flows.scraper.FarsideScraper") as mock_farside:
            mock_client = mock_deribit.return_value
            mock_client.get_spot_price.return_value = 85000.0
            mock_client.fetch_all_options.return_value = [{"name": "OPT"}]
            mock_snap = MagicMock()
            mock_snap.total_net_gex = 500e6
            mock_snap.put_call_ratio = 0.6
            mock_snap.call_wall = 90000.0
            mock_calc.return_value.calculate_gex.return_value = mock_snap
            mock_farside.return_value.fetch.side_effect = RuntimeError("Farside down")
            with pytest.raises(DataUnavailable, match="Fetch flussi fallito"):
                gather_dealer_flow_context()

    def test_barrier_failure_is_graceful(self):
        with patch("src.gex.deribit_client.DeribitClient") as mock_deribit, \
             patch("src.gex.gex_calculator.GexCalculator") as mock_calc, \
             patch("src.flows.scraper.FarsideScraper") as mock_farside, \
             patch("src.flows.price_fetcher.PriceFetcher") as mock_prices, \
             patch("src.flows.correlation.FlowCorrelation") as mock_corr, \
             patch("src.edgar.structured_notes_db.StructuredNotesDB") as mock_db, \
             patch("src.analytics.factor_scorers.SignalModel") as mock_signal, \
             patch("src.flows.coinglass_client.CoinGlassClient"):
            mock_client = mock_deribit.return_value
            mock_client.get_spot_price.return_value = 85000.0
            mock_client.fetch_all_options.return_value = [{"name": "OPT"}]
            mock_snap = MagicMock()
            mock_snap.total_net_gex = 500e6
            mock_snap.put_call_ratio = 0.6
            mock_snap.call_wall = None
            mock_calc.return_value.calculate_gex.return_value = mock_snap

            mock_farside.return_value.aggregate.return_value = []
            mock_prices.return_value.get_all_prices.return_value = MagicMock()
            merged = MagicMock()
            merged.empty = False
            mock_corr.return_value.merge.return_value = merged

            mock_db.return_value.get_active_barriers.side_effect = RuntimeError("DB down")

            mock_result = MagicMock()
            mock_result.score = 50.0
            mock_result.signal = "CAUTION"
            mock_signal.return_value.compute.return_value = mock_result

            ctx = gather_dealer_flow_context()
            assert isinstance(ctx, DealerFlowContext)
            assert ctx.near_barrier is False


class TestDealerFlowContext:
    def test_all_fields_set(self):
        ctx = DealerFlowContext(
            result=MagicMock(),
            snapshot=MagicMock(),
            spot=85000.0,
            inputs=MagicMock(),
            ibit_flow_3d=100e6,
            near_barrier=False,
        )
        assert ctx.spot == 85000.0
        assert ctx.ibit_flow_3d == 100e6
        assert ctx.near_barrier is False
