"""Test per router gex.py — endpoint /api/gex (route existence + error handling)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    from src.api import cache
    cache.cache_clear()
    from src.api.main import app
    return TestClient(app, raise_server_exceptions=False)


class TestGex:
    def test_upstream_error_returns_500(self, client):
        with patch("src.gex.deribit_client.DeribitClient") as mock_deribit:
            mock_deribit.return_value.get_spot_price.side_effect = RuntimeError("Deribit down")
            r = client.get("/api/gex")
            assert r.status_code == 500

    def test_enrichment_adds_multi_venue_oi_breakdown(self, client):
        """Il GEX Deribit-only è arricchito con OI per venue (incl. CME) e market coverage.

        Worked example: 2 opzioni con OI totale 150 contratti; CoinGlass dichiara
        Deribit 835 contratti e mercato totale 1000 → coverage Deribit 18.0%,
        market coverage 15.0%. CME presente nel breakdown con 2.0% di share.
        """
        from src.gex.models import GammaRegime

        options = [
            {
                "instrument_name": "BTC-TEST-100000-C",
                "strike": 100000.0,
                "option_type": "call",
                "gamma": 0.001,
                "open_interest": 100.0,
                "expiration_timestamp": 0,
                "mark_price": 100.0,
                "mark_iv": 50.0,
                "underlying_price": 100000.0,
                "best_bid": 99.0,
                "best_ask": 101.0,
                "delta": 0.5,
                "vega": 10.0,
            },
            {
                "instrument_name": "BTC-TEST-90000-P",
                "strike": 90000.0,
                "option_type": "put",
                "gamma": 0.001,
                "open_interest": 50.0,
                "expiration_timestamp": 0,
                "mark_price": 100.0,
                "mark_iv": 50.0,
                "underlying_price": 100000.0,
                "best_bid": 99.0,
                "best_ask": 101.0,
                "delta": -0.3,
                "vega": 10.0,
            },
        ]

        with patch("src.gex.deribit_client.DeribitClient") as mock_deribit, \
             patch("src.gex.gex_db.GexDB"), \
             patch("src.gex.regime_detector.RegimeDetector") as mock_detector, \
             patch("src.flows.coinglass_client.CoinGlassClient") as mock_cg_cls:
            mock_deribit.return_value.get_spot_price.return_value = 100000.0
            mock_deribit.return_value.fetch_all_options.return_value = options
            mock_detector.return_value.detect.return_value = GammaRegime(
                timestamp=datetime.now(timezone.utc),
                regime="positive_gamma",
                total_net_gex=1_000_000.0,
                spot_price=100000.0,
                put_wall=None,
                call_wall=None,
                gamma_flip=None,
                alerts=[],
                gex_percentile=None,
            )

            mock_cg = mock_cg_cls.return_value
            mock_cg.fetch_options_info.return_value = [
                {
                    "exchange_name": "All",
                    "open_interest": 1000.0,
                    "open_interest_usd": 100_000_000.0,
                    "oi_market_share": 100.0,
                },
                {
                    "exchange_name": "Deribit",
                    "open_interest": 835.0,
                    "open_interest_usd": 83_500_000.0,
                    "oi_market_share": 83.5,
                },
                {
                    "exchange_name": "CME",
                    "open_interest": 20.0,
                    "open_interest_usd": 2_000_000.0,
                    "oi_market_share": 2.0,
                },
            ]
            mock_cg.fetch_options_max_pain.return_value = []

            r = client.get("/api/gex")
            assert r.status_code == 200
            data = r.json()["data"]

            assert data["data_quality"]["coverage_pct"] == 18.0
            assert data["data_quality"]["market_coverage_pct"] == 15.0
            assert data["market_context"]["deribit_share_pct"] == 83.5

            cme = next(
                e for e in data["market_context"]["exchange_oi"] if e["exchange"] == "CME"
            )
            assert cme["open_interest_usd"] == 2_000_000
            assert cme["oi_market_share"] == 2.0
