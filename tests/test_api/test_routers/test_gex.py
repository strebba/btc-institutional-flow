"""Test per router gex.py — endpoint /api/gex (route existence + error handling)."""

from __future__ import annotations

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


class TestGexPine:
    """/api/gex/pine — indicatore TradingView con i livelli GEX incorporati."""

    @pytest.fixture()
    def gex_data(self):
        from datetime import datetime, timezone
        from types import SimpleNamespace

        from src.gex.models import GexByStrike, GexSnapshot

        snapshot = GexSnapshot(
            timestamp=datetime(2026, 8, 22, 10, 0, tzinfo=timezone.utc),
            spot_price=100_000.0,
            total_net_gex=4.5e8,
            gamma_flip_price=98_500.0,
            put_wall=92_000.0,
            call_wall=108_000.0,
            max_pain=99_000.0,
            gex_by_strike=[
                GexByStrike(strike=95_000.0, call_gex=1e6, put_gex=-3e6, net_gex=-2e6),
                GexByStrike(strike=105_000.0, call_gex=4e6, put_gex=-1e6, net_gex=3e6),
            ],
            total_call_oi=1_000.0,
            total_put_oi=1_200.0,
        )
        state = SimpleNamespace(regime="positive_gamma", alerts=[], gex_percentile=70.0)
        gex_db = SimpleNamespace(get_walls_series=lambda days=365: None)
        return {"snapshot": snapshot, "spot": 100_000.0, "state": state, "gex_db": gex_db}

    def test_ritorna_pine_script(self, client, gex_data):
        with patch("src.api.routers.gex._get_gex_data", return_value=gex_data):
            r = client.get("/api/gex/pine")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/plain")
        assert r.text.startswith("//@version=6")
        assert "FLIP        = 98500.0" in r.text
        assert 'REGIME      = "positive_gamma"' in r.text

    def test_download_imposta_content_disposition(self, client, gex_data):
        with patch("src.api.routers.gex._get_gex_data", return_value=gex_data):
            r = client.get("/api/gex/pine", params={"download": True})
        assert r.status_code == 200
        assert "attachment" in r.headers["content-disposition"]
        assert "btc_gex_tradingview.pine" in r.headers["content-disposition"]

    def test_history_days_zero_salta_la_query_al_db(self, client, gex_data):
        called = {"walls": False}

        def _walls(days=365):
            called["walls"] = True
            return None

        gex_data["gex_db"].get_walls_series = _walls
        with patch("src.api.routers.gex._get_gex_data", return_value=gex_data):
            r = client.get("/api/gex/pine", params={"history_days": 0})
        assert r.status_code == 200
        assert called["walls"] is False
        assert "Storico: 0 punti" in r.text

    def test_parametri_fuori_range_rifiutati(self, client, gex_data):
        with patch("src.api.routers.gex._get_gex_data", return_value=gex_data):
            assert client.get("/api/gex/pine", params={"history_days": -1}).status_code == 422
            assert client.get("/api/gex/pine", params={"range_pct": 0}).status_code == 422

    def test_upstream_error_returns_500(self, client):
        with patch("src.api.routers.gex._get_gex_data", side_effect=RuntimeError("Deribit down")):
            r = client.get("/api/gex/pine")
        assert r.status_code == 500
