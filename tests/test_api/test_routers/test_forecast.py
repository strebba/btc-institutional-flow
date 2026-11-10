"""Test per router forecast.py — endpoint forecast e predictions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    from src.api import cache
    cache.cache_clear()
    from src.api.main import app
    return TestClient(app, raise_server_exceptions=False)


class TestPredictions:
    def test_get_predictions_returns_200(self, client):
        with patch("src.forecast.prediction_db.PredictionDB") as mock_db:
            mock_db.return_value.get_with_outcomes.return_value = []
            r = client.get("/api/predictions")
            assert r.status_code == 200
            body = r.json()
            assert body["status"] == "ok"
            assert body["data"]["count"] == 0

    def test_risk_map_is_added_without_removing_the_score(self, client):
        row = {
            "id": 1,
            "source": "dealer_flow",
            "score_ref": 72.0,
            "metadata": '{"risk_map": {"expected_range_5d_pct": 6.5}, "levels_pit": {"max_pain": 100.0}, "reconstructed": true}',
        }
        with patch("src.forecast.prediction_db.PredictionDB") as mock_db:
            mock_db.return_value.get_with_outcomes.return_value = [row]
            r = client.get("/api/predictions")
        body = r.json()["data"]["predictions"][0]
        assert body["score_ref"] == 72.0
        assert body["metadata"].startswith("{")
        assert body["risk_map"]["expected_range_5d_pct"] == 6.5
        assert body["levels_pit"]["max_pain"] == 100.0
        assert "reconstructed" in body["metadata"]


class TestForecastStatus:
    def test_status_returns_200(self, client):
        mock_scheduler = MagicMock()
        mock_scheduler.__bool__.return_value = True
        with patch("src.api.scheduler._forecast_scheduler", mock_scheduler), \
             patch("src.forecast.prediction_db.PredictionDB") as mock_db, \
             patch("src.forecast.calibration.load_weights_config", return_value={"governance": {}}):
            mock_db.return_value.get_recent.return_value = []
            mock_db.return_value.get_open.return_value = []
            mock_db.return_value.count.return_value = 10
            r = client.get("/api/forecast/status")
            assert r.status_code == 200
            body = r.json()
            assert body["status"] == "ok"
            assert body["data"]["last_prediction"] is None
            assert body["data"]["total"] == 10
            assert body["data"]["freeze_weights"] is True
            assert body["data"]["risk_map"] is None


class TestRiskMapEndpoint:
    def test_returns_the_latest_stored_map(self, client):
        pred = MagicMock()
        pred.created_at = "2026-11-10T07:30:00"
        pred.source = "dealer_flow"
        pred.metadata = {
            "risk_map": {"expected_range_5d_usd": 4200.0, "levels": [
                {"name": "max_pain", "touch_prob_5d": 0.42},
            ]},
            "levels_pit": {"max_pain": 100_000.0, "gex_sign": 1},
        }
        older = MagicMock()
        older.created_at = "2026-11-09T07:30:00"
        older.source = "dealer_flow"
        older.metadata = None
        with patch("src.forecast.prediction_db.PredictionDB") as mock_db:
            # get_recent è dal più recente: la mappa di oggi viene prima di ieri.
            mock_db.return_value.get_recent.return_value = [pred, older]
            r = client.get("/api/forecast/risk-map")
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["date"] == "2026-11-10"
        assert data["risk_map"]["expected_range_5d_usd"] == 4200.0
        assert data["levels_pit"]["gex_sign"] == 1
        assert data["risk_map"]["levels"][0]["touch_prob_5d"] == 0.42


class TestVerify:
    def test_verify_returns_200(self, client):
        with patch("src.forecast.prediction_db.PredictionDB"), \
             patch("src.flows.price_fetcher.PriceFetcher"), \
             patch("src.forecast.verifier.score_due_predictions", return_value=[]):
            r = client.post("/api/predictions/verify")
            assert r.status_code == 200
