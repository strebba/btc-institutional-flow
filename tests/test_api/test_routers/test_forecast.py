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


class TestVerify:
    def test_verify_returns_200(self, client):
        with patch("src.forecast.prediction_db.PredictionDB"), \
             patch("src.flows.price_fetcher.PriceFetcher"), \
             patch("src.forecast.verifier.score_due_predictions", return_value=[]):
            r = client.post("/api/predictions/verify")
            assert r.status_code == 200
