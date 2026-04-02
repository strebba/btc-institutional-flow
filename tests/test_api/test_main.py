"""Integration tests per gli endpoint FastAPI.

Usa httpx.TestClient per testare l'intera stack API (routing, cache,
sanitizzazione, serializzazione) con i data layer mockati.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    """TestClient con cache pulita ad ogni test."""
    from src.api import main as api_module
    api_module._cache.clear()
    from src.api.main import app
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def authed_client(monkeypatch):
    """TestClient con API_KEY configurata."""
    monkeypatch.setenv("API_KEY", "test-secret")
    from src.api import main as api_module
    api_module._cache.clear()
    # Ricrea _API_KEY leggendo l'env var aggiornata
    api_module._API_KEY = "test-secret"
    from src.api.main import app
    return TestClient(app, raise_server_exceptions=False)


# ──────────────────────────────────────────────────────────────────────────────
# /api/health
# ──────────────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200

    def test_envelope_structure(self, client):
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert "timestamp" in body
        assert "data" in body

    def test_healthy_true(self, client):
        data = client.get("/api/health").json()["data"]
        assert data["healthy"] is True
        assert data["service"] == "btc-institutional-flow"


# ──────────────────────────────────────────────────────────────────────────────
# /api/gex
# ──────────────────────────────────────────────────────────────────────────────

def _mock_gex_snapshot():
    from src.gex.models import GexSnapshot, GexByStrike, RegimeState
    from datetime import datetime, timezone
    snap = GexSnapshot(
        timestamp=datetime.now(timezone.utc),
        spot_price=85_000.0,
        total_net_gex=500_000_000.0,
        gamma_flip_price=82_000.0,
        put_wall=80_000.0,
        call_wall=90_000.0,
        max_pain=84_000.0,
        gex_by_strike=[
            GexByStrike(strike=80_000, call_gex=1e8, put_gex=-5e7, net_gex=5e7, call_oi=100, put_oi=50),
        ],
        total_call_oi=1000.0,
        total_put_oi=800.0,
    )
    regime = RegimeState(
        timestamp=datetime.now(timezone.utc),
        regime="positive_gamma",
        total_net_gex=500_000_000.0,
        spot_price=85_000.0,
        put_wall=80_000.0,
        call_wall=90_000.0,
        gamma_flip=82_000.0,
        gex_percentile=75.0,
    )
    return snap, regime


class TestGex:
    def test_returns_200(self, client):
        snap, regime = _mock_gex_snapshot()
        with (
            patch("src.gex.deribit_client.DeribitClient.get_spot_price", return_value=85_000.0),
            patch("src.gex.deribit_client.DeribitClient.fetch_all_options", return_value=[]),
            patch("src.gex.gex_calculator.GexCalculator.calculate_gex", return_value=snap),
            patch("src.gex.gex_calculator.GexCalculator.gex_to_dict", return_value={"spot_price": 85000}),
            patch("src.gex.gex_db.GexDB.get_latest_n", return_value=[]),
            patch("src.gex.gex_db.GexDB.insert_snapshot"),
            patch("src.gex.regime_detector.RegimeDetector.detect", return_value=regime),
        ):
            r = client.get("/api/gex")
        assert r.status_code == 200

    def test_response_keys(self, client):
        snap, regime = _mock_gex_snapshot()
        with (
            patch("src.gex.deribit_client.DeribitClient.get_spot_price", return_value=85_000.0),
            patch("src.gex.deribit_client.DeribitClient.fetch_all_options", return_value=[]),
            patch("src.gex.gex_calculator.GexCalculator.calculate_gex", return_value=snap),
            patch("src.gex.gex_calculator.GexCalculator.gex_to_dict", return_value={"spot_price": 85000}),
            patch("src.gex.gex_db.GexDB.get_latest_n", return_value=[]),
            patch("src.gex.gex_db.GexDB.insert_snapshot"),
            patch("src.gex.regime_detector.RegimeDetector.detect", return_value=regime),
        ):
            data = client.get("/api/gex").json()["data"]
        assert "snapshot" in data
        assert "regime" in data
        assert "strike_profile" in data
        assert "options_metrics" in data

    def test_gex_db_persist_failure_does_not_crash(self, client):
        """Se la persistenza GEX fallisce, l'endpoint deve comunque rispondere 200."""
        snap, regime = _mock_gex_snapshot()
        with (
            patch("src.gex.deribit_client.DeribitClient.get_spot_price", return_value=85_000.0),
            patch("src.gex.deribit_client.DeribitClient.fetch_all_options", return_value=[]),
            patch("src.gex.gex_calculator.GexCalculator.calculate_gex", return_value=snap),
            patch("src.gex.gex_calculator.GexCalculator.gex_to_dict", return_value={"spot_price": 85000}),
            patch("src.gex.gex_db.GexDB.get_latest_n", return_value=[]),
            patch("src.gex.gex_db.GexDB.insert_snapshot", side_effect=RuntimeError("DB locked")),
            patch("src.gex.regime_detector.RegimeDetector.detect", return_value=regime),
        ):
            r = client.get("/api/gex")
        assert r.status_code == 200

    def test_upstream_error_returns_500(self, client):
        with patch("src.gex.deribit_client.DeribitClient.get_spot_price", side_effect=ConnectionError("timeout")):
            r = client.get("/api/gex")
        assert r.status_code == 500

    def test_cache_hit_does_not_refetch(self, client):
        snap, regime = _mock_gex_snapshot()
        mock_fetch = MagicMock(return_value=[])
        with (
            patch("src.gex.deribit_client.DeribitClient.get_spot_price", return_value=85_000.0),
            patch("src.gex.deribit_client.DeribitClient.fetch_all_options", mock_fetch),
            patch("src.gex.gex_calculator.GexCalculator.calculate_gex", return_value=snap),
            patch("src.gex.gex_calculator.GexCalculator.gex_to_dict", return_value={"spot_price": 85000}),
            patch("src.gex.gex_db.GexDB.get_latest_n", return_value=[]),
            patch("src.gex.gex_db.GexDB.insert_snapshot"),
            patch("src.gex.regime_detector.RegimeDetector.detect", return_value=regime),
        ):
            client.get("/api/gex")
            client.get("/api/gex")  # secondo call — deve usare cache
        assert mock_fetch.call_count == 1  # fetch avvenuto una sola volta


# ──────────────────────────────────────────────────────────────────────────────
# /api/barriers
# ──────────────────────────────────────────────────────────────────────────────

class TestBarriers:
    def test_returns_200(self, client):
        with (
            patch("src.edgar.structured_notes_db.StructuredNotesDB.compute_btc_prices"),
            patch("src.edgar.structured_notes_db.StructuredNotesDB.get_active_barriers", return_value=[
                {"id": 1, "level_price_btc": 80000.0, "barrier_type": "knock_in", "status": "active"},
            ]),
            patch("src.flows.price_fetcher.PriceFetcher.get_all_prices", side_effect=RuntimeError("no data")),
            patch("src.gex.deribit_client.DeribitClient.get_spot_price", return_value=85_000.0),
        ):
            r = client.get("/api/barriers")
        assert r.status_code == 200

    def test_response_keys(self, client):
        with (
            patch("src.edgar.structured_notes_db.StructuredNotesDB.compute_btc_prices"),
            patch("src.edgar.structured_notes_db.StructuredNotesDB.get_active_barriers", return_value=[]),
            patch("src.flows.price_fetcher.PriceFetcher.get_all_prices", side_effect=RuntimeError("no data")),
            patch("src.gex.deribit_client.DeribitClient.get_spot_price", return_value=85_000.0),
        ):
            data = client.get("/api/barriers").json()["data"]
        assert "count" in data
        assert "barriers" in data
        assert "spot_price" in data

    def test_count_matches_barriers_length(self, client):
        barriers = [
            {"id": 1, "level_price_btc": 80000.0, "barrier_type": "knock_in"},
            {"id": 2, "level_price_btc": 75000.0, "barrier_type": "knock_out"},
        ]
        with (
            patch("src.edgar.structured_notes_db.StructuredNotesDB.compute_btc_prices"),
            patch("src.edgar.structured_notes_db.StructuredNotesDB.get_active_barriers", return_value=barriers),
            patch("src.flows.price_fetcher.PriceFetcher.get_all_prices", side_effect=RuntimeError),
            patch("src.gex.deribit_client.DeribitClient.get_spot_price", return_value=85_000.0),
        ):
            data = client.get("/api/barriers").json()["data"]
        assert data["count"] == 2
        assert len(data["barriers"]) == 2


# ──────────────────────────────────────────────────────────────────────────────
# Autenticazione
# ──────────────────────────────────────────────────────────────────────────────

class TestApiKeyAuth:
    def test_health_always_open(self, authed_client):
        """Health check non richiede autenticazione."""
        r = authed_client.get("/api/health")
        assert r.status_code == 200

    def test_protected_endpoint_without_key_returns_401(self, authed_client):
        r = authed_client.get("/api/gex")
        assert r.status_code == 401

    def test_protected_endpoint_with_wrong_key_returns_401(self, authed_client):
        r = authed_client.get("/api/gex", headers={"X-API-Key": "wrong-key"})
        assert r.status_code == 401

    def test_protected_endpoint_with_correct_key_passes(self, authed_client):
        snap, regime = _mock_gex_snapshot()
        with (
            patch("src.gex.deribit_client.DeribitClient.get_spot_price", return_value=85_000.0),
            patch("src.gex.deribit_client.DeribitClient.fetch_all_options", return_value=[]),
            patch("src.gex.gex_calculator.GexCalculator.calculate_gex", return_value=snap),
            patch("src.gex.gex_calculator.GexCalculator.gex_to_dict", return_value={"spot_price": 85000}),
            patch("src.gex.gex_db.GexDB.get_latest_n", return_value=[]),
            patch("src.gex.gex_db.GexDB.insert_snapshot"),
            patch("src.gex.regime_detector.RegimeDetector.detect", return_value=regime),
        ):
            r = authed_client.get("/api/gex", headers={"X-API-Key": "test-secret"})
        assert r.status_code == 200


# ──────────────────────────────────────────────────────────────────────────────
# Struttura generica della response
# ──────────────────────────────────────────────────────────────────────────────

class TestResponseEnvelope:
    def test_timestamp_is_utc_aware(self, client):
        ts = client.get("/api/health").json()["timestamp"]
        # datetime ISO con offset UTC (+00:00) o Z
        assert "+" in ts or ts.endswith("Z")

    def test_nan_values_serialized_as_null(self, client):
        """NaN float nei dati devono diventare null nel JSON."""
        import numpy as np
        from src.api.main import _ok
        resp = _ok({"value": float("nan"), "nested": {"x": np.nan}})
        body = resp.body
        import json
        parsed = json.loads(body)
        assert parsed["data"]["value"] is None
        assert parsed["data"]["nested"]["x"] is None
