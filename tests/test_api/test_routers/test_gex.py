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
