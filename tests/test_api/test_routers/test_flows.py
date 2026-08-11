"""Test per router flows.py — endpoint /api/flows (route existence + error handling)."""

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


class TestFlows:
    def test_upstream_error_returns_500(self, client):
        with patch("src.flows.scraper.FarsideScraper") as mock_farside:
            mock_farside.return_value.fetch.side_effect = RuntimeError("no data")
            r = client.get("/api/flows")
            assert r.status_code == 500
