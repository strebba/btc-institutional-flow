"""Test per router signals.py — endpoint /api/signals, /api/pillars/series."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    from src.api import cache
    cache.cache_clear()
    from src.api.main import app
    return TestClient(app, raise_server_exceptions=False)


class TestPillarsSeries:
    def test_invalid_pillar_returns_400(self, client):
        r = client.get("/api/pillars/series?pillar=invalid_pillar")
        assert r.status_code == 400
