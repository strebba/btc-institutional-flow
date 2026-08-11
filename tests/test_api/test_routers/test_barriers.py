"""Test per router barriers.py — endpoint /api/barriers, /api/notes, /api/notes/by-url."""

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


class TestBarriers:
    def test_barriers_empty_returns_zero(self, client):
        with patch("src.edgar.structured_notes_db.StructuredNotesDB") as mock_db, \
             patch("src.edgar.structured_notes_db.refresh_barrier_btc_prices"), \
             patch("src.gex.deribit_client.DeribitClient") as mock_deribit, \
             patch("src.edgar.barrier_utils.compute_confluence", return_value=[]), \
             patch("src.edgar.barrier_utils.detect_clusters", return_value=[], create=True), \
             patch("src.api.routers.gex._get_gex_data", return_value={}):
            mock_db.return_value.get_active_barriers.return_value = []
            mock_deribit.return_value.get_spot_price.return_value = 85000.0

            r = client.get("/api/barriers")
            assert r.status_code == 200
            assert r.json()["data"]["count"] == 0


class TestNotes:
    def test_get_notes_returns_200(self, client):
        with patch("src.edgar.structured_notes_db.StructuredNotesDB") as mock_db:
            mock_db.return_value.get_all_notes.return_value = []
            r = client.get("/api/notes")
            assert r.status_code == 200
            assert r.json()["data"]["count"] == 0

    def test_get_note_by_url_not_found(self, client):
        with patch("src.edgar.structured_notes_db.StructuredNotesDB") as mock_db:
            mock_db.return_value.get_note_by_url.return_value = None
            r = client.get("/api/notes/by-url?url=https://missing.com")
            assert r.status_code == 404
