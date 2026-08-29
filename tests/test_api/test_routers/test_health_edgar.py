"""Test per /api/health/edgar.

La salute della pipeline si misura sull'ultimo refresh riuscito, non sull'ultima
nota scritta: il workflow gira due volte a settimana, e se in quella finestra
nessuna banca deposita note IBIT il DB resta legittimamente fermo.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    from src.api import cache

    cache.cache_clear()
    from src.api.main import app

    return TestClient(app, raise_server_exceptions=False)


def _iso(giorni_fa: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=giorni_fa)).isoformat()


def _stats(**kw) -> dict:
    base = {
        "last_refresh_at": _iso(1),
        "last_refresh_ok": True,
        "last_refresh_filings_seen": 0,
        "last_note_written_at": _iso(12),
        "last_update": _iso(12),
        "total_notes": 680,
        "total_barriers": 715,
        "active_barriers": 715,
    }
    base.update(kw)
    return base


def _con_stats(stats: dict):
    return patch("src.edgar.structured_notes_db.StructuredNotesDB.get_edgar_stats",
                 return_value=stats)


class TestSaluteSulRefresh:
    def test_refresh_recente_e_note_vecchie_resta_sano(self, client):
        """Il caso reale del 29 agosto: refresh riuscito, nessuna nota da 12 giorni.

        Prima questo dava stale_days=12 e sarebbe diventato unhealthy il 31,
        con la pipeline perfettamente funzionante.
        """
        with _con_stats(_stats()):
            d = client.get("/api/health/edgar").json()["data"]
        assert d["healthy"] is True

    def test_refresh_assente_da_troppo_e_malato(self, client):
        with _con_stats(_stats(last_refresh_at=_iso(15))):
            d = client.get("/api/health/edgar").json()["data"]
        assert d["healthy"] is False

    def test_ultimo_refresh_fallito_e_malato(self, client):
        with _con_stats(_stats(last_refresh_ok=False)):
            d = client.get("/api/health/edgar").json()["data"]
        assert d["healthy"] is False

    def test_senza_refresh_registrati_ripiega_sulle_note(self, client):
        """DB anteriore alla tabella refresh_runs: non deve esplodere."""
        with _con_stats(_stats(last_refresh_at=None, last_refresh_ok=None,
                               last_note_written_at=_iso(2), last_update=_iso(2))):
            r = client.get("/api/health/edgar")
        assert r.status_code == 200
        assert r.json()["data"]["healthy"] is True

    def test_senza_refresh_e_con_note_vecchie_e_malato(self, client):
        with _con_stats(_stats(last_refresh_at=None, last_refresh_ok=None,
                               last_note_written_at=_iso(40), last_update=_iso(40))):
            assert client.get("/api/health/edgar").json()["data"]["healthy"] is False


class TestPayload:
    def test_distingue_le_due_eta(self, client):
        with _con_stats(_stats()):
            edgar = client.get("/api/health/edgar").json()["data"]["edgar"]
        assert edgar["refresh_age_days"] is not None
        assert edgar["notes_age_days"] is not None
        assert edgar["refresh_age_days"] < edgar["notes_age_days"]

    def test_spiega_a_parole_cosa_sta_succedendo(self, client):
        """Un solo stale_days ambiguo non basta a chi legge l'alert."""
        with _con_stats(_stats()):
            edgar = client.get("/api/health/edgar").json()["data"]["edgar"]
        assert "reason" in edgar
        assert edgar["reason"]

    def test_i_conteggi_restano(self, client):
        with _con_stats(_stats()):
            edgar = client.get("/api/health/edgar").json()["data"]["edgar"]
        assert edgar["total_notes"] == 680
        assert edgar["active_barriers"] == 715


class TestErrori:
    def test_un_errore_del_db_da_503_non_500(self, client):
        """http_error(msg, code=503): gli argomenti erano invertiti."""
        with patch("src.edgar.structured_notes_db.StructuredNotesDB.get_edgar_stats",
                   side_effect=RuntimeError("db corrotto")):
            r = client.get("/api/health/edgar")
        assert r.status_code == 503
