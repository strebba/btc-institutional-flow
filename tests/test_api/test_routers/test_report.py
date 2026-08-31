"""Integration test per gli endpoint del Desk Note.

Le fonti a monte sono mockate: qui si verifica che il router le componga, che
regga quando una cade, e che la pagina HTML esca renderizzabile.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient


def _resp(data: dict) -> JSONResponse:
    return JSONResponse(content={"status": "ok", "timestamp": "2026-08-29T14:37:00Z", "data": data})


_GEX = {
    "snapshot": {
        "spot_price": 77_703.9, "total_net_gex": 171_930_539.8,
        "gamma_flip_price": 79_783.46, "put_wall": 75_000.0,
        "call_wall": 82_000.0, "max_pain": 72_000.0,
    },
    "regime": {"label": "positive_gamma", "gex_percentile": 76.47},
    "options_metrics": {"put_call_ratio": 0.4535},
    "strike_profile": [
        {"strike": 75_000.0, "net_gex_m": -11.11, "call_oi": 7089.2, "put_oi": 9166.8},
        {"strike": 82_000.0, "net_gex_m": 41.97, "call_oi": 14999.0, "put_oi": 499.0},
    ],
}
_BARRIERS = {
    "count": 2, "spot_price": 77_722.68, "meta": {"total_active": 293, "priced": 253},
    "barriers": [
        {"barrier_type": "knock_in", "level_price_btc": 76_897.43,
         "issuer": "JPMorgan", "notional_usd": None},
        {"barrier_type": "autocall", "level_price_btc": 95_000.0,
         "issuer": "Goldman Sachs", "notional_usd": None},
    ],
}
_FLOWS = {
    "summary": {
        "ibit": {"net_flow_usd_b": 63.36, "days_with_data": 660},
        "full_period_corr_ibit_btc_next1d": 0.1575,
        "by_ticker": {"GBTC": {"net_flow_usd_b": -27.6}},
    }
}
_SIGNALS = {
    "signal": "CAUTION", "score": 56.5,
    "inputs": {"ibit_flow_3d_usd_m": 445.0},
    "pillars": [
        {"name": "gex", "score": 56.6, "components": {"regime": 0.65, "flip": 0.37}},
        {"name": "macro", "score": 15.0,
         "components": {"funding": None, "oi_change": None, "long_short": None,
                        "put_call": 0.15, "liquidations": None}},
    ],
}
_FORECAST = {"open": 12, "total": 12}

#: Coerente con _SIGNALS, dove il pilastro macro ha tutti i fattori a None.
#: Va mockato come gli altri: senza, ogni test di questo file uscirebbe in rete
#: verso CoinGlass e CoinGecko, e il risultato cambierebbe col mercato.
_MACRO = {"source_status": "no_api_key"}

#: Il ripiego CoinGecko: due fattori su cinque, ma il funding e' quello che pesa.
_MACRO_COINGECKO = {
    "source_status": "partial_coingecko",
    "funding_source": "coingecko",
    "funding_rate_annualized_pct": 12.33,
    "futures_oi_usd": 66_238_576_882.0,
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient con cache pulita e stato del report su un DB usa-e-getta."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "runtime.db"))
    from src.api import cache

    cache.cache_clear()
    from src.api.main import app

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def fonti():
    """Mocka i sei endpoint a monte del Desk Note."""
    with (
        patch("src.api.routers.gex.get_gex", return_value=_resp(_GEX)),
        patch("src.api.routers.barriers.get_barriers", return_value=_resp(_BARRIERS)),
        patch("src.api.routers.flows.get_flows", return_value=_resp(_FLOWS)),
        patch("src.api.routers.signals.get_signals", return_value=_resp(_SIGNALS)),
        patch("src.api.routers.forecast.forecast_status", return_value=_resp(_FORECAST)),
        patch("src.api.routers.signals.get_macro", return_value=_resp(_MACRO)),
    ):
        yield


class TestCards:
    def test_ritorna_le_card(self, client, fonti):
        r = client.get("/api/report/cards")
        assert r.status_code == 200
        d = r.json()["data"]
        assert len(d["cards"]) >= 3
        assert d["cards"][0]["kind"] == "cover"

    def test_la_tape_porta_il_contesto(self, client, fonti):
        d = client.get("/api/report/cards").json()["data"]
        assert "BTC 77.704" in d["tape"]

    def test_segnala_il_pilastro_scoperto(self, client, fonti):
        d = client.get("/api/report/cards").json()["data"]
        assert any("macro" in w for w in d["warnings"])

    def test_una_fonte_giu_non_fa_fallire_l_edizione(self, client):
        """Se Deribit non risponde escono meno card, non un 500."""
        with (
            patch("src.api.routers.gex.get_gex", side_effect=RuntimeError("Deribit down")),
            patch("src.api.routers.barriers.get_barriers", return_value=_resp(_BARRIERS)),
            patch("src.api.routers.flows.get_flows", return_value=_resp(_FLOWS)),
            patch("src.api.routers.signals.get_signals", return_value=_resp(_SIGNALS)),
            patch("src.api.routers.forecast.forecast_status", return_value=_resp(_FORECAST)),
            patch("src.api.routers.signals.get_macro", return_value=_resp(_MACRO)),
        ):
            r = client.get("/api/report/cards")
        assert r.status_code == 200
        assert any(c["source_key"] == "barrier_nearest" for c in r.json()["data"]["cards"])

    def test_tutte_le_fonti_giu_da_un_edizione_vuota(self, client):
        with (
            patch("src.api.routers.gex.get_gex", side_effect=RuntimeError("giù")),
            patch("src.api.routers.barriers.get_barriers", side_effect=RuntimeError("giù")),
            patch("src.api.routers.flows.get_flows", side_effect=RuntimeError("giù")),
            patch("src.api.routers.signals.get_signals", side_effect=RuntimeError("giù")),
            patch("src.api.routers.forecast.forecast_status", side_effect=RuntimeError("giù")),
            patch("src.api.routers.signals.get_macro", side_effect=RuntimeError("giù")),
        ):
            r = client.get("/api/report/cards")
        assert r.status_code == 200
        assert r.json()["data"]["cards"] == []

    @staticmethod
    def _con_macro(client, macro: dict) -> dict:
        # l'edizione e' in cache: senza svuotarla la seconda chiamata
        # restituirebbe la prima, e il confronto sarebbe fra due copie uguali
        from src.api import cache

        cache.cache_clear()
        with (
            patch("src.api.routers.gex.get_gex", return_value=_resp(_GEX)),
            patch("src.api.routers.barriers.get_barriers", return_value=_resp(_BARRIERS)),
            patch("src.api.routers.flows.get_flows", return_value=_resp(_FLOWS)),
            patch("src.api.routers.signals.get_signals", return_value=_resp(_SIGNALS)),
            patch("src.api.routers.forecast.forecast_status", return_value=_resp(_FORECAST)),
            patch("src.api.routers.signals.get_macro", return_value=_resp(macro)),
        ):
            return client.get("/api/report/cards").json()["data"]

    def test_il_ripiego_non_dice_piu_che_il_pilastro_e_spento(self, client):
        d = self._con_macro(client, _MACRO_COINGECKO)
        assert not any("spento" in w for w in d["warnings"])
        assert any("CoinGecko" in w for w in d["warnings"])

    def test_il_funding_entra_fra_i_candidati(self, client):
        """Con 12,3% il funding e' un fatto vero, ma non per forza uno dei cinque:
        in questo tape ci sono cinque fatti piu' salienti, e la selezione lo dice."""
        senza = self._con_macro(client, _MACRO)
        con = self._con_macro(client, _MACRO_COINGECKO)
        assert con["facts_considered"] == senza["facts_considered"] + 1

    def test_un_funding_rovente_si_prende_una_card(self, client):
        """A 60% annuo il costo del carry diventa la notizia, e vince uno slot."""
        d = self._con_macro(
            client, {**_MACRO_COINGECKO, "funding_rate_annualized_pct": 60.0}
        )
        assert any(c["source_key"] == "funding_cost" for c in d["cards"])

    def test_il_payload_e_json_serializzabile(self, client, fonti):
        json.dumps(client.get("/api/report/cards").json())


class TestPagina:
    def test_serve_html(self, client, fonti):
        r = client.get("/report")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")
        assert "<!doctype html>" in r.text.lower()

    def test_contiene_le_card_e_i_colori_wagmi(self, client, fonti):
        html = client.get("/report").text
        assert "#00FF9D" in html
        assert "WAGMI LAB" in html
        assert html.count('class="card"') + html.count('class="card cover"') >= 3

    def test_modalita_export_toglie_la_riduzione(self, client, fonti):
        normale = client.get("/report").text
        export = client.get("/report?export=true").text
        assert "transform:scale(" in normale
        assert "transform:none" in export

    def test_l_intestazione_avvisa_dei_dati_incompleti(self, client, fonti):
        html = client.get("/report").text
        assert "non vanno pubblicate" in html

    def test_export_non_mostra_l_intestazione(self, client, fonti):
        """Nell'export ci sono solo le card: l'avviso non deve finire in un PNG."""
        assert "masthead" not in client.get("/report?export=true").text


class TestEventi:
    def test_primo_giro_nessun_evento(self, client, fonti):
        d = client.get("/api/report/events").json()["data"]
        assert d["events"] == []
        assert d["should_publish"] is False

    def test_commit_poi_confronto(self, client, fonti):
        assert client.post("/api/report/events/commit").json()["data"]["committed"] is True
        d = client.get("/api/report/events").json()["data"]
        assert d["events"] == []  # niente è cambiato dal commit

    def test_rileva_il_cambiamento_dopo_il_commit(self, client, fonti):
        client.post("/api/report/events/commit")
        from src.api import cache

        cache.cache_clear()
        mosso = {
            **_GEX,
            "snapshot": {**_GEX["snapshot"], "spot_price": 74_000.0},
            "regime": {"label": "negative_gamma", "gex_percentile": 12.0},
        }
        with (
            patch("src.api.routers.gex.get_gex", return_value=_resp(mosso)),
            patch("src.api.routers.barriers.get_barriers", return_value=_resp(_BARRIERS)),
            patch("src.api.routers.flows.get_flows", return_value=_resp(_FLOWS)),
            patch("src.api.routers.signals.get_signals", return_value=_resp(_SIGNALS)),
            patch("src.api.routers.forecast.forecast_status", return_value=_resp(_FORECAST)),
        ):
            d = client.get("/api/report/events").json()["data"]

        assert d["should_publish"] is True
        chiavi = {e["key"] for e in d["events"]}
        assert "gamma_regime_flip" in chiavi
        assert "barrier_breached" in chiavi
