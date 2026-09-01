"""Test per il client CoinGecko.

Serve a coprire i due fattori macro che CoinGlass darebbe ma che senza chiave
mancano: funding rate e open interest. Non copre long/short né liquidazioni —
CoinGecko non li espone.

La conversione del funding è testata in ``test_funding.py``: è condivisa con
CoinGlass e sbagliarla sposta il numero di due ordini di grandezza.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.flows.coingecko_client import (
    CoinGeckoClient,
    CoinGeckoError,
    annualize_funding_pct,
)


def _contratto(market: str, oi: float, funding: float, *, index="BTC", tipo="perpetual") -> dict:
    return {
        "market": market, "symbol": "BTCUSDT", "index_id": index,
        "contract_type": tipo, "open_interest": oi, "funding_rate": funding,
        "price": "78000", "volume_24h": 1e9,
    }


@pytest.fixture
def client() -> CoinGeckoClient:
    return CoinGeckoClient()


def _con_risposta(payload):
    """Mocka la GET del client, non requests in generale."""
    resp = MagicMock()
    resp.json.return_value = payload
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    return patch.object(requests.Session, "get", return_value=resp)


class TestConversioneFunding:
    """Entrambe le fonti danno punti percentuali per 8 ore.

    La convenzione unica vive in :mod:`src.flows.funding`, con i test che la
    verificano sui valori reali delle due API. Qui si controlla solo che il
    client CoinGecko la usi invece di rifarsi la propria.
    """

    def test_lo_zero_virgola_zero_uno_percento_fa_dodici_percento(self):
        assert annualize_funding_pct(0.01) == pytest.approx(10.95, abs=0.01)

    def test_il_valore_reale_osservato(self):
        """0,0112% per 8h è quanto misurato sulla chain vera."""
        assert annualize_funding_pct(0.0112) == pytest.approx(12.26, abs=0.01)

    def test_non_applica_il_fattore_cento_di_troppo(self):
        assert annualize_funding_pct(0.01) < 100, "conversione fuori di due ordini di grandezza"

    def test_funding_negativo_resta_negativo(self):
        assert annualize_funding_pct(-0.005) < 0

    def test_zero_resta_zero(self):
        assert annualize_funding_pct(0.0) == 0.0


class TestFiltroContratti:
    def test_tiene_solo_i_perpetui_btc(self, client):
        payload = [
            _contratto("Binance", 8e9, 0.01),
            _contratto("Binance ETH", 5e9, 0.009, index="ETH"),
            _contratto("Binance futures", 1e9, 0.01, tipo="futures"),
        ]
        with _con_risposta(payload):
            righe = client.fetch_btc_derivatives()
        assert len(righe) == 1
        assert righe[0]["market"] == "Binance"

    def test_scarta_i_contratti_senza_oi_o_funding(self, client):
        payload = [
            _contratto("Buono", 8e9, 0.01),
            _contratto("SenzaOI", 0, 0.01),
            {**_contratto("SenzaFunding", 5e9, 0.0), "funding_rate": None},
        ]
        with _con_risposta(payload):
            assert len(client.fetch_btc_derivatives()) == 1

    def test_payload_malformato_non_esplode(self, client):
        with _con_risposta({"errore": "qualcosa"}):
            assert client.fetch_btc_derivatives() == []

    def test_valori_non_numerici_vengono_saltati(self, client):
        payload = [
            _contratto("Buono", 8e9, 0.01),
            {**_contratto("Rotto", 1e9, 0.01), "open_interest": "non-un-numero"},
        ]
        with _con_risposta(payload):
            assert len(client.fetch_btc_derivatives()) == 1


class TestFundingPesatoPerOi:
    def test_pesa_per_open_interest(self, client):
        """Un exchange con OI dieci volte maggiore deve contare dieci volte di più."""
        payload = [_contratto("Grande", 9e9, 0.02), _contratto("Piccolo", 1e9, 0.00)]
        with _con_risposta(payload):
            funding, oi, n = client.fetch_funding_and_oi()
        atteso = annualize_funding_pct(0.02 * 0.9)
        assert funding == pytest.approx(atteso, rel=1e-9)
        assert oi == pytest.approx(1e10)
        assert n == 2

    def test_media_semplice_darebbe_un_numero_diverso(self, client):
        """Verifica che la ponderazione sia davvero applicata, non equivalente."""
        payload = [_contratto("Grande", 9e9, 0.02), _contratto("Piccolo", 1e9, 0.00)]
        with _con_risposta(payload):
            funding, _, _ = client.fetch_funding_and_oi()
        media_semplice = annualize_funding_pct(0.01)
        assert funding != pytest.approx(media_semplice, rel=1e-6)

    def test_senza_contratti_torna_none(self, client):
        with _con_risposta([]):
            assert client.fetch_funding_and_oi() == (None, None, 0)

    def test_oi_totale_nullo_non_divide_per_zero(self, client):
        with _con_risposta([_contratto("Vuoto", 0, 0.01)]):
            assert client.fetch_funding_and_oi() == (None, None, 0)


class TestErrori:
    def test_errore_di_rete_solleva_coingecko_error(self, client):
        with patch.object(requests.Session, "get", side_effect=requests.Timeout("timeout")):
            with pytest.raises(CoinGeckoError):
                client.fetch_btc_derivatives()

    def test_fetch_funding_and_oi_non_propaga_l_errore(self, client):
        """Il chiamante vuole (None, None, 0), non un'eccezione: e' un fallback."""
        with patch.object(requests.Session, "get", side_effect=requests.Timeout("timeout")):
            assert client.fetch_funding_and_oi() == (None, None, 0)


class TestChiave:
    def test_funziona_senza_chiave(self, client):
        """L'endpoint pubblico non la richiede: has_api_key e' informativo."""
        assert client.has_api_key in (True, False)

    def test_la_chiave_finisce_nell_header_demo(self, monkeypatch):
        monkeypatch.setenv("COINGECKO_API_KEY", "abc123")
        c = CoinGeckoClient()
        assert c.has_api_key is True
        assert c._session.headers.get("x-cg-demo-api-key") == "abc123"

    def test_senza_chiave_nessun_header(self, monkeypatch):
        monkeypatch.delenv("COINGECKO_API_KEY", raising=False)
        c = CoinGeckoClient()
        assert "x-cg-demo-api-key" not in c._session.headers
