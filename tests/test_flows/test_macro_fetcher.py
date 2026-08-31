"""Test per il fetch unificato dei dati macro.

Il punto: "non lo sappiamo" e "il mercato e' piatto" devono essere distinguibili
a valle. Con tutti i campi a None erano la stessa cosa, e il pilastro macro
finiva per pubblicare un giudizio che nessun dato sosteneva.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.flows.macro_fetcher import (
    STATUS_NO_API_KEY,
    STATUS_OK,
    STATUS_PARTIAL_COINGECKO,
    STATUS_UNAVAILABLE,
    MacroData,
    fetch_macro_data,
)


@pytest.fixture(autouse=True)
def _niente_rete(monkeypatch):
    """Nessun test deve uscire in rete.

    Il ripiego CoinGecko si costruisce un client da solo quando non gliene passi
    uno: senza questo, i test scritti prima del ripiego farebbero una GET vera da
    8 MB per scoprire che CoinGlass era muto.
    """
    muto = MagicMock()
    muto.fetch_funding_and_oi.return_value = (None, None, 0)
    monkeypatch.setattr("src.flows.macro_fetcher.CoinGeckoClient", lambda *a, **k: muto)


def _client(*, has_key: bool = True, funding=None, oi=None, ls=None, liq=None) -> MagicMock:
    cg = MagicMock()
    cg.has_api_key = has_key
    cg.fetch_funding_rate_history.return_value = (
        funding if funding is not None else pd.Series(dtype=float)
    )
    cg.fetch_aggregated_oi_history.return_value = (
        oi if oi is not None else pd.Series(dtype=float)
    )
    cg.fetch_long_short_ratio.return_value = ls if ls is not None else pd.Series(dtype=float)
    cg.fetch_liquidations.return_value = (
        liq if liq is not None else pd.DataFrame(columns=["long_usd", "short_usd"])
    )
    return cg


class TestSenzaChiave:
    def test_lo_stato_dice_che_manca_la_chiave(self):
        out = fetch_macro_data(cg_client=_client(has_key=False))
        assert out.source_status == STATUS_NO_API_KEY

    def test_non_chiama_l_api_a_vuoto(self):
        """Senza chiave ogni chiamata fallirebbe: cinque richieste e cinque warning
        per niente, a ogni giro di /api/signals."""
        cg = _client(has_key=False)
        fetch_macro_data(cg_client=cg)
        cg.fetch_funding_rate_history.assert_not_called()
        cg.fetch_aggregated_oi_history.assert_not_called()
        cg.fetch_long_short_ratio.assert_not_called()
        cg.fetch_liquidations.assert_not_called()

    def test_i_valori_restano_none(self):
        out = fetch_macro_data(cg_client=_client(has_key=False))
        assert out.funding_rate_annualized_pct is None
        assert out.long_short_ratio is None


class TestConChiave:
    def test_dati_presenti_danno_ok(self):
        cg = _client(funding=pd.Series([0.0001]), ls=pd.Series([1.05]))
        out = fetch_macro_data(cg_client=cg)
        assert out.source_status == STATUS_OK
        assert out.funding_rate_annualized_pct is not None

    def test_chiave_presente_ma_tutto_vuoto_e_unavailable(self):
        """CoinGlass raggiungibile ma senza dati: diverso da chiave assente."""
        out = fetch_macro_data(cg_client=_client())
        assert out.source_status == STATUS_UNAVAILABLE

    def test_anche_un_solo_fattore_basta_per_ok(self):
        out = fetch_macro_data(cg_client=_client(ls=pd.Series([0.9])))
        assert out.source_status == STATUS_OK


class TestCache:
    def test_la_cache_evita_il_refetch(self):
        cg = _client(funding=pd.Series([0.0001]))
        cached = {"funding_rate_annualized_pct": 12.0, "long_short_ratio": 1.1}
        out = fetch_macro_data(cg_client=cg, cache_data=cached)
        assert out.funding_rate_annualized_pct == 12.0
        cg.fetch_funding_rate_history.assert_not_called()

    def test_valori_da_cache_contano_come_ok(self):
        out = fetch_macro_data(
            cg_client=_client(), cache_data={"funding_rate_annualized_pct": 12.0}
        )
        assert out.source_status == STATUS_OK


class TestSerializzazione:
    def test_to_dict_include_lo_stato(self):
        d = fetch_macro_data(cg_client=_client(has_key=False)).to_dict()
        assert d["source_status"] == STATUS_NO_API_KEY

    def test_round_trip(self):
        originale = MacroData(funding_rate_annualized_pct=5.0, source_status=STATUS_OK)
        assert MacroData.from_dict(originale.to_dict()) == originale

    def test_from_dict_su_dizionario_vecchio_non_esplode(self):
        """Cache serializzata prima di questo campo: deve restare leggibile."""
        vecchio = {"funding_rate_annualized_pct": 5.0, "long_short_ratio": 1.0}
        assert MacroData.from_dict(vecchio).funding_rate_annualized_pct == 5.0


@pytest.mark.parametrize("stato", [STATUS_OK, STATUS_NO_API_KEY, STATUS_UNAVAILABLE])
def test_gli_stati_sono_stringhe_distinte(stato):
    assert isinstance(stato, str)
    assert len({STATUS_OK, STATUS_NO_API_KEY, STATUS_UNAVAILABLE}) == 3


# ─── CoinGecko come fonte di ripiego ────────────────────────────────────────


def _gecko(*, funding=None, oi=None, n=0) -> MagicMock:
    """CoinGecko finto. Di default non sa niente: e' il caso peggiore."""
    g = MagicMock()
    g.fetch_funding_and_oi.return_value = (funding, oi, n)
    return g


def _db(*, oi_change=None) -> MagicMock:
    db = MagicMock()
    db.get_oi_change_pct.return_value = oi_change
    return db


class TestRipiegoCoinGecko:
    """Senza chiave CoinGlass il pilastro macro non deve restare spento.

    CoinGecko copre due fattori su cinque — funding e open interest — e questo
    e' molto piu' di zero: il funding da solo pesa 0,30 del pilastro.
    """

    def test_senza_chiave_coinglass_usa_coingecko(self):
        out = fetch_macro_data(
            cg_client=_client(has_key=False),
            gecko_client=_gecko(funding=12.3, oi=6.5e10, n=138),
        )
        assert out.funding_rate_annualized_pct == 12.3
        assert out.oi_usd == 6.5e10

    def test_lo_stato_dice_che_la_fonte_e_di_ripiego(self):
        """Non e' `ok`: long/short e liquidazioni mancano ancora."""
        out = fetch_macro_data(
            cg_client=_client(has_key=False),
            gecko_client=_gecko(funding=12.3, oi=6.5e10, n=138),
        )
        assert out.source_status == STATUS_PARTIAL_COINGECKO

    def test_dice_da_dove_viene_il_funding(self):
        out = fetch_macro_data(
            cg_client=_client(has_key=False),
            gecko_client=_gecko(funding=12.3, oi=6.5e10, n=138),
        )
        assert out.funding_source == "coingecko"

    def test_i_fattori_che_coingecko_non_ha_restano_none(self):
        """Long/short e liquidazioni non esistono su CoinGecko: mai inventarli."""
        out = fetch_macro_data(
            cg_client=_client(has_key=False),
            gecko_client=_gecko(funding=12.3, oi=6.5e10, n=138),
        )
        assert out.long_short_ratio is None
        assert out.liquidations_long_24h_usd is None

    def test_anche_coingecko_muto_lascia_no_api_key(self):
        out = fetch_macro_data(cg_client=_client(has_key=False), gecko_client=_gecko())
        assert out.source_status == STATUS_NO_API_KEY

    def test_la_variazione_oi_viene_dallo_storico_accumulato(self):
        """CoinGecko da' il livello, non la serie: i 7 giorni li mette il DB."""
        out = fetch_macro_data(
            cg_client=_client(has_key=False),
            gecko_client=_gecko(funding=12.3, oi=6.5e10, n=138),
            notes_db=_db(oi_change=4.2),
        )
        assert out.oi_change_7d_pct == 4.2

    def test_senza_storico_sufficiente_la_variazione_resta_none(self):
        out = fetch_macro_data(
            cg_client=_client(has_key=False),
            gecko_client=_gecko(funding=12.3, oi=6.5e10, n=138),
            notes_db=_db(oi_change=None),
        )
        assert out.oi_change_7d_pct is None

    def test_un_db_che_esplode_non_fa_fallire_il_fetch(self):
        db = MagicMock()
        db.get_oi_change_pct.side_effect = RuntimeError("db lock")
        out = fetch_macro_data(
            cg_client=_client(has_key=False),
            gecko_client=_gecko(funding=12.3, oi=6.5e10, n=138),
            notes_db=db,
        )
        assert out.funding_rate_annualized_pct == 12.3
        assert out.oi_change_7d_pct is None


class TestCoinGlassRestaPreferito:
    def test_con_la_chiave_non_interroga_coingecko(self):
        """CoinGlass da' tutti e cinque i fattori: il ripiego non serve."""
        g = _gecko(funding=99.0, oi=1e10, n=1)
        cg = _client(funding=pd.Series([0.0001]), ls=pd.Series([1.05]))
        out = fetch_macro_data(cg_client=cg, gecko_client=g)
        g.fetch_funding_and_oi.assert_not_called()
        assert out.source_status == STATUS_OK
        assert out.funding_source == "coinglass"

    def test_coinglass_muto_scende_su_coingecko(self):
        """Chiave presente ma API a vuoto: meglio due fattori che zero."""
        out = fetch_macro_data(
            cg_client=_client(), gecko_client=_gecko(funding=12.3, oi=6.5e10, n=138)
        )
        assert out.funding_rate_annualized_pct == 12.3
        assert out.source_status == STATUS_PARTIAL_COINGECKO

    def test_coinglass_parziale_non_viene_sovrascritto(self):
        """Il long/short di CoinGlass sopravvive, il funding lo riempie CoinGecko."""
        cg = _client(ls=pd.Series([1.4]))
        out = fetch_macro_data(cg_client=cg, gecko_client=_gecko(funding=12.3, oi=6.5e10, n=1))
        assert out.long_short_ratio == 1.4
        assert out.funding_rate_annualized_pct == 12.3
        assert out.source_status == STATUS_OK, "un fattore CoinGlass c'e': non e' ripiego puro"


class TestSerializzazioneNuoviCampi:
    def test_to_dict_espone_oi_e_fonte(self):
        out = fetch_macro_data(
            cg_client=_client(has_key=False),
            gecko_client=_gecko(funding=12.3, oi=6.5e10, n=138),
        )
        d = out.to_dict()
        assert d["oi_usd"] == 6.5e10
        assert d["funding_source"] == "coingecko"

    def test_round_trip_con_i_nuovi_campi(self):
        originale = MacroData(
            funding_rate_annualized_pct=12.3, oi_usd=6.5e10,
            funding_source="coingecko", source_status=STATUS_PARTIAL_COINGECKO,
        )
        assert MacroData.from_dict(originale.to_dict()) == originale

    def test_cache_vecchia_senza_i_campi_nuovi_resta_leggibile(self):
        vecchio = {"funding_rate_annualized_pct": 5.0, "source_status": STATUS_OK}
        m = MacroData.from_dict(vecchio)
        assert m.oi_usd is None and m.funding_source is None

    def test_la_cache_conserva_lo_stato_di_ripiego(self):
        """Rileggere dalla cache non deve promuovere `partial` a `ok`."""
        cached = {
            "funding_rate_annualized_pct": 12.3,
            "source_status": STATUS_PARTIAL_COINGECKO,
            "funding_source": "coingecko",
        }
        out = fetch_macro_data(
            cg_client=_client(has_key=False), gecko_client=_gecko(), cache_data=cached
        )
        assert out.source_status == STATUS_PARTIAL_COINGECKO


def test_i_quattro_stati_sono_distinti():
    assert len({STATUS_OK, STATUS_NO_API_KEY, STATUS_UNAVAILABLE, STATUS_PARTIAL_COINGECKO}) == 4
