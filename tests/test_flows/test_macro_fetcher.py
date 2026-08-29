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
    STATUS_UNAVAILABLE,
    MacroData,
    fetch_macro_data,
)


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
