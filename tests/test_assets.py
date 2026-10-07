"""Registro degli asset: un solo punto di verità per le cablature BTC/ETH."""
from __future__ import annotations

import pytest

from src.assets import ASSETS, get_asset


def test_btc_conserva_i_nomi_legacy_delle_colonne():
    btc = get_asset("BTC")
    assert btc.price_col == "btc_close"
    assert btc.return_col == "btc_return"
    assert btc.vol_col == "btc_vol_7d"
    assert btc.lead_flow_col == "ibit_flow"
    assert btc.lead_etf == "IBIT"
    assert btc.spot_ticker == "BTC-USD"
    assert btc.deribit_index == "btc_usd"


def test_eth_usa_colonne_con_prefisso_e_etha_come_lead_etf():
    eth = get_asset("ETH")
    assert eth.price_col == "eth_close"
    assert eth.return_col == "eth_return"
    assert eth.vol_col == "eth_vol_7d"
    assert eth.lead_flow_col == "etha_flow"
    assert eth.lead_etf == "ETHA"
    assert eth.spot_ticker == "ETH-USD"
    assert eth.deribit_currency == "ETH"
    assert eth.deribit_index == "eth_usd"
    assert "ethereum" in eth.farside_url


def test_eth_espone_solo_le_feature_dati():
    assert get_asset("ETH").features == frozenset({"gex", "flows", "macro"})
    assert {"analytics", "barriers", "edgar"} <= get_asset("BTC").features


def test_get_asset_normalizza_le_maiuscole():
    assert get_asset("eth") is ASSETS["ETH"]
    assert get_asset(" btc ") is ASSETS["BTC"]


def test_get_asset_rifiuta_chiavi_sconosciute():
    with pytest.raises(ValueError):
        get_asset("SOL")
