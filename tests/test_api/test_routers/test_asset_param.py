"""Parametro ?asset= su /api/gex, /api/flows, /api/macro.

Il contratto BTC (nessun parametro) non cambia: ETH è additivo e ha cache proprie.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.flows.models import EtfFlowData


@pytest.fixture()
def client():
    from src.api import cache
    cache.cache_clear()
    from src.api.main import app
    return TestClient(app, raise_server_exceptions=False)


def _option(name: str, strike: float, tipo: str, spot: float) -> dict:
    return {
        "instrument_name": name, "strike": strike, "option_type": tipo, "gamma": 0.001,
        "open_interest": 100.0, "expiration_timestamp": 0, "mark_price": 10.0,
        "mark_iv": 50.0, "underlying_price": spot, "best_bid": 9.0, "best_ask": 11.0,
        "delta": 0.5, "vega": 1.0,
    }


def _regime(spot: float):
    from src.gex.models import GammaRegime

    return GammaRegime(
        timestamp=datetime.now(timezone.utc), regime="positive_gamma", total_net_gex=1.0,
        spot_price=spot, put_wall=None, call_wall=None, gamma_flip=None, alerts=[],
        gex_percentile=None,
    )


class TestGexAsset:
    def test_eth_usa_deribit_eth_e_persiste_come_eth(self, client):
        with patch("src.gex.deribit_client.DeribitClient") as deribit, \
             patch("src.gex.gex_db.GexDB") as gex_db, \
             patch("src.gex.regime_detector.RegimeDetector") as detector, \
             patch("src.flows.coinglass_client.CoinGlassClient") as cg:
            deribit.return_value.get_spot_price.return_value = 2500.0
            deribit.return_value.fetch_all_options.return_value = [
                _option("ETH-T-2600-C", 2600.0, "call", 2500.0)
            ]
            detector.return_value.detect.return_value = _regime(2500.0)
            cg.return_value.fetch_options_info.return_value = []
            cg.return_value.fetch_options_max_pain.return_value = []

            r = client.get("/api/gex?asset=eth")

        assert r.status_code == 200
        assert r.json()["data"]["asset"] == "ETH"
        deribit.return_value.get_spot_price.assert_called_once_with("ETH")
        deribit.return_value.fetch_all_options.assert_called_once_with("ETH")
        assert detector.call_args.kwargs["asset"] == "ETH"
        assert gex_db.return_value.insert_snapshot.call_args.kwargs["asset"] == "ETH"
        cg.return_value.fetch_options_info.assert_called_once_with("ETH")
        assert {c.args[0] for c in cg.return_value.fetch_options_max_pain.call_args_list} == {"ETH"}

    def test_cache_eth_e_btc_separate(self, client):
        with patch("src.gex.deribit_client.DeribitClient") as deribit, \
             patch("src.gex.gex_db.GexDB"), \
             patch("src.gex.regime_detector.RegimeDetector") as detector, \
             patch("src.flows.coinglass_client.CoinGlassClient") as cg:
            deribit.return_value.get_spot_price.side_effect = lambda a="BTC": {"BTC": 80_000.0, "ETH": 2500.0}[a]
            deribit.return_value.fetch_all_options.side_effect = lambda a="BTC": [
                _option(f"{a}-T-C", 2600.0 if a == "ETH" else 82_000.0, "call",
                        2500.0 if a == "ETH" else 80_000.0)
            ]
            detector.return_value.detect.return_value = _regime(1.0)
            cg.return_value.fetch_options_info.return_value = []
            cg.return_value.fetch_options_max_pain.return_value = []

            eth = client.get("/api/gex?asset=eth").json()["data"]
            btc = client.get("/api/gex").json()["data"]

        assert eth["snapshot"]["spot_price"] == 2500.0
        assert btc["snapshot"]["spot_price"] == 80_000.0
        assert btc["asset"] == "BTC"

    def test_asset_sconosciuto_e_422(self, client):
        assert client.get("/api/gex?asset=sol").status_code == 422


def _eth_context() -> dict:
    idx = pd.date_range("2026-01-01", periods=40, freq="D")
    rng = np.random.default_rng(1)
    merged = pd.DataFrame({
        "etha_flow": rng.normal(5e7, 1e8, 40), "total_flow": rng.normal(8e7, 1e8, 40),
        "eth_close": 2500 + np.cumsum(rng.normal(0, 20, 40)), "eth_return": rng.normal(0, 0.03, 40),
        "eth_vol_7d": rng.uniform(0.4, 0.8, 40), "etha_close": rng.uniform(20, 25, 40),
    }, index=idx)
    merged["etha_eth_ratio"] = merged["etha_close"] / merged["eth_close"]
    merged["eth_return_next1d"] = merged["eth_return"].shift(-1)
    raw = [EtfFlowData(date=d.date(), ticker=t, flow_usd=1e6, source="coinglass")
           for d in idx for t in ("ETHA", "FETH")]
    return {"raw": raw, "agg": [], "prices": pd.DataFrame(), "merged_df": merged}


class TestFlowsAsset:
    def test_eth_usa_le_colonne_eth(self, client):
        with patch("src.api.data_pipeline.get_flow_context", return_value=_eth_context()) as ctx:
            r = client.get("/api/flows?asset=eth")

        assert r.status_code == 200, r.text
        assert ctx.call_args.kwargs["asset"] == "ETH"
        data = r.json()["data"]
        assert data["asset"] == "ETH"
        last = data["history"][-1]
        assert {"etha_flow_usd", "eth_close", "eth_vol_7d", "etha_eth_ratio", "feth_flow_usd"} <= set(last)
        assert not any(k.startswith(("btc", "ibit")) for k in last)
        assert "etha" in data["summary"]


class TestMacroAsset:
    def test_eth_chiede_eth_a_coinglass(self, client):
        with patch("src.flows.coinglass_client.CoinGlassClient") as cg:
            inst = cg.return_value
            inst.has_api_key = True
            inst.fetch_funding_rate_history.return_value = pd.Series([0.01], index=pd.to_datetime(["2026-01-01"]))
            inst.fetch_aggregated_oi_history.return_value = pd.Series(dtype=float)
            inst.fetch_long_short_ratio.return_value = pd.Series(dtype=float)
            inst.fetch_liquidations.return_value = pd.DataFrame()
            inst.fetch_taker_volume.return_value = pd.Series(dtype=float)

            r = client.get("/api/macro?asset=eth")

        assert r.status_code == 200, r.text
        assert r.json()["data"]["asset"] == "ETH"
        for m in ("fetch_funding_rate_history", "fetch_aggregated_oi_history",
                  "fetch_long_short_ratio", "fetch_liquidations", "fetch_taker_volume"):
            assert getattr(inst, m).call_args.kwargs["asset"] == "ETH", m
