"""Dashboard multi-asset: pagine visibili, grafici e Panoramica per ETH."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.assets import get_asset


def _eth_merged(n: int = 40) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "etha_flow": rng.normal(5e7, 1e8, n), "feth_flow": rng.normal(1e7, 3e7, n),
        "total_flow": rng.normal(8e7, 1e8, n), "eth_close": 2500 + np.cumsum(rng.normal(0, 20, n)),
        "eth_return": rng.normal(0, 0.03, n), "eth_vol_7d": rng.uniform(0.4, 0.8, n),
    }, index=idx)
    df["etha_flow_3d"] = df["etha_flow"].rolling(3, min_periods=1).sum()
    return df


class TestPagineVisibili:
    def test_btc_ha_tutte_e_sette_le_pagine(self):
        from src.dashboard.navigation import visible_pages

        assert [p.title for p in visible_pages(get_asset("BTC"))] == [
            "Panoramica", "Segnali", "GEX", "ETF Flows", "Barrier Map", "EDGAR", "Validation",
        ]

    def test_eth_vede_solo_le_pagine_con_dati(self):
        from src.dashboard.navigation import visible_pages

        assert [p.title for p in visible_pages(get_asset("ETH"))] == ["Panoramica", "GEX", "ETF Flows"]


class TestGrafici:
    def test_flows_chart_eth_usa_etha_ed_eth(self):
        from src.dashboard.charts import flows_chart

        fig = flows_chart(_eth_merged(), get_asset("ETH"))
        nomi = {t.name for t in fig.data}
        assert {"ETHA Flow (M$)", "ETH"} <= nomi
        titoli = [a.text for a in fig.layout.annotations]
        assert "ETHA Flows (M$)" in titoli and "ETH Price ($)" in titoli

    def test_flows_chart_senza_colonna_lead_non_esplode(self):
        from src.dashboard.charts import flows_chart

        flows_chart(_eth_merged().drop(columns=["etha_flow"]), get_asset("ETH"))

    def test_assi_gex_parlano_dell_asset(self):
        from src.dashboard.charts import gex_profile, gex_walls

        prof = gex_profile([{"strike": 2600.0, "net_gex": 1e5}], 2500.0, asset="ETH")
        assert prof.layout.xaxis.title.text == "Strike ETH ($)"
        walls = gex_walls({"spot_price": 2500.0, "put_wall": 2400.0}, asset="ETH")
        assert walls.layout.yaxis.title.text == "Prezzo ETH ($)"


def _panoramica_eth_app():
    """Script AppTest: Panoramica ETH con dati sintetici, segnale vietato."""
    import numpy as np
    import pandas as pd

    import src.dashboard.tabs.panoramica as pan
    from src.assets import get_asset

    def _vietato(*a, **k):
        raise AssertionError("segnale composito calcolato su ETH")

    pan.load_macro = lambda asset="BTC": {"funding_rate_annualized_pct": -6.5, "oi_change_7d_pct": 2.0}
    pan.compute_composite = _vietato
    idx = pd.date_range("2026-01-01", periods=40, freq="D")
    merged = pd.DataFrame({
        "etha_flow": np.linspace(-1e8, 1e8, 40), "total_flow": np.linspace(-2e8, 2e8, 40),
        "eth_close": np.linspace(2400, 2600, 40), "eth_return": np.full(40, 0.01),
    }, index=idx)
    merged["etha_flow_3d"] = merged["etha_flow"].rolling(3, min_periods=1).sum()
    snap = {"spot_price": 2500.0, "gamma_flip_price": 2550.0, "put_wall": 2400.0,
            "call_wall": 2700.0, "regime": "positive_gamma", "total_net_gex": 3e5}
    pan._tab_panoramica(snap, merged, [], get_asset("ETH"))


class TestPanoramicaEth:
    def test_niente_segnale_ne_riferimenti_btc(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_function(_panoramica_eth_app, default_timeout=30).run()

        assert not at.exception, at.exception
        testi = " ".join(
            [m.value for m in at.markdown] + [c.value for c in at.caption]
            + [i.value for i in at.info] + [m.label for m in at.metric]
        )
        assert "solo per BTC" in testi
        assert "ETHA" in testi
        assert "IBIT" not in testi and "barriere" not in testi.lower()


class TestPineAsset:
    def test_titolo_dell_indicatore_segue_l_asset(self):
        from src.gex.pine_export import build_pine_indicator

        snap = {"gamma_flip_price": 2550.0, "spot_price": 2500.0, "regime": "positive_gamma"}
        assert 'indicator("WAGMI Lab — ETH GEX Levels"' in build_pine_indicator(snap, asset="ETH")
        assert 'indicator("WAGMI Lab — BTC GEX Levels"' in build_pine_indicator(snap)


def _header_senza_flip_app():
    import pandas as pd

    from src.assets import get_asset
    from src.dashboard.header import _render_header

    snap = {"spot_price": 2500.0, "gamma_flip_price": None, "put_wall": 2400.0,
            "call_wall": None, "regime": "neutral", "total_net_gex": -7e5}
    _render_header(snap, pd.DataFrame(), get_asset("ETH"))


class TestHeaderLivelliMancanti:
    def test_un_livello_assente_e_n_d_non_zero_dollari(self):
        """Con GEX vicino a zero il flip spesso non esiste: "$0 a -100%" sarebbe un dato inventato."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_function(_header_senza_flip_app, default_timeout=30).run()
        valori = {m.label: m.value for m in at.metric}
        assert valori["Gamma Flip"] == "n/d"
        assert valori["Call Wall"] == "n/d"
        assert valori["Put Wall"] == "$2,400"
