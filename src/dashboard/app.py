"""Dashboard Streamlit per ibit-gamma-tracker — orchestratore.

Visualizza in tempo reale:
  - Panoramica   — segnale composito, livelli GEX e flussi a colpo d'occhio
  - Barrier Map  — mappa livelli critici note strutturate IBIT
  - GEX          — Gamma Exposure BTC (Deribit), regime, profilo per strike
  - ETF Flows    — flussi istituzionali IBIT, correlazione rolling
  - EDGAR Monitor — monitor note strutturate SEC

Architettura:
  - app.py        — entrypoint: carica i dati condivisi una volta, renderizza
                    header + sidebar, poi `st.navigation` (solo la pagina attiva
                    viene eseguita)
  - app_pages/*   — una pagina Streamlit per sezione (thin wrapper sulle funzioni
                    `_tab_*` in tabs/)
  - header/sidebar — KPI strip e stato dati
  - tabs/*        — contenuto delle 5 sezioni
  - charts        — funzioni Plotly condivise
  - navigation    — pagine visibili per asset (BTC: 5, ETH: Panoramica/GEX/Flussi)

Asset: il selettore BTC/ETH in sidebar (``?asset=eth`` nell'URL) decide cosa si
carica. Lo spec scelto va in ``st.session_state["asset_spec"]`` per le pagine.

Avvio:
    streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Aggiunge la root del progetto al sys.path (necessario quando il package
# non e' installato in editable mode)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed

import pandas as pd
import streamlit as st

from src.dashboard.data_loader import (
    load_barriers,
    load_db_summary,
    load_gex,
    load_macro,
    load_prices_and_flows,
    run_event_study,
    run_granger,
    run_regime,
)
from src.assets import AssetSpec
from src.dashboard.components import inject_style
from src.dashboard.header import _render_header
from src.dashboard.navigation import visible_pages
from src.dashboard.sidebar import _asset_selector, _sidebar

st.set_page_config(
    page_title="ibit-gamma-tracker",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _load_shared(spec: AssetSpec) -> tuple[dict, list[dict], pd.DataFrame, list[dict]]:
    """Carica in parallelo i dataset condivisi dell'asset (GEX, flussi, barriere).

    Sono usati dall'header KPI e da quasi tutte le pagine: li carichiamo qui una
    volta sola (con cache 15 min) e li mettiamo in session_state. Le barriere
    EDGAR esistono solo per BTC: per gli altri asset non si caricano.
    """
    snap: dict = {"spot_price": 0, "total_net_gex": 0, "regime": "unknown", "alerts": []}
    gex_by_strike: list[dict] = []
    merged_df = pd.DataFrame()
    barriers: list[dict] = []

    with st.spinner(f"Caricamento dati {spec.key} in parallelo..."):
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(load_gex, spec.key): "gex",
                pool.submit(load_prices_and_flows, spec.key): "flows",
            }
            if spec.has("barriers"):
                futures[pool.submit(load_barriers)] = "barriers"
            for future in as_completed(futures):
                key = futures[future]
                try:
                    result = future.result(timeout=15)
                except TimeoutError:
                    st.warning(f"Timeout caricamento {key} (15s) — dati parziali")
                    continue
                except Exception as e:
                    if key == "gex":
                        st.warning(f"GEX non disponibile: {e}")
                    elif key == "flows":
                        st.error(f"Errore caricamento prezzi/flussi: {e}")
                    else:
                        st.warning(f"Barriere non disponibili: {e}")
                    continue
                if key == "gex":
                    snap, gex_by_strike = result
                elif key == "flows":
                    merged_df = result
                else:
                    barriers = result

    return snap, gex_by_strike, merged_df, barriers


def main() -> None:
    spec = _asset_selector()
    snap, gex_by_strike, merged_df, barriers = _load_shared(spec)

    # Dati condivisi tra le pagine (evitano di ricaricarli in ogni script)
    st.session_state["asset_spec"] = spec
    st.session_state["snap"] = snap
    st.session_state["gex_by_strike"] = gex_by_strike
    st.session_state["merged_df"] = merged_df
    st.session_state["barriers"] = barriers

    manual_refresh = _sidebar(snap, merged_df, barriers, spec)

    if manual_refresh:
        for fn in [
            load_prices_and_flows,
            load_gex,
            load_barriers,
            load_db_summary,
            load_macro,
            run_granger,
            run_regime,
            run_event_study,
        ]:
            fn.clear()
        st.rerun()

    _render_header(snap, merged_df, spec)
    inject_style()

    # Solo le pagine con dati per l'asset: con ETH spariscono Barrier Map ed
    # EDGAR (se eri su una di quelle, si torna alla default).
    pages = [
        st.Page(p.path, title=p.title, icon=p.icon, default=p.default)
        for p in visible_pages(spec)
    ]
    pg = st.navigation(pages, position="top")
    pg.run()

    fonti = (
        "GEX: Deribit (pubblica) · Flussi: Farside/yfinance · Note strutturate: SEC EDGAR"
        if spec.has("edgar")
        else "GEX: Deribit (pubblica) · Flussi: CoinGlass/Farside · Derivati: CoinGlass/CoinGecko"
    )
    st.caption(
        f"Ultimo aggiornamento: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} · "
        f"{fonti} · Sviluppato da WAGMI-LAB"
    )


if __name__ == "__main__":
    main()
