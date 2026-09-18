"""Dashboard Streamlit per ibit-gamma-tracker — orchestratore.

Visualizza in tempo reale:
  - Barrier Map  — mappa livelli critici note strutturate IBIT
  - GEX          — Gamma Exposure BTC (Deribit), regime, profilo per strike
  - ETF Flows    — flussi istituzionali IBIT, correlazione rolling
  - Segnali      — segnale composito operativo + backtest
  - EDGAR Monitor — monitor note strutturate SEC
  - Validation   — walk-forward, factor decomposition, parameter sensitivity

Architettura:
  - app.py        — entrypoint: carica i dati condivisi una volta, renderizza
                    header + sidebar, poi `st.navigation` (solo la pagina attiva
                    viene eseguita → niente più backtest/sensitivity a ogni load)
  - app_pages/*   — una pagina Streamlit per sezione (thin wrapper sulle funzioni
                    `_tab_*` in tabs/)
  - header/sidebar — KPI strip e stato dati
  - tabs/*        — contenuto delle 6 sezioni
  - charts        — funzioni Plotly condivise

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
    run_backtest,
    run_event_study,
    run_factor_decomp,
    run_granger,
    run_regime,
    run_sensitivity,
    run_signal_ic,
    run_walk_forward,
)
from src.dashboard.components import inject_style
from src.dashboard.header import _render_header
from src.dashboard.sidebar import _sidebar

_PAGES_DIR = Path(__file__).resolve().parent / "app_pages"

st.set_page_config(
    page_title="ibit-gamma-tracker",
    page_icon=":material/currency_bitcoin:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _load_shared() -> tuple[dict, list[dict], pd.DataFrame, list[dict]]:
    """Carica in parallelo i tre dataset condivisi (GEX, flussi, barriere).

    Sono usati dall'header KPI e da quasi tutte le pagine: li carichiamo qui una
    volta sola (con cache 15 min) e li mettiamo in session_state.
    """
    snap: dict = {"spot_price": 0, "total_net_gex": 0, "regime": "unknown", "alerts": []}
    gex_by_strike: list[dict] = []
    merged_df = pd.DataFrame()
    barriers: list[dict] = []

    with st.spinner("Caricamento dati in parallelo (GEX · Flussi · Barriere)..."):
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(load_gex): "gex",
                pool.submit(load_prices_and_flows): "flows",
                pool.submit(load_barriers): "barriers",
            }
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
    snap, gex_by_strike, merged_df, barriers = _load_shared()

    # Dati condivisi tra le pagine (evitano di ricaricarli in ogni script)
    st.session_state["snap"] = snap
    st.session_state["gex_by_strike"] = gex_by_strike
    st.session_state["merged_df"] = merged_df
    st.session_state["barriers"] = barriers

    manual_refresh = _sidebar(snap, merged_df, barriers)

    if manual_refresh:
        for fn in [
            load_prices_and_flows,
            load_gex,
            load_barriers,
            load_db_summary,
            load_macro,
            run_granger,
            run_regime,
            run_backtest,
            run_event_study,
            run_walk_forward,
            run_factor_decomp,
            run_sensitivity,
            run_signal_ic,
        ]:
            fn.clear()
        st.rerun()

    _render_header(snap, merged_df)
    inject_style()

    pages = [
        st.Page(
            str(_PAGES_DIR / "panoramica.py"),
            title="Panoramica",
            icon=":material/dashboard:",
            default=True,
        ),
        st.Page(str(_PAGES_DIR / "signals.py"), title="Segnali", icon=":material/traffic:"),
        st.Page(str(_PAGES_DIR / "gex.py"), title="GEX", icon=":material/candlestick_chart:"),
        st.Page(str(_PAGES_DIR / "flows.py"), title="ETF Flows", icon=":material/water:"),
        st.Page(str(_PAGES_DIR / "barrier_map.py"), title="Barrier Map", icon=":material/sell:"),
        st.Page(str(_PAGES_DIR / "edgar.py"), title="EDGAR", icon=":material/search:"),
        st.Page(str(_PAGES_DIR / "validation.py"), title="Validation", icon=":material/science:"),
    ]
    pg = st.navigation(pages, position="top")
    pg.run()

    st.caption(
        f"Ultimo aggiornamento: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} · "
        f"GEX: Deribit (pubblica) · Flussi: Farside/yfinance · "
        f"Note strutturate: SEC EDGAR · Sviluppato da WAGMI-LAB"
    )


if __name__ == "__main__":
    main()
