"""Dashboard Streamlit per ibit-gamma-tracker — orchestratore.

Visualizza in tempo reale:
  - Barrier Map  — mappa livelli critici note strutturate IBIT
  - GEX          — Gamma Exposure BTC (Deribit), regime, profilo per strike
  - ETF Flows    — flussi istituzionali IBIT, correlazione rolling
  - Segnali      — segnale composito operativo + backtest
  - EDGAR Monitor — monitor note strutturate SEC

Moduli:
  - data_loader   — @st.cache_data functions (fetch + caching)
  - header        — _render_header (KPI strip)
  - sidebar       — _sidebar (data status + backtest thresholds)
  - tabs/*        — _tab_barrier_map, _tab_gex, _tab_flows, _tab_signals, _tab_edgar_monitor
  - charts        — Plotly chart functions
  - config        — theme constants

Avvio:
    streamlit run src/dashboard/app.py
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import get_settings, setup_logging

_log = setup_logging("dashboard.app")
_settings = get_settings()
_theme = _settings["dashboard"]["theme"]
_REFRESH = _settings["dashboard"]["refresh_interval_s"]

_SURFACE = _theme.get("surface", "#161b22")
_BORDER = _theme.get("border", "#30363d")
_TEXT_MUTED = _theme.get("text_muted", "#8b949e")
_ACCENT = _theme.get("accent", "#a371f7")

# ──────────────────────────────────────────────────────────────────────────────
# Page config + CSS
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ibit-gamma-tracker",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

  .stApp {{
      background-color: {_theme["background"]};
      font-family: 'Proxima Nova', 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif;
  }}
  .main .block-container {{
      padding-top: 1.5rem;
      padding-bottom: 2rem;
      max-width: 1400px;
  }}
  section[data-testid="stSidebar"] {{
      background-color: {_SURFACE};
      border-right: 1px solid {_BORDER};
  }}
  section[data-testid="stSidebar"] .stButton button {{
      background: {_theme["background"]};
      color: {_theme["text"]};
      border: 1px solid {_theme["positive"]};
      border-radius: 8px;
      font-weight: 500;
      transition: all 0.2s;
  }}
  section[data-testid="stSidebar"] .stButton button:hover {{
      border-color: {_theme["positive"]};
      color: {_theme["positive"]};
      background: {_SURFACE};
      box-shadow: 0 0 10px {_theme["positive"]}40;
  }}
  [data-testid="metric-container"] {{
      background: {_SURFACE};
      border: 1px solid {_BORDER};
      border-radius: 10px;
      padding: 10px 16px;
      transition: border-color 0.15s;
  }}
  [data-testid="metric-container"]:hover {{ border-color: {_theme["positive"]}50; }}
  [data-testid="stMetricValue"] {{
      font-size: 22px !important;
      font-weight: 600;
      color: {_theme["text"]} !important;
  }}
  [data-testid="stMetricDelta"] svg {{ display: none; }}
  [data-testid="stMetricDelta"] > div {{
      font-family: 'JetBrains Mono', monospace !important;
      font-size: 12px !important;
  }}
  .stTabs [data-baseweb="tab-list"] {{
      background-color: {_SURFACE};
      border-radius: 10px;
      padding: 4px;
      border: 1px solid {_BORDER};
      gap: 2px;
  }}
  .stTabs [data-baseweb="tab"] {{
      background: transparent;
      border-radius: 7px;
      border: none;
      color: {_TEXT_MUTED};
      font-weight: 500;
      font-size: 14px;
      padding: 8px 18px;
      transition: all 0.15s;
  }}
  .stTabs [aria-selected="true"] {{
      background: {_BORDER} !important;
      color: {_theme["positive"]} !important;
      text-shadow: 0 0 8px {_theme["positive"]}80;
  }}
  .stTabs [data-baseweb="tab"]:hover {{
      color: {_theme["text"]} !important;
      background: rgba(0,255,157,0.06) !important;
  }}
  .stTabs [data-baseweb="tab-highlight"] {{ display: none; }}
  h1 {{
      font-size: 1.75rem !important;
      font-weight: 700 !important;
      letter-spacing: -0.02em !important;
      color: {_theme["text"]} !important;
  }}
  h2, h3 {{
      font-size: 0.72rem !important;
      font-weight: 700 !important;
      text-transform: uppercase !important;
      letter-spacing: 0.07em !important;
      color: {_TEXT_MUTED} !important;
  }}
  hr {{ border-color: {_BORDER} !important; margin: 1.25rem 0 !important; }}
  [data-testid="stDataFrame"] {{
      border: 1px solid {_BORDER};
      border-radius: 10px;
      overflow: hidden;
  }}
  .stAlert {{ border-radius: 8px !important; }}
  .regime-badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 14px;
      border-radius: 20px;
      font-weight: 700;
      font-size: 11px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
  }}
  .regime-dot {{
      width: 6px; height: 6px;
      border-radius: 50%;
      display: inline-block;
      flex-shrink: 0;
  }}
  .stCaptionContainer p {{
      color: {_TEXT_MUTED} !important;
      font-size: 0.78rem !important;
  }}
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Imports dai moduli estratti
# ──────────────────────────────────────────────────────────────────────────────

from src.dashboard.data_loader import (
    load_prices_and_flows,
    load_gex,
    load_barriers,
    load_db_summary,
    load_macro,
    run_granger,
    run_regime,
    run_backtest,
    run_event_study,
    compute_composite,
)
from src.dashboard.header import _render_header
from src.dashboard.sidebar import _sidebar
from src.dashboard.tabs.barrier_map import _tab_barrier_map
from src.dashboard.tabs.gex import _tab_gex
from src.dashboard.tabs.flows import _tab_flows
from src.dashboard.tabs.signals import _tab_signals
from src.dashboard.tabs.edgar import _tab_edgar_monitor


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
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
                    result = future.result()
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

    manual_refresh = _sidebar(snap, merged_df, barriers)

    if manual_refresh:
        for fn in [load_prices_and_flows, load_gex, load_barriers,
                   load_db_summary, load_macro, run_granger, run_regime,
                   run_backtest, run_event_study]:
            fn.clear()
        st.rerun()

    _render_header(snap, merged_df)
    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Barrier Map",
        "📊 GEX",
        "💰 ETF Flows",
        "🚦 Segnali",
        "🔍 EDGAR Monitor",
    ])

    with tab1:
        _tab_barrier_map(barriers, snap)
    with tab2:
        _tab_gex(snap, gex_by_strike, merged_df)
    with tab3:
        _tab_flows(merged_df)
    with tab4:
        _tab_signals(snap, merged_df, barriers)
    with tab5:
        _tab_edgar_monitor(barriers, merged_df)

    st.caption(
        f"Ultimo aggiornamento: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} · "
        f"GEX: Deribit (pubblica) · Flussi: Farside/yfinance · "
        f"Note strutturate: SEC EDGAR · Sviluppato da WAGMI-LAB"
    )


if __name__ == "__main__":
    main()
