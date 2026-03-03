"""Dashboard Streamlit per ibit-gamma-tracker.

Visualizza in tempo reale:
  - GEX BTC (Deribit) — regime, profilo per strike, livelli chiave
  - ETF Flows IBIT — storico flussi, correlazione rolling con BTC
  - Analytics — Granger causality, regime analysis, backtest
  - EDGAR Barriers — note strutturate attive

Avvio:
    streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Assicura import locali
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from src.config import get_settings, setup_logging

_log = setup_logging("dashboard.app")
_settings = get_settings()
_theme = _settings["dashboard"]["theme"]
_REFRESH = _settings["dashboard"]["refresh_interval_s"]

# Convenience aliases for new theme keys (with safe fallbacks)
_SURFACE    = _theme.get("surface",    "#161b22")
_BORDER     = _theme.get("border",     "#30363d")
_TEXT_MUTED = _theme.get("text_muted", "#8b949e")
_ACCENT     = _theme.get("accent",     "#a371f7")


# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ibit-gamma-tracker",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system: professional financial dark theme ──────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  /* ── Base ── */
  .stApp {{
      background-color: {_theme["background"]};
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  }}
  .main .block-container {{
      padding-top: 1.5rem;
      padding-bottom: 2rem;
      max-width: 1400px;
  }}

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {{
      background-color: {_SURFACE};
      border-right: 1px solid {_BORDER};
  }}
  section[data-testid="stSidebar"] .stButton button {{
      background: {_theme["background"]};
      color: {_theme["text"]};
      border: 1px solid {_BORDER};
      border-radius: 8px;
      font-family: 'Inter', sans-serif;
      font-weight: 500;
      transition: all 0.2s;
  }}
  section[data-testid="stSidebar"] .stButton button:hover {{
      border-color: {_theme["neutral"]};
      color: {_theme["neutral"]};
      background: {_SURFACE};
  }}

  /* ── KPI Metric Cards ── */
  [data-testid="metric-container"] {{
      background: {_SURFACE};
      border: 1px solid {_BORDER};
      border-radius: 12px;
      padding: 16px 20px !important;
      transition: border-color 0.2s;
  }}
  [data-testid="metric-container"]:hover {{
      border-color: {_theme["neutral"]};
  }}
  [data-testid="stMetricLabel"] > div {{
      font-size: 10px !important;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: {_TEXT_MUTED} !important;
      font-weight: 600;
      font-family: 'Inter', sans-serif;
  }}
  [data-testid="stMetricValue"] {{
      font-family: 'JetBrains Mono', monospace !important;
      font-size: 22px !important;
      font-weight: 600;
      color: {_theme["text"]} !important;
  }}
  [data-testid="stMetricDelta"] svg {{ display: none; }}
  [data-testid="stMetricDelta"] > div {{
      font-family: 'JetBrains Mono', monospace !important;
      font-size: 12px !important;
  }}

  /* ── Tab Navigation ── */
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
      font-family: 'Inter', sans-serif;
  }}
  .stTabs [aria-selected="true"] {{
      background: {_BORDER} !important;
      color: {_theme["text"]} !important;
  }}
  .stTabs [data-baseweb="tab"]:hover {{
      color: {_theme["text"]} !important;
      background: rgba(48,54,61,0.5) !important;
  }}
  .stTabs [data-baseweb="tab-highlight"] {{ display: none; }}

  /* ── Headings ── */
  h1 {{
      font-size: 1.75rem !important;
      font-weight: 700 !important;
      letter-spacing: -0.02em !important;
      color: {_theme["text"]} !important;
      font-family: 'Inter', sans-serif !important;
  }}
  h2, h3 {{
      font-size: 0.72rem !important;
      font-weight: 600 !important;
      text-transform: uppercase !important;
      letter-spacing: 0.07em !important;
      color: {_TEXT_MUTED} !important;
      font-family: 'Inter', sans-serif !important;
  }}

  /* ── Dividers ── */
  hr {{ border-color: {_BORDER} !important; margin: 1.25rem 0 !important; }}

  /* ── DataFrames ── */
  [data-testid="stDataFrame"] {{
      border: 1px solid {_BORDER};
      border-radius: 10px;
      overflow: hidden;
  }}

  /* ── Alert / Info boxes ── */
  .stAlert {{ border-radius: 8px !important; }}

  /* ── Regime Badge ── */
  .regime-badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 14px;
      border-radius: 20px;
      font-weight: 600;
      font-size: 11px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      font-family: 'Inter', sans-serif;
  }}
  .regime-dot {{
      width: 6px;
      height: 6px;
      border-radius: 50%;
      display: inline-block;
      flex-shrink: 0;
  }}

  /* ── Captions / Footer ── */
  .stCaptionContainer p {{
      color: {_TEXT_MUTED} !important;
      font-size: 0.78rem !important;
      font-family: 'Inter', sans-serif !important;
  }}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading (cached)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=_REFRESH, show_spinner=False)
def load_prices_and_flows() -> pd.DataFrame:
    """Carica prezzi BTC/IBIT e flussi ETF aggregati."""
    from src.flows.correlation import FlowCorrelation
    from src.flows.price_fetcher import PriceFetcher
    from src.flows.scraper import FarsideScraper

    pf = PriceFetcher()
    prices = pf.get_all_prices()
    if prices.empty:
        pf.fetch_and_store(tickers=["BTC-USD", "IBIT"], days_back=365)
        prices = pf.get_all_prices()

    scraper = FarsideScraper()
    flows = scraper.fetch()
    agg_flows = scraper.aggregate(flows)

    corr = FlowCorrelation()
    return corr.merge(agg_flows, prices)


@st.cache_data(ttl=_REFRESH, show_spinner=False)
def load_gex() -> tuple[dict, list[dict]]:
    """Carica snapshot GEX live da Deribit.

    Returns:
        (snapshot_dict, gex_by_strike_list)
    """
    from src.gex.deribit_client import DeribitClient
    from src.gex.gex_calculator import GexCalculator
    from src.gex.regime_detector import RegimeDetector

    client = DeribitClient()
    calc = GexCalculator()
    detector = RegimeDetector()

    spot = client.get_spot_price()
    options = client.fetch_all_options("BTC")
    snap = calc.calculate_gex(options, spot)
    state = detector.detect(snap)

    snap_dict = calc.gex_to_dict(snap)
    snap_dict["regime"] = state.regime
    snap_dict["alerts"] = state.alerts
    snap_dict["gex_percentile"] = state.gex_percentile

    by_strike = [
        {
            "strike":  g.strike,
            "net_gex": g.net_gex,
            "call_gex": g.call_gex,
            "put_gex":  g.put_gex,
        }
        for g in snap.gex_by_strike
    ]
    return snap_dict, by_strike


@st.cache_data(ttl=_REFRESH, show_spinner=False)
def load_barriers() -> list[dict]:
    """Carica barriere attive dal DB EDGAR."""
    from src.edgar.structured_notes_db import StructuredNotesDB
    db = StructuredNotesDB()
    return db.get_active_barriers()


@st.cache_data(ttl=_REFRESH, show_spinner=False)
def run_granger(merged_df: pd.DataFrame) -> tuple[dict, pd.DataFrame, str]:
    """Esegue il test di Granger causality."""
    from src.analytics.granger import GrangerAnalysis
    analyzer = GrangerAnalysis()
    results = analyzer.run(merged_df)
    df = analyzer.to_dataframe(results)
    interp = analyzer.interpret(results)
    return results, df, interp


@st.cache_data(ttl=_REFRESH, show_spinner=False)
def run_regime(merged_df: pd.DataFrame, gex_today: float):
    """Esegue la regime analysis."""
    from src.analytics.regime_analysis import RegimeAnalysis
    analyzer = RegimeAnalysis()
    # Tenta di trovare colonna total_gex nel merged_df
    return analyzer.analyze(merged_df)


@st.cache_data(ttl=_REFRESH, show_spinner=False)
def run_backtest(merged_df: pd.DataFrame, barriers: list[dict]):
    """Esegue il backtest."""
    from src.analytics.backtest import Backtest
    bt = Backtest()
    return bt, bt.run(merged_df, active_barriers=barriers if barriers else None)


@st.cache_data(ttl=_REFRESH, show_spinner=False)
def run_event_study(barriers: list[dict], merged_df: pd.DataFrame) -> list:
    """Esegue l'event study sui barrier levels."""
    from src.analytics.event_study import EventStudy
    if not barriers or "btc_close" not in merged_df.columns:
        return []
    study = EventStudy()
    prices_df = merged_df[["btc_close"]].rename(columns={"btc_close": "close"}).dropna()
    return study.run(barriers, prices_df)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

def _sidebar() -> bool:
    """Renderizza sidebar e restituisce True se si vuole refresh manuale."""
    with st.sidebar:
        st.markdown(f"""
<div style="display:flex;align-items:center;gap:8px;margin-bottom:0.15rem">
  <span style="font-size:1rem;font-weight:700;color:{_theme['text']};
               font-family:'Inter',sans-serif">Impostazioni</span>
</div>
<p style="color:{_TEXT_MUTED};font-size:0.75rem;margin:0 0 0.75rem;
          font-family:'Inter',sans-serif">
  Refresh automatico ogni {_REFRESH // 60} min
</p>
""", unsafe_allow_html=True)

        refresh = st.button("Aggiorna dati", use_container_width=True)

        st.divider()

        st.markdown(f"""
<p style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;
          color:{_TEXT_MUTED};margin:0 0 0.5rem;font-family:'Inter',sans-serif">
  Soglie Backtest
</p>
""", unsafe_allow_html=True)

        cfg = _settings.get("backtest", {})
        st.metric("Long GEX threshold", "$0")
        st.metric("Long Flow (3d)", f"+{cfg.get('long_flow_threshold_usd_m', 100)}M")
        st.metric("Short Flow (3d)", f"{cfg.get('short_flow_threshold_usd_m', -200)}M")
        st.metric("Barrier exclusion", f"±{cfg.get('barrier_exclusion_pct', 5)}%")

        st.divider()

        st.markdown(f"""
<div style="color:{_TEXT_MUTED};font-size:0.72rem;line-height:1.7;font-family:'Inter',sans-serif">
  <span style="color:{_theme['text']};font-weight:600">ibit-gamma-tracker</span> v1.0<br>
  Deribit · EDGAR · yfinance
</div>
""", unsafe_allow_html=True)

    return refresh


# ──────────────────────────────────────────────────────────────────────────────
# Header — KPI
# ──────────────────────────────────────────────────────────────────────────────

def _render_header(snap: dict, merged_df: pd.DataFrame) -> None:
    regime = snap.get("regime", "unknown")
    color_map = {
        "positive_gamma": _theme["positive"],
        "negative_gamma": _theme["negative"],
        "neutral":        _theme["neutral"],
    }
    regime_color = color_map.get(regime, _theme["text"])

    # Title + regime badge on the same line
    st.markdown(f"""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:0.2rem">
  <span style="font-size:1.75rem;font-weight:700;letter-spacing:-0.02em;
               font-family:'Inter',sans-serif;color:{_theme['text']}">
    ₿ ibit-gamma-tracker
  </span>
  <span class="regime-badge"
        style="background:{regime_color}18;color:{regime_color};
               border:1px solid {regime_color}50">
    <span class="regime-dot" style="background:{regime_color}"></span>
    {regime.upper().replace("_", " ")}
  </span>
</div>
<p style="color:{_TEXT_MUTED};font-size:0.875rem;margin:0 0 1.25rem;
          font-family:'Inter',sans-serif;line-height:1.4">
  Analisi dealer hedging su note strutturate IBIT · BTC
</p>
""", unsafe_allow_html=True)

    spot = snap.get("spot_price", 0)
    gex_m = snap.get("total_net_gex", 0) / 1e6

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("BTC Spot", f"${spot:,.0f}")
    col2.metric("GEX Totale", f"{gex_m:+.1f}M$")
    col3.metric(
        "Put Wall",
        f"${snap.get('put_wall', 0):,.0f}",
        delta=f"{snap.get('distance_to_put_wall_pct', 0):.1f}%",
        delta_color="inverse",
    )
    col4.metric(
        "Call Wall",
        f"${snap.get('call_wall', 0):,.0f}",
        delta=f"{snap.get('distance_to_call_wall_pct', 0):.1f}%",
    )
    if not merged_df.empty and "btc_return" in merged_df.columns:
        last_ret = merged_df["btc_return"].dropna().iloc[-1] * 100 if not merged_df["btc_return"].dropna().empty else 0
        col5.metric("BTC Return (ieri)", f"{last_ret:+.2f}%")
    else:
        col5.metric("Gamma Flip", f"${snap.get('gamma_flip_price', 0):,.0f}")

    # Alerts
    alerts = snap.get("alerts", [])
    if alerts:
        for alert in alerts:
            st.warning(f"⚠️ {alert}")


# ──────────────────────────────────────────────────────────────────────────────
# Tab: GEX
# ──────────────────────────────────────────────────────────────────────────────

def _tab_gex(snap: dict, gex_by_strike: list[dict]) -> None:
    from src.dashboard.charts import gex_profile, gex_walls

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(
            gex_profile(gex_by_strike, snap.get("spot_price", 0)),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            gex_walls(snap),
            use_container_width=True,
        )

    # Metriche dettagliate
    st.subheader("Statistiche GEX")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gamma Flip",   f"${snap.get('gamma_flip_price', 0):,.0f}")
    c2.metric("Max Pain",     f"${snap.get('max_pain', 0):,.0f}")
    c3.metric("Put/Call OI",  f"{snap.get('put_call_ratio', 0):.2f}")
    c4.metric("Strumenti BTC", f"{snap.get('n_instruments', 0)}")


# ──────────────────────────────────────────────────────────────────────────────
# Tab: Flows
# ──────────────────────────────────────────────────────────────────────────────

def _tab_flows(merged_df: pd.DataFrame) -> None:
    from src.dashboard.charts import flows_chart

    if merged_df.empty:
        st.warning("Dati flussi non disponibili.")
        return

    st.plotly_chart(flows_chart(merged_df), use_container_width=True)

    # Statistiche riepilogative
    st.subheader("Riepilogo ultimi 30 giorni")
    recent = merged_df.last("30D")
    c1, c2, c3 = st.columns(3)
    if "ibit_flow" in recent.columns:
        total_flow = recent["ibit_flow"].sum() / 1e6
        c1.metric("Flusso IBIT (30d)", f"{total_flow:+.0f}M$")
        pos_days = (recent["ibit_flow"] > 0).sum()
        c2.metric("Giorni inflow", f"{pos_days}/30")
    if "btc_return" in recent.columns:
        btc_cum = (1 + recent["btc_return"].dropna()).prod() - 1
        c3.metric("BTC Return (30d)", f"{btc_cum*100:+.1f}%")


# ──────────────────────────────────────────────────────────────────────────────
# Tab: Analytics
# ──────────────────────────────────────────────────────────────────────────────

def _tab_analytics(merged_df: pd.DataFrame, gex_today: float) -> None:
    from src.dashboard.charts import granger_heatmap, regime_bars

    if merged_df.empty:
        st.warning("Dati insufficienti per gli analytics.")
        return

    st.subheader("Granger Causality: IBIT Flows → BTC Returns")
    with st.spinner("Calcolo Granger causality..."):
        _, granger_df, granger_text = run_granger(merged_df)

    st.plotly_chart(granger_heatmap(granger_df), use_container_width=True)

    with st.expander("📋 Interpretazione completa Granger"):
        st.text(granger_text)

    st.divider()
    st.subheader("Regime Analysis: Positive vs Negative Gamma")

    with st.spinner("Calcolo regime analysis..."):
        regime_result = run_regime(merged_df, gex_today)

    if regime_result.positive_stats or regime_result.negative_stats:
        st.plotly_chart(regime_bars(regime_result), use_container_width=True)
        with st.expander("📋 Interpretazione completa Regime"):
            st.text(regime_result.interpretation)

        if regime_result.gex_vol_correlation is not None:
            corr_mean = regime_result.gex_vol_correlation.dropna().mean()
            st.info(f"Correlazione media GEX ↔ BTC Vol (rolling 30d): **{corr_mean:.3f}**")
    else:
        st.info(
            "Regime analysis richiede dati GEX storici. "
            "Il sistema accumula snapshot GEX nel tempo — i risultati miglioreranno."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Tab: Backtest
# ──────────────────────────────────────────────────────────────────────────────

def _tab_backtest(merged_df: pd.DataFrame, barriers: list[dict]) -> None:
    from src.dashboard.charts import backtest_equity

    if merged_df.empty:
        st.warning("Dati insufficienti per il backtest.")
        return

    with st.spinner("Esecuzione backtest..."):
        bt, results = run_backtest(merged_df, barriers)

    if not results:
        st.error("Backtest non disponibile.")
        return

    # Tabella riepilogativa
    st.subheader("Performance Summary")
    table = bt.summary_table(results)
    st.dataframe(table, use_container_width=True)

    # Delta Sharpe
    strat = results["strategy"]
    bah   = results["buy_and_hold"]
    delta_sharpe = strat.sharpe_ratio - bah.sharpe_ratio
    col1, col2, col3 = st.columns(3)
    col1.metric("Delta Sharpe (vs B&H)", f"{delta_sharpe:+.2f}")
    col2.metric("Giorni Long", strat.days_long)
    col3.metric("N Trades", strat.n_trades)

    # Equity curve
    st.subheader("Equity Curve")
    st.plotly_chart(backtest_equity(results), use_container_width=True)

    if strat.days_long == 0 and strat.days_short == 0:
        st.info(
            "La strategia è flat su tutto il periodo: il GEX storico non è ancora "
            "disponibile. Il backtest mostrerà segnali reali man mano che il sistema "
            "accumula snapshot GEX giornalieri."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Tab: EDGAR Barriers
# ──────────────────────────────────────────────────────────────────────────────

def _tab_edgar(barriers: list[dict], merged_df: pd.DataFrame) -> None:
    from src.dashboard.charts import event_study_car

    if not barriers:
        st.info("Nessuna barriera attiva nel DB. Esegui `scripts/run_edgar.py` per popolare il database.")
        return

    st.subheader(f"Barriere attive ({len(barriers)})")

    # Tabella barriere
    rows = []
    for b in barriers:
        rows.append({
            "Tipo":           b.get("barrier_type", ""),
            "Emittente":      b.get("issuer", ""),
            "Prodotto":       b.get("product_type", ""),
            "Livello %":      f"{b.get('level_pct', 0):.0f}%" if b.get("level_pct") else "—",
            "Prezzo IBIT":    f"${b.get('level_price_ibit', 0):.2f}" if b.get("level_price_ibit") else "—",
            "Prezzo BTC":     f"${b.get('level_price_btc', 0):,.0f}" if b.get("level_price_btc") else "—",
            "Scadenza":       b.get("maturity_date", ""),
            "Status":         b.get("status", ""),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Event study
    st.subheader("Event Study — CAR intorno ai Barrier Levels")
    with st.spinner("Calcolo event study..."):
        event_results = run_event_study(barriers, merged_df)

    fig = event_study_car(event_results)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        n_total = sum(r.n_events for r in event_results) if event_results else 0
        if n_total == 0:
            st.info(
                "Nessun evento trovato: i prezzi BTC delle barriere non sono ancora "
                "stati raggiuti nel periodo analizzato, oppure i prezzi BTC dei barrier "
                "levels non sono disponibili nel DB."
            )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    manual_refresh = _sidebar()

    if manual_refresh:
        load_prices_and_flows.clear()
        load_gex.clear()
        load_barriers.clear()
        run_granger.clear()
        run_regime.clear()
        run_backtest.clear()
        run_event_study.clear()
        st.rerun()

    # ── Caricamento dati ────────────────────────────────────────────────────
    with st.spinner("Carico prezzi e flussi ETF..."):
        try:
            merged_df = load_prices_and_flows()
        except Exception as e:
            st.error(f"Errore caricamento prezzi/flussi: {e}")
            merged_df = pd.DataFrame()

    with st.spinner("Carico GEX live da Deribit..."):
        try:
            snap, gex_by_strike = load_gex()
        except Exception as e:
            st.warning(f"GEX non disponibile: {e}")
            snap = {"spot_price": 0, "total_net_gex": 0, "regime": "unknown", "alerts": []}
            gex_by_strike = []

    with st.spinner("Carico barriere EDGAR..."):
        try:
            barriers = load_barriers()
        except Exception as e:
            st.warning(f"Barriere non disponibili: {e}")
            barriers = []

    # ── Header KPI ─────────────────────────────────────────────────────────
    _render_header(snap, merged_df)

    st.divider()

    # ── Tabs ────────────────────────────────────────────────────────────────
    tab_gex, tab_flows, tab_analytics, tab_backtest, tab_edgar = st.tabs([
        "📊 GEX",
        "💸 ETF Flows",
        "🔬 Analytics",
        "📈 Backtest",
        "🏛️ EDGAR Barriers",
    ])

    with tab_gex:
        _tab_gex(snap, gex_by_strike)

    with tab_flows:
        _tab_flows(merged_df)

    with tab_analytics:
        _tab_analytics(merged_df, snap.get("total_net_gex", 0))

    with tab_backtest:
        _tab_backtest(merged_df, barriers)

    with tab_edgar:
        _tab_edgar(barriers, merged_df)

    # ── Footer ──────────────────────────────────────────────────────────────
    st.caption(
        f"Ultimo aggiornamento: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} · "
        f"Dati GEX: Deribit (pubblica) · Flussi: yfinance fallback · "
        f"Note strutturate: SEC EDGAR"
    )


if __name__ == "__main__":
    main()
