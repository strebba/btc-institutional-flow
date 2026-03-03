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


# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ibit-gamma-tracker",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS dark theme minimale
st.markdown(f"""
<style>
    .stApp {{ background-color: {_theme["background"]}; }}
    .metric-card {{
        background: {_theme["grid"]};
        border-radius: 8px;
        padding: 12px 16px;
        margin: 4px 0;
    }}
    .regime-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 14px;
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
        st.title("⚙️ Impostazioni")
        st.caption(f"Refresh automatico: ogni {_REFRESH // 60} min")

        refresh = st.button("🔄 Aggiorna dati", use_container_width=True)

        st.divider()
        st.subheader("Soglie Backtest")
        cfg = _settings.get("backtest", {})
        st.metric("Long GEX threshold", "$0")
        st.metric("Long Flow (3d)", f"+{cfg.get('long_flow_threshold_usd_m', 100)}M")
        st.metric("Short Flow (3d)", f"{cfg.get('short_flow_threshold_usd_m', -200)}M")
        st.metric("Barrier exclusion", f"±{cfg.get('barrier_exclusion_pct', 5)}%")

        st.divider()
        st.caption("ibit-gamma-tracker v1.0")
        st.caption("Dati: Deribit · EDGAR · yfinance")
    return refresh


# ──────────────────────────────────────────────────────────────────────────────
# Header — KPI
# ──────────────────────────────────────────────────────────────────────────────

def _render_header(snap: dict, merged_df: pd.DataFrame) -> None:
    st.title("₿ ibit-gamma-tracker")
    st.caption("Analisi dealer hedging su note strutturate IBIT · BTC")

    regime = snap.get("regime", "unknown")
    color_map = {
        "positive_gamma": _theme["positive"],
        "negative_gamma": _theme["negative"],
        "neutral":        _theme["neutral"],
    }
    regime_color = color_map.get(regime, _theme["text"])

    col1, col2, col3, col4, col5 = st.columns(5)

    spot = snap.get("spot_price", 0)
    gex_m = snap.get("total_net_gex", 0) / 1e6

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

    # Regime badge
    st.markdown(
        f'<span class="regime-badge" style="background:{regime_color}20;'
        f'color:{regime_color};border:1px solid {regime_color}">'
        f'Regime: {regime.upper().replace("_", " ")}</span>',
        unsafe_allow_html=True,
    )

    # Alert
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
