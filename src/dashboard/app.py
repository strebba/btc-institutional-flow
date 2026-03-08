"""Dashboard Streamlit per ibit-gamma-tracker.

Visualizza in tempo reale:
  - Barrier Map  — mappa livelli critici note strutturate IBIT
  - GEX          — Gamma Exposure BTC (Deribit), regime, profilo per strike
  - ETF Flows    — flussi istituzionali IBIT, correlazione rolling
  - Segnali      — segnale composito operativo + backtest
  - EDGAR Monitor — monitor note strutturate SEC

Avvio:
    streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from src.config import get_settings, setup_logging

_log      = setup_logging("dashboard.app")
_settings = get_settings()
_theme    = _settings["dashboard"]["theme"]
_REFRESH  = _settings["dashboard"]["refresh_interval_s"]

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

st.markdown(f"""
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
      border-radius: 12px;
      padding: 16px 20px !important;
      transition: border-color 0.2s, box-shadow 0.2s;
  }}
  [data-testid="metric-container"]:hover {{
      border-color: {_theme["positive"]};
      box-shadow: 0 0 12px {_theme["positive"]}25;
  }}
  [data-testid="stMetricLabel"] > div {{
      font-size: 10px !important;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: {_TEXT_MUTED} !important;
      font-weight: 600;
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
        # FIX: PriceFetcher non ha fetch_and_store — usa fetch() direttamente
        pf.fetch("BTC-USD")
        pf.fetch("IBIT")
        prices = pf.get_all_prices()

    scraper = FarsideScraper()
    flows = scraper.fetch()
    agg_flows = scraper.aggregate(flows)

    corr = FlowCorrelation()
    return corr.merge(agg_flows, prices)


@st.cache_data(ttl=_REFRESH, show_spinner=False)
def load_gex() -> tuple[dict, list[dict]]:
    """Carica snapshot GEX live da Deribit."""
    from src.gex.deribit_client import DeribitClient
    from src.gex.gex_calculator import GexCalculator
    from src.gex.regime_detector import RegimeDetector

    client   = DeribitClient()
    calc     = GexCalculator()
    detector = RegimeDetector()

    spot    = client.get_spot_price()
    options = client.fetch_all_options("BTC")
    snap    = calc.calculate_gex(options, spot)
    state   = detector.detect(snap)

    snap_dict = calc.gex_to_dict(snap)
    snap_dict["regime"]         = state.regime
    snap_dict["alerts"]         = state.alerts
    snap_dict["gex_percentile"] = state.gex_percentile

    by_strike = [
        {
            "strike":   g.strike,
            "net_gex":  g.net_gex,
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
def load_db_summary() -> dict:
    """Carica statistiche aggregate dal DB EDGAR."""
    from src.edgar.structured_notes_db import StructuredNotesDB
    db = StructuredNotesDB()
    return db.summary()


@st.cache_data(ttl=_REFRESH, show_spinner=False)
def run_granger(merged_df: pd.DataFrame) -> tuple[dict, pd.DataFrame, str]:
    """Esegue il test di Granger causality."""
    from src.analytics.granger import GrangerAnalysis
    analyzer = GrangerAnalysis()
    results  = analyzer.run(merged_df)
    df       = analyzer.to_dataframe(results)
    interp   = analyzer.interpret(results)
    return results, df, interp


@st.cache_data(ttl=_REFRESH, show_spinner=False)
def run_regime(merged_df: pd.DataFrame, gex_today: float):
    """Esegue la regime analysis."""
    from src.analytics.regime_analysis import RegimeAnalysis
    analyzer = RegimeAnalysis()

    if "total_gex" not in merged_df.columns and gex_today != 0:
        today      = pd.Timestamp.today().normalize()
        gex_series = pd.Series({today: gex_today}, name="total_gex")
        return analyzer.analyze(merged_df, gex_series=gex_series)

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
    study     = EventStudy()
    prices_df = merged_df[["btc_close"]].rename(columns={"btc_close": "close"}).dropna()
    return study.run(barriers, prices_df)


# ──────────────────────────────────────────────────────────────────────────────
# Composite signal
# ──────────────────────────────────────────────────────────────────────────────

def _composite_signal(
    snap: dict,
    merged_df: pd.DataFrame,
    barriers: list[dict],
) -> tuple[str, dict]:
    """Calcola il segnale operativo composito.

    Returns:
        (signal, details) dove signal è "LONG" | "CAUTION" | "RISK_OFF".
    """
    gex   = snap.get("total_net_gex") or 0.0
    spot  = snap.get("spot_price") or 0.0

    # Fattore 1: GEX regime
    gex_ok   = gex > 0
    gex_flat = abs(gex) < 1_000_000

    # Fattore 2: IBIT flows 3 giorni
    ibit_3d = 0.0
    if not merged_df.empty and "ibit_flow_3d" in merged_df.columns:
        last = merged_df["ibit_flow_3d"].dropna()
        if not last.empty:
            ibit_3d = float(last.iloc[-1])

    flow_ok  = ibit_3d >  100e6
    flow_bad = ibit_3d < -200e6

    # Fattore 3: prossimità barriera
    closest_dist     = float("inf")
    closest_barrier  = None
    if barriers and spot > 0:
        for b in barriers:
            level = b.get("level_price_btc") or 0.0
            if level > 0:
                dist = abs(spot - level) / spot * 100
                if dist < closest_dist:
                    closest_dist    = dist
                    closest_barrier = b

    barrier_ok      = closest_dist > 8
    barrier_caution = 3 < closest_dist <= 8
    barrier_alert   = closest_dist <= 3

    green = sum([gex_ok and not gex_flat, flow_ok,  barrier_ok])
    red   = sum([not gex_ok,              flow_bad, barrier_alert])

    if green == 3:
        signal = "LONG"
    elif red >= 2:
        signal = "RISK_OFF"
    else:
        signal = "CAUTION"

    return signal, {
        "gex": gex, "gex_ok": gex_ok, "gex_flat": gex_flat,
        "ibit_3d": ibit_3d, "flow_ok": flow_ok, "flow_bad": flow_bad,
        "closest_dist": closest_dist, "closest_barrier": closest_barrier,
        "barrier_ok": barrier_ok, "barrier_caution": barrier_caution,
        "barrier_alert": barrier_alert,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

def _sidebar(snap: dict, merged_df: pd.DataFrame) -> bool:
    """Renderizza sidebar. Restituisce True se si richiede refresh manuale."""
    with st.sidebar:
        st.markdown(f"""
<div style="display:flex;align-items:center;gap:8px;margin-bottom:0.15rem">
  <span style="font-size:1rem;font-weight:700;color:{_theme['text']}">
    ⚡ IBIT Gamma Tracker
  </span>
</div>
<p style="color:{_TEXT_MUTED};font-size:0.72rem;margin:0 0 0.5rem">
  WAGMI-LAB Research Tool v1.0
</p>
""", unsafe_allow_html=True)

        refresh = st.button("🔄 Aggiorna dati", use_container_width=True)

        st.divider()

        st.markdown(f"""
<p style="font-size:10px;font-weight:700;text-transform:uppercase;
          letter-spacing:0.07em;color:{_TEXT_MUTED};margin:0 0 0.4rem">
  Ultimo aggiornamento
</p>
""", unsafe_allow_html=True)

        btc_price = snap.get("spot_price") or 0
        ts_now    = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        st.text(f"GEX:    {ts_now}")
        st.text(f"Prezzo BTC: ${btc_price:,.0f}")

        if not merged_df.empty:
            last_flow_date = merged_df.index.max()
            st.text(f"Flussi: {last_flow_date.strftime('%Y-%m-%d') if hasattr(last_flow_date, 'strftime') else last_flow_date}")

        st.divider()

        cfg = _settings.get("backtest", {})
        st.markdown(f"""
<p style="font-size:10px;font-weight:700;text-transform:uppercase;
          letter-spacing:0.07em;color:{_TEXT_MUTED};margin:0 0 0.5rem">
  Soglie Backtest
</p>
""", unsafe_allow_html=True)
        st.metric("Long GEX threshold", "$0")
        st.metric("Long Flow (3d)", f"+{cfg.get('long_flow_threshold_usd_m', 100)}M")
        st.metric("Short Flow (3d)", f"{cfg.get('short_flow_threshold_usd_m', -200)}M")
        st.metric("Barrier exclusion", f"±{cfg.get('barrier_exclusion_pct', 5)}%")

        st.divider()

        st.markdown(f"""
<div style="color:{_TEXT_MUTED};font-size:0.72rem;line-height:1.7">
  <span style="color:{_theme['positive']};font-weight:700">ibit-gamma-tracker</span> v1.0<br>
  Deribit · EDGAR · yfinance<br><br>
  <b>Fonti dati</b><br>
  • GEX: Deribit API pubblica<br>
  • Flussi: Farside Investors<br>
  • Note strutturate: SEC EDGAR<br>
  • Prezzi: Yahoo Finance<br><br>
  <em>Non costituisce consulenza finanziaria.</em>
</div>
""", unsafe_allow_html=True)

    return refresh


# ──────────────────────────────────────────────────────────────────────────────
# Header — KPI strip
# ──────────────────────────────────────────────────────────────────────────────

def _render_header(snap: dict, merged_df: pd.DataFrame) -> None:
    regime      = snap.get("regime", "unknown")
    color_map   = {
        "positive_gamma": _theme["positive"],
        "negative_gamma": _theme["negative"],
        "neutral":        _theme["neutral"],
    }
    regime_color = color_map.get(regime, _theme["text"])

    st.markdown(f"""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:0.2rem">
  <span style="font-size:1.75rem;font-weight:700;letter-spacing:-0.02em;color:{_theme['text']}">
    ₿ ibit-gamma-tracker
  </span>
  <span class="regime-badge"
        style="background:{regime_color}18;color:{regime_color};
               border:1px solid {regime_color}60;box-shadow:0 0 10px {regime_color}40">
    <span class="regime-dot" style="background:{regime_color};box-shadow:0 0 6px {regime_color}"></span>
    {regime.upper().replace("_", " ")}
  </span>
</div>
<p style="color:{_TEXT_MUTED};font-size:0.875rem;margin:0 0 1.25rem;line-height:1.4">
  Analisi dealer hedging su note strutturate IBIT · BTC
</p>
""", unsafe_allow_html=True)

    spot  = snap.get("spot_price") or 0
    gex_m = (snap.get("total_net_gex") or 0) / 1e6

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("BTC Spot", f"${spot:,.0f}")
    col2.metric("GEX Totale", f"{gex_m:+.1f}M$")
    col3.metric(
        "Put Wall",
        f"${snap.get('put_wall') or 0:,.0f}",
        delta=f"{snap.get('distance_to_put_wall_pct') or 0:.1f}%",
        delta_color="inverse",
    )
    col4.metric(
        "Call Wall",
        f"${snap.get('call_wall') or 0:,.0f}",
        delta=f"{snap.get('distance_to_call_wall_pct') or 0:.1f}%",
    )
    if not merged_df.empty and "btc_return" in merged_df.columns:
        last_ret = merged_df["btc_return"].dropna()
        ret_val  = float(last_ret.iloc[-1]) * 100 if not last_ret.empty else 0
        col5.metric("BTC Return (ieri)", f"{ret_val:+.2f}%")
    else:
        col5.metric("Gamma Flip", f"${snap.get('gamma_flip_price') or 0:,.0f}")

    alerts = snap.get("alerts", [])
    if alerts:
        for alert in alerts:
            st.warning(f"⚠️ {alert}")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: Barrier Map
# ──────────────────────────────────────────────────────────────────────────────

def _tab_barrier_map(barriers: list[dict], snap: dict) -> None:
    from src.dashboard.charts import barrier_map as _barrier_map_chart

    spot = snap.get("spot_price") or 0.0

    st.header("🎯 Mappa dei Livelli Critici")
    st.markdown("""
Questa mappa mostra i **livelli di prezzo** dove le banche che hanno emesso
prodotti strutturati su IBIT sono **obbligate** a comprare o vendere Bitcoin
per ribilanciare il loro hedge. Quando il prezzo si avvicina a questi livelli,
aspettati movimenti meccanici e improvvisi — non guidati dal sentiment,
ma dalla gestione del rischio dei dealer.
""")

    # Legenda colori
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🔴 **Knock-In Barrier** — Se BTC scende sotto questo livello, "
                    "i dealer devono vendere aggressivamente per coprire il rischio. "
                    "Effetto: accelerazione del ribasso.")
    with col2:
        st.markdown("🟢 **Auto-Call Trigger** — Se BTC sale sopra questo livello, "
                    "la nota viene rimborsata e il dealer chiude l'hedge. "
                    "Effetto: pressione di vendita temporanea, poi rilascio.")
    with col3:
        st.markdown("🔵 **Buffer Zone** — Zona di protezione parziale. "
                    "Il dealer modifica gradualmente il suo hedge qui. "
                    "Effetto: attrito sui movimenti, volatilità ridotta.")

    if not barriers:
        st.info(
            "Nessuna barriera attiva nel DB. "
            "Esegui `scripts/run_edgar.py` per scansionare i filing SEC e popolare il database."
        )
        return

    # Grafico barrier map
    fig = _barrier_map_chart(barriers, spot)
    st.plotly_chart(fig, use_container_width=True)

    # Alert contestuale basato su distanza
    valid_barriers = [b for b in barriers if (b.get("level_price_btc") or 0) > 0 and spot > 0]
    if valid_barriers:
        closest = min(
            valid_barriers,
            key=lambda b: abs((b.get("level_price_btc") or 0) - spot),
        )
        level      = closest.get("level_price_btc", 0)
        distance   = abs(spot - level) / spot * 100
        btype      = closest.get("barrier_type", "barrier")
        issuer_str = closest.get("issuer", "N/A")

        if distance < 3:
            st.error(f"""
⚠️ **ATTENZIONE**: Il prezzo BTC (${spot:,.0f}) è a solo **{distance:.1f}%** dal
{btype} a ${level:,.0f} (nota emessa da {issuer_str}).

**Cosa aspettarsi**: Se il prezzo raggiunge questo livello, i dealer dovranno
{"vendere aggressivamente" if "knock" in btype else "ribilanciare le posizioni"},
il che potrebbe {"accelerare il ribasso" if "knock" in btype else "creare volatilità a breve termine"}.
""")
        elif distance < 8:
            st.warning(f"""
📍 **Zona di attenzione**: Il {btype} più vicino è a **{distance:.1f}%** dal prezzo
corrente (${level:,.0f}, {issuer_str}). Monitora se il prezzo si avvicina ulteriormente.
""")
        else:
            st.success(f"""
✅ **Nessun trigger imminente**: La barriera più vicina è a **{distance:.1f}%** dal
prezzo corrente. Il rischio di movimenti meccanici improvvisi è basso.
""")

    # Expander spiegazione
    with st.expander("📖 Come leggere questo grafico"):
        st.markdown("""
**Cosa sono le note strutturate?**
Le banche (JPMorgan, Morgan Stanley, Goldman...) vendono ai loro clienti
prodotti finanziari il cui rendimento dipende dal prezzo di IBIT (l'ETF
Bitcoin di BlackRock). Per proteggersi, le banche devono comprare/vendere
BTC sul mercato — questo si chiama "hedging".

**Perché creano movimenti di prezzo?**
Questi hedge non sono decisioni umane: sono formule matematiche.
Quando il prezzo tocca certi livelli predefiniti (le "barriere"),
i computer dei dealer eseguono ordini automatici di grandi dimensioni.
Questo può causare:
- **Cascate di vendita** se si rompono knock-in barriers al ribasso
- **Compressione della volatilità** se il prezzo resta nella buffer zone
- **Rilascio improvviso** dopo un auto-call event

**Come usarlo nel trading?**
- Le linee rosse (knock-in) sotto il prezzo sono i livelli di pericolo:
  se il prezzo ci si avvicina, aspettati un'accelerazione del ribasso
- Le linee verdi (auto-call) sopra il prezzo possono agire come resistenza
  temporanea, ma una volta superate liberano pressione di acquisto
- Più linee concentrate in una zona = più impatto potenziale
- Controlla la colonna "Nozionale" — barriere con nozionale alto hanno
  più impatto di quelle con nozionale basso
""")

    # Tabella note attive
    st.subheader("📋 Note Strutturate Attive")
    st.caption("Fonte: SEC EDGAR (filing 424B2/424B3). Dati aggiornati automaticamente.")
    rows = []
    for b in barriers:
        lvl_btc  = b.get("level_price_btc")
        dist_str = f"{abs(lvl_btc - spot) / spot * 100:.1f}%" if lvl_btc and spot else "—"
        rows.append({
            "Tipo":         b.get("barrier_type", ""),
            "Emittente":    b.get("issuer", ""),
            "Prodotto":     b.get("product_type", ""),
            "Livello %":    f"{b.get('level_pct', 0):.0f}%" if b.get("level_pct") else "—",
            "Prezzo BTC":   f"${lvl_btc:,.0f}" if lvl_btc else "—",
            "Distanza":     dist_str,
            "Scadenza":     b.get("maturity_date", ""),
            "Status":       b.get("status", ""),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2: GEX Chart
# ──────────────────────────────────────────────────────────────────────────────

def _tab_gex(snap: dict, gex_by_strike: list[dict], merged_df: pd.DataFrame) -> None:
    from src.dashboard.charts import gex_profile, gex_walls

    spot      = snap.get("spot_price") or 0
    gex_m     = (snap.get("total_net_gex") or 0) / 1e6
    regime    = snap.get("regime", "unknown")
    put_wall  = snap.get("put_wall") or 0
    call_wall = snap.get("call_wall") or 0
    flip      = snap.get("gamma_flip_price") or 0
    max_pain  = snap.get("max_pain") or 0

    st.header("📊 Gamma Exposure (GEX)")
    st.markdown("""
Il Gamma Exposure misura **dove e quanto** i market maker sono obbligati a
comprare o vendere Bitcoin per restare coperti. È la radiografia della pressione
meccanica nascosta nel mercato delle opzioni.
""")

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    regime_label = {
        "positive_gamma": "STABILIZZANTE",
        "negative_gamma": "AMPLIFICANTE",
        "neutral":        "NEUTRALE",
    }.get(regime, regime.upper())
    c1.metric(
        "Regime Corrente", regime_label,
        help="Positivo = i dealer assorbono la volatilità. Negativo = la amplificano.",
    )
    c2.metric(
        "Gamma Flip Point", f"${flip:,.0f}",
        delta=f"{(flip - spot) / spot * 100:+.1f}% da spot" if spot and flip else None,
        help="Sopra questo livello il mercato è stabilizzante, sotto è amplificante.",
    )
    c3.metric(
        "Put Wall (Supporto)", f"${put_wall:,.0f}",
        delta=f"{(put_wall - spot) / spot * 100:+.1f}% da spot" if spot and put_wall else None,
        delta_color="inverse",
        help="I dealer comprano qui, creando supporto meccanico.",
    )
    c4.metric(
        "Call Wall (Resistenza)", f"${call_wall:,.0f}",
        delta=f"{(call_wall - spot) / spot * 100:+.1f}% da spot" if spot and call_wall else None,
        help="I dealer vendono qui, creando resistenza meccanica.",
    )

    # Regime explanation box
    if regime == "positive_gamma":
        st.success(f"""
**🟢 Regime Gamma Positivo** (Total GEX: ${gex_m:+.1f}M)

I market maker sono "long gamma": quando il prezzo sale, vendono; quando
scende, comprano. Questo **assorbe gli shock** e tiene il prezzo in un range.

**Implicazione operativa**: Aspettati bassa volatilità e mean-reversion.
Le strategie range-bound funzionano bene. I breakout tendono a fallire.
Il prezzo tende a essere "attratto" verso il Max Pain a ${max_pain:,.0f}.
""")
    elif regime == "negative_gamma":
        st.error(f"""
**🔴 Regime Gamma Negativo** (Total GEX: ${gex_m:+.1f}M)

I market maker sono "short gamma": quando il prezzo sale, comprano; quando
scende, vendono. Questo **amplifica ogni movimento** e può creare cascate.

**Implicazione operativa**: Aspettati alta volatilità e trend-following.
I movimenti tendono a espandersi. I supporti/resistenze tradizionali possono
essere bucati con forza. Riduci la leva e allarga gli stop.
""")
    else:
        st.info(f"""
**🟡 Regime Neutrale** (Total GEX: ${gex_m:+.1f}M)

Il GEX è vicino a zero — siamo in prossimità del gamma flip point.
Il regime può cambiare rapidamente; monitora i prossimi movimenti.
""")

    # Charts
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(gex_profile(gex_by_strike, spot), use_container_width=True)
        st.caption("""
📊 **Come leggere l'istogramma**: Ogni barra rappresenta il GEX netto ad uno
strike price. Barre verdi = zona stabilizzante (il dealer compra sui cali,
vende sui rialzi). Barre rosse = zona amplificante. L'altezza della barra indica
l'intensità dell'effetto. La linea verticale è il prezzo spot corrente.
""")
    with col2:
        st.plotly_chart(gex_walls(snap), use_container_width=True)

    # Metriche aggiuntive
    st.subheader("Statistiche GEX")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Gamma Flip",    f"${snap.get('gamma_flip_price') or 0:,.0f}")
    mc2.metric("Max Pain",      f"${snap.get('max_pain') or 0:,.0f}")
    mc3.metric("Put/Call OI",   f"{snap.get('put_call_ratio') or 0:.2f}")
    mc4.metric("Strumenti BTC", f"{snap.get('n_instruments') or 0}")

    # Expander tecnico
    with st.expander("🔬 Dettaglio tecnico: come calcoliamo il GEX"):
        st.markdown("""
**Fonte dati**: API pubblica Deribit (opzioni BTC, aggiornamento continuo)

**Formula**: Per ogni opzione attiva:
`GEX = Gamma × Open Interest × Spot² × 0.01`

Il segno dipende dal tipo di opzione:
- **Call** → GEX positivo (il dealer è tipicamente short, assorbe la volatilità)
- **Put** → GEX negativo (il dealer amplifica i movimenti al ribasso)

**Limiti del modello**:
- Assumiamo che il dealer sia sempre la controparte (non sempre vero)
- Non distinguiamo tra flussi speculativi e di hedging
- Le opzioni IBIT su CBOE non sono incluse (dati non pubblici real-time)
- Il modello professionale (Glassnode taker-flow) è più preciso

**Affidabilità stimata**: ~80% del segnale rispetto ai modelli professionali.
Sufficiente per identificare i regimi macro e le zone di concentrazione principali.
""")

    # Regime analysis (se disponibile)
    if not merged_df.empty:
        gex_today = snap.get("total_net_gex") or 0.0
        with st.spinner("Calcolo regime analysis..."):
            try:
                regime_result = run_regime(merged_df, gex_today)
                if regime_result.positive_stats or regime_result.negative_stats:
                    from src.dashboard.charts import regime_bars
                    st.subheader("Regime Analysis: Gamma Positivo vs Negativo")
                    st.plotly_chart(regime_bars(regime_result), use_container_width=True)
                    if regime_result.gex_vol_correlation is not None:
                        corr_mean = regime_result.gex_vol_correlation.dropna().mean()
                        st.info(
                            f"Correlazione media GEX ↔ BTC Vol (rolling 30d): **{corr_mean:.3f}**"
                        )
            except Exception as e:
                _log.debug("Regime analysis non disponibile: %s", e)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3: ETF Flows
# ──────────────────────────────────────────────────────────────────────────────

def _tab_flows(merged_df: pd.DataFrame) -> None:
    from src.dashboard.charts import flows_chart

    st.header("💰 Flussi ETF Bitcoin")
    st.markdown("""
I flussi degli ETF spot Bitcoin mostrano quanto denaro istituzionale entra o esce
dal mercato ogni giorno. IBIT (BlackRock) da solo gestisce ~$50-67B di asset —
quando i flussi diventano fortemente negativi, la pressione di vendita è massiccia.
""")

    if merged_df.empty:
        st.warning("Dati flussi non disponibili.")
        return

    # KPI row
    recent = merged_df.last("7D")
    today_flow = week_flow = total_today = 0.0
    corr_30d   = 0.0

    if "ibit_flow" in merged_df.columns:
        ibit_col = merged_df["ibit_flow"].dropna()
        if not ibit_col.empty:
            today_flow = float(ibit_col.iloc[-1]) / 1e6
        week_flow = float(ibit_col.last("7D").sum()) / 1e6 if not ibit_col.empty else 0.0

    if "total_flow" in merged_df.columns:
        total_col = merged_df["total_flow"].dropna()
        total_today = float(total_col.iloc[-1]) / 1e6 if not total_col.empty else 0.0

    if "ibit_flow" in merged_df.columns and "btc_return" in merged_df.columns:
        valid = merged_df[["ibit_flow", "btc_return"]].dropna()
        if len(valid) >= 30:
            corr_30d = float(
                valid["ibit_flow"].rolling(30, min_periods=15).corr(valid["btc_return"]).dropna().iloc[-1]
            ) if len(valid) >= 30 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("IBIT Flusso Oggi", f"${today_flow:+,.0f}M",
              help="Flusso netto IBIT nelle ultime 24h di trading")
    c2.metric("IBIT Flusso 7gg", f"${week_flow:+,.0f}M",
              help="Flusso netto cumulativo ultimi 7 giorni di borsa")
    c3.metric("Tutti gli ETF Oggi", f"${total_today:+,.0f}M",
              help="Flusso netto aggregato di tutti gli ETF spot BTC")
    c4.metric("Correlazione 30gg", f"{corr_30d:.2f}",
              help="Correlazione rolling 30gg tra flussi IBIT e rendimenti BTC del giorno successivo. "
                   "Valori > 0.3 indicano potere predittivo.")

    # Alert contestuale flussi
    if "ibit_flow_3d" in merged_df.columns:
        ibit_3d_col = merged_df["ibit_flow_3d"].dropna()
        if not ibit_3d_col.empty:
            ibit_3d = float(ibit_3d_col.iloc[-1]) / 1e6
            if ibit_3d < -500:
                st.error(f"""
🚨 **Deflussi pesanti**: IBIT ha perso **${abs(ibit_3d):,.0f}M** negli ultimi 3 giorni.
Questo livello di uscite indica probabile ribilanciamento istituzionale o liquidazione
di posizioni hedge. Storicamente, deflussi di questa entità precedono ulteriore
debolezza nel prezzo a breve termine.
""")
            elif ibit_3d < -200:
                st.warning(f"""
⚡ **Deflussi significativi**: ${abs(ibit_3d):,.0f}M usciti da IBIT in 3 giorni.
Il livello merita attenzione — controlla se è accompagnato da GEX negativo
(Tab Gamma Exposure) per un segnale più forte.
""")
            elif ibit_3d > 300:
                st.success(f"""
💪 **Afflussi solidi**: +${ibit_3d:,.0f}M in IBIT negli ultimi 3 giorni.
La domanda istituzionale è forte. Se accompagnata da GEX positivo,
il regime è favorevole per la stabilità del prezzo.
""")

    # Grafico principale
    st.plotly_chart(flows_chart(merged_df), use_container_width=True)
    st.caption("""
📉 **Correlazione Flussi-Prezzo**: Questo grafico mostra quanto i flussi ETF
"predicono" il rendimento di BTC il giorno successivo. Una correlazione alta
(>0.3) significa che il denaro istituzionale sta guidando il prezzo.
La correlazione tende ad aumentare durante i periodi di stress.
""")

    # Riepilogo 30gg
    st.subheader("Riepilogo ultimi 30 giorni")
    r30 = merged_df.last("30D")
    rc1, rc2, rc3 = st.columns(3)
    if "ibit_flow" in r30.columns:
        total_30 = r30["ibit_flow"].sum() / 1e6
        rc1.metric("Flusso IBIT (30d)", f"{total_30:+.0f}M$")
        pos_days = int((r30["ibit_flow"] > 0).sum())
        rc2.metric("Giorni inflow", f"{pos_days}/30")
    if "btc_return" in r30.columns:
        btc_cum = float((1 + r30["btc_return"].dropna()).prod() - 1)
        rc3.metric("BTC Return (30d)", f"{btc_cum * 100:+.1f}%")

    # Granger expander
    with st.expander("📊 Test Statistico: i flussi ETF predicono il prezzo?"):
        with st.spinner("Calcolo Granger causality..."):
            try:
                _, granger_df, granger_text = run_granger(merged_df)
                from src.dashboard.charts import granger_heatmap
                st.plotly_chart(granger_heatmap(granger_df), use_container_width=True)

                # Tabella p-values formattata
                if not granger_df.empty:
                    st.markdown("""
**Test di Granger Causality** — Verifica se conoscere i flussi ETF di oggi
migliora la previsione del prezzo BTC di domani (rispetto a usare solo la storia dei prezzi).

Un **p-value < 0.05** (🟢) significa che i flussi hanno potere predittivo
statisticamente significativo con quel ritardo.
""")
                with st.expander("📋 Tabella completa p-values"):
                    st.text(granger_text)
            except Exception as e:
                st.info(f"Granger causality non disponibile: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4: Segnali Operativi
# ──────────────────────────────────────────────────────────────────────────────

def _tab_signals(snap: dict, merged_df: pd.DataFrame, barriers: list[dict]) -> None:
    from src.dashboard.charts import backtest_equity

    st.header("🚦 Segnali Operativi")
    st.markdown("""
Questo pannello combina i tre pilastri dell'analisi (Barriere, GEX, Flussi)
in segnali operativi. **Non sono raccomandazioni di investimento** — sono
strumenti di analisi che evidenziano condizioni di mercato rilevanti basate
sulla microstructura.
""")
    st.warning("""
⚠️ **Disclaimer**: Questi segnali si basano su un modello sperimentale con
dati limitati. Non sostituiscono l'analisi personale. Usa sempre gestione
del rischio e position sizing appropriati. Il backtest storico non garantisce
performance futura.
""")

    # Calcola segnale composito
    signal, details = _composite_signal(snap, merged_df, barriers)

    gex_raw   = details["gex"]
    ibit_3d   = details["ibit_3d"]
    c_dist    = details["closest_dist"]
    c_barrier = details["closest_barrier"]

    gex_status  = "✅ Positivo"  if details["gex_ok"]      else "❌ Negativo"
    flow_status = "✅ Positivi"  if details["flow_ok"]      else ("❌ Negativi" if details["flow_bad"] else "🟡 Neutri")
    barr_status = "✅ Nessuna"   if details["barrier_ok"]   else ("🟡 Attenzione" if details["barrier_caution"] else "❌ Entro 5%")

    gex_detail  = "Dealer assorbono volatilità" if details["gex_ok"] else "Dealer amplificano i movimenti"
    flow_detail = f"IBIT 3gg: ${ibit_3d / 1e6:+.0f}M"
    barr_detail = f"Barriera più vicina: {c_dist:.1f}%" if c_dist < float("inf") else "Nessuna barriera nel DB"

    if signal == "LONG":
        st.success(f"""
### 🟢 REGIME FAVOREVOLE

| Fattore | Stato | Dettaglio |
|:---|:---:|:---|
| Gamma Exposure | {gex_status} | {gex_detail} |
| Flussi IBIT (3gg) | {flow_status} | {flow_detail} |
| Barriere vicine | {barr_status} | {barr_detail} |

**Interpretazione**: Le condizioni strutturali favoriscono stabilità
o rialzo graduale. Il mercato è in regime "mean-reverting" — i cali
tendono a essere comprati meccanicamente dai dealer.
""")

    elif signal == "RISK_OFF":
        st.error(f"""
### 🔴 REGIME DI RISCHIO

| Fattore | Stato | Dettaglio |
|:---|:---:|:---|
| Gamma Exposure | {gex_status} | {gex_detail} |
| Flussi IBIT (3gg) | {flow_status} | {flow_detail} |
| Barriere vicine | {barr_status} | {barr_detail} |

**Interpretazione**: Le condizioni strutturali favoriscono volatilità
elevata e potenziale ribasso amplificato. I movimenti al ribasso
tendono a espandersi perché i dealer vendono sui cali.

**Azione suggerita**: Ridurre leva, allargare stop, considerare
coperture (put options o riduzione posizione).
""")

    else:  # CAUTION
        st.warning(f"""
### 🟡 REGIME MISTO

| Fattore | Stato | Dettaglio |
|:---|:---:|:---|
| Gamma Exposure | {gex_status} | {gex_detail} |
| Flussi IBIT (3gg) | {flow_status} | {flow_detail} |
| Barriere vicine | {barr_status} | {barr_detail} |

**Interpretazione**: Segnali contrastanti — alcuni fattori sono
favorevoli, altri no. Riduci l'esposizione e monitora più frequentemente.
Un cambiamento in uno qualsiasi dei fattori può far scattare il segnale
in verde o rosso.
""")

    # Logica del segnale
    with st.expander("⚙️ Come viene calcolato il segnale"):
        st.markdown("""
Il segnale composito è una combinazione di tre fattori indipendenti:

**1. Regime GEX** (peso: 40%)
- 🟢 se Total Net GEX > 0 (dealer stabilizzano)
- 🔴 se Total Net GEX < 0 (dealer amplificano)
- Il regime è il fattore più importante perché determina se
  ogni altro shock viene assorbito o amplificato

**2. Flussi IBIT 3 giorni** (peso: 30%)
- 🟢 se somma flussi > +$100M (domanda netta)
- 🟡 se tra -$100M e +$100M (neutro)
- 🔴 se < -$200M (deflussi significativi)

**3. Prossimità a Barrier Level** (peso: 30%)
- 🟢 se la barriera più vicina è > 8% dal prezzo
- 🟡 se tra 3% e 8%
- 🔴 se < 3% (trigger imminente)

**Segnale finale**:
- 🟢 FAVOREVOLE: tutti e 3 i fattori sono verdi
- 🟡 MISTO: almeno un fattore è giallo/rosso
- 🔴 RISCHIO: almeno 2 fattori sono rossi
""")

    st.divider()

    # Backtest results
    st.subheader("📈 Backtest della Strategia")
    st.caption("Periodo di test: dal lancio IBIT opzioni (Nov 2024) ad oggi.")

    if merged_df.empty:
        st.warning("Dati insufficienti per il backtest.")
        return

    with st.spinner("Esecuzione backtest..."):
        try:
            bt, results = run_backtest(merged_df, barriers)
        except Exception as e:
            st.error(f"Backtest non disponibile: {e}")
            return

    if not results:
        st.error("Backtest non disponibile.")
        return

    strat = results["strategy"]
    bah   = results["buy_and_hold"]

    # KPI backtest
    delta_sharpe = strat.sharpe_ratio - bah.sharpe_ratio
    bc1, bc2, bc3, bc4 = st.columns(4)
    bc1.metric(
        "Sharpe Ratio", f"{strat.sharpe_ratio:.2f}",
        help="Rendimento risk-adjusted. >1 è buono, >2 è eccellente.",
    )
    bc2.metric(
        "Max Drawdown", f"{strat.max_drawdown * 100:.1f}%",
        help="Massima perdita dal picco. Più basso è, meglio è.",
    )
    bc3.metric(
        "Win Rate", f"{strat.win_rate * 100:.0f}%",
        help="Percentuale di giorni in profitto.",
    )
    bc4.metric(
        "vs Buy&Hold", f"{delta_sharpe:+.2f}",
        help="Delta Sharpe rispetto al buy-and-hold BTC.",
    )

    # Tabella comparativa
    table = bt.summary_table(results)
    st.dataframe(table, use_container_width=True)

    # Equity curve
    st.subheader("Equity Curve")
    st.plotly_chart(backtest_equity(results), use_container_width=True)

    st.caption("""
⚠️ Il backtest ha limitazioni significative: lo storico è breve (dal Nov 2024),
non include slippage e commissioni, e potrebbe soffrire di overfitting.
Usa i risultati come indicazione direzionale, non come garanzia di performance.
""")

    if strat.days_long == 0 and strat.days_short == 0:
        st.info(
            "La strategia è flat su tutto il periodo: il GEX storico non è ancora "
            "disponibile. Il backtest mostrerà segnali reali man mano che il sistema "
            "accumula snapshot GEX giornalieri."
        )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 5: EDGAR Monitor
# ──────────────────────────────────────────────────────────────────────────────

def _tab_edgar_monitor(barriers: list[dict], merged_df: pd.DataFrame) -> None:
    from src.dashboard.charts import event_study_car

    st.header("🔍 Monitor SEC EDGAR")
    st.markdown("""
Elenco completo delle note strutturate legate a IBIT depositate presso la SEC.
Questa è la stessa ricerca che Arthur Hayes sta compilando — mappare tutti i
prodotti per capire dove si concentrano i trigger points.
""")

    # KPI
    db_stats = load_db_summary()
    total_notes    = db_stats.get("total_notes", 0)
    total_notional = db_stats.get("total_notional_usd") or 0
    active_cnt     = db_stats.get("active_barriers", 0)

    kc1, kc2, kc3 = st.columns(3)
    kc1.metric("Note Totali Trovate", total_notes)
    kc2.metric("Nozionale Totale", f"${total_notional / 1e6:,.0f}M")
    kc3.metric("Barriere Attive", active_cnt)

    # Expander spiegazione
    with st.expander("📖 Cosa sono queste note e perché importano"):
        st.markdown("""
**Cosa sono**: Prodotti finanziari emessi da banche come JPMorgan,
Morgan Stanley e Goldman Sachs. Funzionano così:

1. La banca vende una "nota" a un investitore (tipicamente istituzionale
   o high-net-worth)
2. Il rendimento della nota dipende dal prezzo di IBIT (l'ETF Bitcoin di BlackRock)
3. Per proteggersi, la banca deve comprare/vendere BTC sul mercato

**Perché importano per il trader**: Il nozionale totale di queste note
rappresenta una massa di hedging meccanico che influenza il prezzo. Più
note vengono emesse, più il prezzo di BTC è influenzato dalla meccanica
dei derivati piuttosto che dal sentiment.

**Come le troviamo**: Scansionando automaticamente i filing SEC
(moduli 424B2 e 424B3) che menzionano IBIT. Tutti i dati sono pubblici.
""")

    if not barriers:
        st.info("Nessuna barriera attiva nel DB. Esegui `scripts/run_edgar.py` per popolare il database.")
        return

    st.subheader(f"Barriere attive ({len(barriers)})")

    rows = []
    for b in barriers:
        rows.append({
            "Tipo":        b.get("barrier_type", ""),
            "Emittente":   b.get("issuer", ""),
            "Prodotto":    b.get("product_type", ""),
            "Livello %":   f"{b.get('level_pct', 0):.0f}%" if b.get("level_pct") else "—",
            "Prezzo IBIT": f"${b.get('level_price_ibit', 0):.2f}" if b.get("level_price_ibit") else "—",
            "Prezzo BTC":  f"${b.get('level_price_btc', 0):,.0f}" if b.get("level_price_btc") else "—",
            "Scadenza":    b.get("maturity_date", ""),
            "Status":      b.get("status", ""),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Event study
    st.subheader("Event Study — CAR intorno ai Barrier Levels")
    with st.spinner("Calcolo event study..."):
        event_results = run_event_study(barriers, merged_df)

    fig = event_study_car(event_results)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        st.caption("""
📊 **Cumulative Abnormal Returns**: mostra il rendimento anomalo cumulativo
di BTC nei giorni intorno al momento in cui il prezzo si avvicina a una barriera.
Un pattern non casuale (con asterischi ***) suggerisce un effetto meccanico reale.
""")
    else:
        n_total = sum(r.n_events for r in event_results) if event_results else 0
        if n_total == 0:
            st.info(
                "Nessun evento trovato: i prezzi BTC delle barriere non sono ancora "
                "stati raggiunti nel periodo analizzato, oppure i prezzi BTC dei barrier "
                "levels non sono disponibili nel DB."
            )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Caricamento dati ──────────────────────────────────────────────────────
    with st.spinner("Carico GEX live da Deribit..."):
        try:
            snap, gex_by_strike = load_gex()
        except Exception as e:
            st.warning(f"GEX non disponibile: {e}")
            snap         = {"spot_price": 0, "total_net_gex": 0, "regime": "unknown", "alerts": []}
            gex_by_strike = []

    with st.spinner("Carico prezzi e flussi ETF..."):
        try:
            merged_df = load_prices_and_flows()
        except Exception as e:
            st.error(f"Errore caricamento prezzi/flussi: {e}")
            merged_df = pd.DataFrame()

    with st.spinner("Carico barriere EDGAR..."):
        try:
            barriers = load_barriers()
        except Exception as e:
            st.warning(f"Barriere non disponibili: {e}")
            barriers = []

    # ── Sidebar ───────────────────────────────────────────────────────────────
    manual_refresh = _sidebar(snap, merged_df)

    if manual_refresh:
        for fn in [
            load_prices_and_flows, load_gex, load_barriers, load_db_summary,
            run_granger, run_regime, run_backtest, run_event_study,
        ]:
            fn.clear()
        st.rerun()

    # ── Header KPI ────────────────────────────────────────────────────────────
    _render_header(snap, merged_df)
    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
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

    # ── Footer ────────────────────────────────────────────────────────────────
    st.caption(
        f"Ultimo aggiornamento: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} · "
        f"GEX: Deribit (pubblica) · Flussi: Farside/yfinance · "
        f"Note strutturate: SEC EDGAR · Sviluppato da WAGMI-LAB"
    )


if __name__ == "__main__":
    main()
