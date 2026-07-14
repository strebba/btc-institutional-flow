from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import get_settings

_settings = get_settings()
_theme = _settings["dashboard"]["theme"]
_TEXT_MUTED = _theme.get("text_muted", "#8b949e")

def _sidebar(snap: dict, merged_df: pd.DataFrame, barriers: list[dict]) -> bool:
    """Renderizza sidebar. Restituisce True se si richiede refresh manuale."""
    with st.sidebar:
        st.markdown(
            f"""
<div style="display:flex;align-items:center;gap:8px;margin-bottom:0.15rem">
  <span style="font-size:1rem;font-weight:700;color:{_theme["text"]}">
    ⚡ IBIT Gamma Tracker
  </span>
</div>
<p style="color:{_TEXT_MUTED};font-size:0.72rem;margin:0 0 0.5rem">
  WAGMI-LAB Research Tool v1.0
</p>
""",
            unsafe_allow_html=True,
        )

        refresh = st.button("🔄 Aggiorna dati", width="stretch")

        st.divider()

        st.markdown(
            f"""
<p style="font-size:10px;font-weight:700;text-transform:uppercase;
          letter-spacing:0.07em;color:{_TEXT_MUTED};margin:0 0 0.4rem">
  Stato dati
</p>
""",
            unsafe_allow_html=True,
        )

        btc_price = snap.get("spot_price") or 0
        ts_now = pd.Timestamp.now().strftime("%H:%M:%S")
        gex_ok = snap.get("total_net_gex", 0) != 0 or snap.get("n_instruments", 0) > 0
        flows_ok = not merged_df.empty
        barriers_ok = len(barriers) > 0

        def _status(ok: bool) -> str:
            return f'<span style="color:{_theme["positive"]}">OK</span>' if ok else \
                   f'<span style="color:{_theme["negative"]}">—</span>'

        st.markdown(
            f"""
<small style="color:{_TEXT_MUTED};line-height:1.6">
  GEX ({ts_now}) {_status(gex_ok)}<br>
  Flussi {_status(flows_ok)}<br>
  Barriere {_status(barriers_ok)}<br>
  BTC: ${btc_price:,.0f}
</small>
""",
            unsafe_allow_html=True,
        )

        st.divider()

        cfg = _settings.get("backtest", {})
        st.markdown(
            f"""
<p style="font-size:10px;font-weight:700;text-transform:uppercase;
          letter-spacing:0.07em;color:{_TEXT_MUTED};margin:0 0 0.5rem">
  Soglie Backtest
</p>
""",
            unsafe_allow_html=True,
        )
        st.metric("Long GEX threshold", "$0")
        st.metric("Long Flow (3d)", f"+{cfg.get('long_flow_threshold_usd_m', 100)}M")
        st.metric("Short Flow (3d)", f"{cfg.get('short_flow_threshold_usd_m', -200)}M")
        st.metric("Barrier exclusion", f"±{cfg.get('barrier_exclusion_pct', 5)}%")

        st.divider()

        st.markdown(
            f"""
<div style="color:{_TEXT_MUTED};font-size:0.72rem;line-height:1.7">
  <span style="color:{_theme["positive"]};font-weight:700">ibit-gamma-tracker</span> v1.0<br>
  Deribit · EDGAR · yfinance<br><br>
  <b>Fonti dati</b><br>
  • GEX: Deribit API pubblica<br>
  • Flussi: Farside Investors<br>
  • Note strutturate: SEC EDGAR<br>
  • Prezzi: Yahoo Finance<br><br>
  <em>Non costituisce consulenza finanziaria.</em>
</div>
""",
            unsafe_allow_html=True,
        )

    return refresh
