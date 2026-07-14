from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import get_settings

_settings = get_settings()
_theme = _settings["dashboard"]["theme"]
_TEXT_MUTED = _theme.get("text_muted", "#888888")

def _render_header(snap: dict, merged_df: pd.DataFrame) -> None:
    regime = snap.get("regime", "unknown")
    color_map = {
        "positive_gamma": _theme["positive"],
        "negative_gamma": _theme["negative"],
        "neutral": _theme["neutral"],
    }
    regime_color = color_map.get(regime, _theme["text"])

    st.markdown(
        f"""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:0.2rem">
  <span style="font-size:1.75rem;font-weight:700;letter-spacing:-0.02em;color:{_theme["text"]}">
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
""",
        unsafe_allow_html=True,
    )

    spot = snap.get("spot_price") or 0
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
        ret_val = float(last_ret.iloc[-1]) * 100 if not last_ret.empty else 0
        col5.metric("BTC Return (ieri)", f"{ret_val:+.2f}%")
    else:
        col5.metric("Gamma Flip", f"${snap.get('gamma_flip_price') or 0:,.0f}")

    alerts = snap.get("alerts", [])
    if alerts:
        for alert in alerts:
            st.warning(f"⚠️ {alert}")
