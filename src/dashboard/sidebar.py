from __future__ import annotations

import pandas as pd
import streamlit as st


def _sidebar(snap: dict, merged_df: pd.DataFrame, barriers: list[dict]) -> bool:
    """Renderizza la sidebar: brand, refresh, stato dati e fonti.

    Restituisce True se l'utente richiede un refresh manuale.
    """
    with st.sidebar:
        st.markdown("**⚡ IBIT Gamma Tracker**")
        st.caption("WAGMI-LAB Research Tool")

        refresh = st.button(
            "Aggiorna dati",
            icon=":material/refresh:",
            type="primary",
            width="stretch",
        )

        st.divider()

        st.markdown("**Stato dati**")
        btc_price = snap.get("spot_price") or 0
        gex_ok = snap.get("total_net_gex", 0) != 0 or snap.get("n_instruments", 0) > 0
        flows_ok = not merged_df.empty
        barriers_ok = len(barriers) > 0

        def _status(ok: bool) -> str:
            return ":green-badge[OK]" if ok else ":red-badge[—]"

        st.markdown(
            f"GEX {_status(gex_ok)} · Flussi {_status(flows_ok)} · Barriere {_status(barriers_ok)}"
        )
        st.caption(f"BTC: ${btc_price:,.0f}")

        st.divider()

        st.caption(
            "**Fonti** — GEX: Deribit · Flussi: Farside/yfinance · "
            "Note: SEC EDGAR · Prezzi: Yahoo Finance"
        )
        st.caption("Non costituisce consulenza finanziaria.")

    return refresh
