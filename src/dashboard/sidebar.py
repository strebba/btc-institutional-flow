from __future__ import annotations

import pandas as pd
import streamlit as st

from src.assets import ASSETS, DEFAULT_ASSET, AssetSpec, get_asset


def _asset_selector() -> AssetSpec:
    """Selettore BTC/ETH in cima alla sidebar, sincronizzato con ``?asset=`` nell'URL.

    Va chiamato prima di caricare i dati: tutto il resto della dashboard legge
    l'asset scelto. ``bind="query-params"`` rende il link condivisibile (utile
    per l'embed su wagmi-lab.com) e ignora valori sconosciuti nell'URL.
    """
    with st.sidebar:
        st.markdown("**⚡ IBIT Gamma Tracker**")
        st.caption("WAGMI-LAB Research Tool")
        scelto = st.segmented_control(
            "Asset",
            list(ASSETS),
            default=DEFAULT_ASSET,
            required=True,
            key="asset",
            bind="query-params",
            width="stretch",
        )
    return get_asset(scelto or DEFAULT_ASSET)


def _sidebar(
    snap: dict, merged_df: pd.DataFrame, barriers: list[dict], spec: AssetSpec | None = None
) -> bool:
    """Renderizza la sidebar: refresh, stato dati e fonti (brand e asset li mette ``_asset_selector``).

    Restituisce True se l'utente richiede un refresh manuale.
    """
    spec = spec or get_asset("BTC")
    with st.sidebar:
        refresh = st.button(
            "Aggiorna dati",
            icon=":material/refresh:",
            type="primary",
            width="stretch",
        )

        st.divider()

        st.markdown("**Stato dati**")
        spot_price = snap.get("spot_price") or 0
        gex_ok = snap.get("total_net_gex", 0) != 0 or snap.get("n_instruments", 0) > 0
        flows_ok = not merged_df.empty
        barriers_ok = len(barriers) > 0

        def _status(ok: bool) -> str:
            return ":green-badge[OK]" if ok else ":red-badge[—]"

        stato = f"GEX {_status(gex_ok)} · Flussi {_status(flows_ok)}"
        if spec.has("barriers"):
            stato += f" · Barriere {_status(barriers_ok)}"
        st.markdown(stato)
        st.caption(f"{spec.key}: ${spot_price:,.0f}")

        st.divider()

        if spec.has("edgar"):
            st.caption(
                "**Fonti** — GEX: Deribit · Flussi: Farside/yfinance · "
                "Note: SEC EDGAR · Prezzi: Yahoo Finance"
            )
        else:
            st.caption(
                "**Fonti** — GEX: Deribit · Flussi: CoinGlass/Farside · "
                "Derivati: CoinGlass/CoinGecko · Prezzi: Yahoo Finance"
            )
        st.caption("Non costituisce consulenza finanziaria.")

    return refresh
