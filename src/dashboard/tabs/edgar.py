"""Pagina EDGAR Monitor: note strutturate IBIT depositate alla SEC."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.charts import event_study_car
from src.dashboard.data_loader import load_db_summary, run_event_study


def _tab_edgar_monitor(barriers: list[dict], merged_df: pd.DataFrame) -> None:
    st.header("Monitor SEC EDGAR", icon=":material/search:")
    st.caption(
        "Note strutturate su IBIT depositate alla SEC: dove si concentrano i "
        "trigger point dell'hedging meccanico."
    )

    db_stats = load_db_summary()
    total_notes = db_stats.get("total_notes", 0)
    total_notional = db_stats.get("total_notional_usd") or 0
    active_cnt = db_stats.get("active_barriers", 0)

    with st.container(horizontal=True):
        st.metric("Note totali", total_notes, border=True)
        st.metric("Nozionale totale", f"${total_notional / 1e6:,.0f}M", border=True)
        st.metric("Barriere attive", active_cnt, border=True)

    if not barriers:
        st.info("Nessuna barriera attiva nel DB. Esegui `scripts/run_edgar.py`.")
        return

    with st.expander(f"Barriere attive ({len(barriers)})", icon=":material/table_chart:"):
        rows = []
        for b in barriers:
            rows.append(
                {
                    "Tipo": b.get("barrier_type", ""),
                    "Emittente": b.get("issuer", ""),
                    "Prodotto": b.get("product_type", ""),
                    "Livello %": b.get("level_pct"),
                    "Prezzo IBIT": b.get("level_price_ibit"),
                    "Prezzo BTC": b.get("level_price_btc"),
                    "Scadenza": b.get("maturity_date", ""),
                    "Status": b.get("status", ""),
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
            column_config={
                "Tipo": st.column_config.TextColumn("Tipo"),
                "Emittente": st.column_config.TextColumn("Emittente"),
                "Prodotto": st.column_config.TextColumn("Prodotto"),
                "Livello %": st.column_config.NumberColumn("Livello %", format="%.0f%%"),
                "Prezzo IBIT": st.column_config.NumberColumn("Prezzo IBIT", format="$%.2f"),
                "Prezzo BTC": st.column_config.NumberColumn("Prezzo BTC", format="$%,.0f"),
                "Scadenza": st.column_config.DateColumn("Scadenza"),
                "Status": st.column_config.TextColumn("Status"),
            },
        )

    st.subheader("Event study — CAR intorno alle barriere")
    with st.spinner("Calcolo event study..."):
        event_results = run_event_study(barriers, merged_df)

    fig = event_study_car(event_results)
    if fig:
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Rendimento anomalo cumulativo di BTC nei giorni intorno all'avvicinamento "
            "a una barriera. Un pattern non casuale (***) suggerisce un effetto reale."
        )
    else:
        n_total = sum(r.n_events for r in event_results) if event_results else 0
        if n_total == 0:
            st.info(
                "Nessun evento trovato: i prezzi BTC delle barriere non sono stati "
                "raggiunti nel periodo analizzato."
            )
