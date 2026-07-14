"""TAB 5: EDGAR Monitor."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.charts import event_study_car
from src.dashboard.data_loader import load_db_summary, run_event_study


def _tab_edgar_monitor(barriers: list[dict], merged_df: pd.DataFrame) -> None:
    st.header("🔍 Monitor SEC EDGAR")
    st.markdown("""
Elenco completo delle note strutturate legate a IBIT depositate presso la SEC.
Questa è la stessa ricerca che Arthur Hayes sta compilando — mappare tutti i
prodotti per capire dove si concentrano i trigger points.
""")

    db_stats = load_db_summary()
    total_notes = db_stats.get("total_notes", 0)
    total_notional = db_stats.get("total_notional_usd") or 0
    active_cnt = db_stats.get("active_barriers", 0)

    kc1, kc2, kc3 = st.columns(3)
    kc1.metric("Note Totali Trovate", total_notes)
    kc2.metric("Nozionale Totale", f"${total_notional / 1e6:,.0f}M")
    kc3.metric("Barriere Attive", active_cnt)

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
        st.info(
            "Nessuna barriera attiva nel DB. Esegui `scripts/run_edgar.py` per popolare il database."
        )
        return

    st.subheader(f"Barriere attive ({len(barriers)})")

    rows = []
    for b in barriers:
        rows.append({
            "Tipo": b.get("barrier_type", ""),
            "Emittente": b.get("issuer", ""),
            "Prodotto": b.get("product_type", ""),
            "Livello %": f"{b.get('level_pct', 0):.0f}%" if b.get("level_pct") else "—",
            "Prezzo IBIT": f"${b.get('level_price_ibit', 0):.2f}" if b.get("level_price_ibit") else "—",
            "Prezzo BTC": f"${b.get('level_price_btc', 0):,.0f}" if b.get("level_price_btc") else "—",
            "Scadenza": b.get("maturity_date", ""),
            "Status": b.get("status", ""),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch")

    st.subheader("Event Study — CAR intorno ai Barrier Levels")
    with st.spinner("Calcolo event study..."):
        event_results = run_event_study(barriers, merged_df)

    fig = event_study_car(event_results)
    if fig:
        st.plotly_chart(fig, width="stretch")
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
