from __future__ import annotations

import pandas as pd
import streamlit as st


from src.dashboard.charts import barrier_map as _barrier_map_chart, barrier_gex_confluence_chart

from src.edgar.barrier_utils import compute_confluence, detect_clusters

_TYPE_INFO = {
    "knock_in": ("Rottura al ribasso → dealer vendono aggressivamente.", "red"),
    "autocall": ("Superamento rialzo → rimborso nota + chiusura hedge.", "green"),
    "buffer": ("Protezione parziale, hedging graduale.", "blue"),
    "knock_out": ("Estinzione automatica al superamento.", "orange"),
}


def _tab_barrier_map(barriers: list[dict], snap: dict) -> None:
    spot = snap.get("spot_price") or 0.0

    st.header("Mappa dei livelli critici", icon=":material/sell:")
    st.caption(
        "Livelli di prezzo dove le banche emittenti di note strutturate su IBIT sono "
        "obbligate a comprare o vendere BTC per ribilanciare l'hedge."
    )

    type_counts: dict[str, int] = {}
    type_notional: dict[str, float] = {}
    for b in barriers:
        t = b.get("barrier_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
        type_notional[t] = type_notional.get(t, 0) + (b.get("notional_usd") or 0)

    if not any(type_counts.get(t, 0) > 0 for t in _TYPE_INFO):
        if not barriers:
            st.info(
                "Nessuna barriera attiva nel DB. Esegui `scripts/run_edgar.py` per "
                "popolare il database."
            )
        else:
            st.info(
                "Nessuna barriera con prezzo BTC calcolabile: esegui `make update-edgar` "
                "per ricalcolare i prezzi."
            )
        return

    # Legenda condizionale (conteggi reali)
    legend_cols = st.columns(len([t for t in _TYPE_INFO if type_counts.get(t, 0) > 0]))
    idx = 0
    for t, (desc, color) in _TYPE_INFO.items():
        n = type_counts.get(t, 0)
        if n == 0:
            continue
        noz = type_notional.get(t, 0) / 1e6
        with legend_cols[idx]:
            st.markdown(f":{color}-badge[{t.upper()}]")
            st.caption(f"{n} barriere · ${noz:,.0f}M")
            st.caption(desc)
        idx += 1

    st.plotly_chart(_barrier_map_chart(barriers, spot), width="stretch")

    _proximity_alert(barriers, spot)

    valid_barriers = [b for b in barriers if (b.get("level_price_btc") or 0) > 0 and spot > 0]
    if valid_barriers:
        clusters = detect_clusters(barriers, spot)
        confluence = compute_confluence(
            clusters,
            put_wall=snap.get("put_wall"),
            call_wall=snap.get("call_wall"),
            gamma_flip=snap.get("gamma_flip_price"),
        )
        if clusters:
            st.subheader("Confluenza barriere ↔ GEX")
            st.caption(
                "Un cluster di barriere che coincide con un wall di gamma raddoppia "
                "l'effetto meccanico allo stesso prezzo."
            )
            st.plotly_chart(
                barrier_gex_confluence_chart(clusters, snap, spot, confluence),
                width="stretch",
            )
            for c in confluence or []:
                ctype = c.get("confluence_type")
                px = c.get("gex_level_price", 0)
                notional = c.get("cluster_notional_usd", 0)
                if ctype == "bearish_reinforced":
                    st.error(
                        f"Ribasso rinforzato a ${px:,.0f} (nozionale ${notional:,.0f}).",
                        icon=":material/error:",
                    )
                elif ctype == "bullish_reinforced":
                    st.success(
                        f"Rialzo rinforzato a ${px:,.0f} (nozionale ${notional:,.0f}).",
                        icon=":material/check_circle:",
                    )

    with st.expander("Come leggere questa mappa", icon=":material/menu_book:"):
        st.markdown(
            "Le banche vendono note il cui rendimento dipende da IBIT; per coprirsi "
            "compravendono BTC con formule matematiche, non decisioni umane. Quando il "
            "prezzo tocca una barriera partono ordini automatici di grandi dimensioni.\n\n"
            "- **Rosse (knock-in)** sotto il prezzo = livelli di pericolo.\n"
            "- **Verdi (auto-call)** sopra = resistenza temporanea.\n"
            "- Più linee concentrate in una zona = più impatto.\n"
            "- Nozionale alto = impatto maggiore."
        )

    with st.expander(f"Note strutturate attive ({len(barriers)})", icon=":material/receipt_long:"):
        st.caption("Fonte: SEC EDGAR (filing 424B2/424B3).")
        _barriers_table(barriers, spot)


def _proximity_alert(barriers: list[dict], spot: float) -> None:
    valid = [b for b in barriers if (b.get("level_price_btc") or 0) > 0 and spot > 0]
    if not valid:
        return
    closest = min(valid, key=lambda b: abs((b.get("level_price_btc") or 0) - spot))
    level = closest.get("level_price_btc", 0)
    distance = abs(spot - level) / spot * 100
    btype = closest.get("barrier_type", "barrier")
    issuer = closest.get("issuer", "N/A")

    if distance < 3:
        st.error(
            f"Prezzo a {distance:.1f}% dal {btype} di {issuer} (${level:,.0f}).",
            icon=":material/error:",
        )
    elif distance < 8:
        st.warning(
            f"{btype} più vicino a {distance:.1f}% (${level:,.0f}, {issuer}).",
            icon=":material/warning:",
        )
    else:
        st.success(
            f"Barriera più vicina a {distance:.1f}% dal prezzo.",
            icon=":material/check_circle:",
        )


def _barriers_table(barriers: list[dict], spot: float) -> None:
    rows = []
    for b in barriers:
        lvl_btc = b.get("level_price_btc")
        rows.append(
            {
                "Tipo": b.get("barrier_type", ""),
                "Emittente": b.get("issuer", ""),
                "Prodotto": b.get("product_type", ""),
                "Livello %": b.get("level_pct"),
                "Prezzo BTC": lvl_btc,
                "Distanza %": (abs(lvl_btc - spot) / spot * 100) if lvl_btc and spot else None,
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
            "Prezzo BTC": st.column_config.NumberColumn("Prezzo BTC", format="$%,.0f"),
            "Distanza %": st.column_config.NumberColumn("Distanza", format="%.1f%%"),
            "Scadenza": st.column_config.DateColumn("Scadenza"),
            "Status": st.column_config.TextColumn("Status"),
        },
    )
