from __future__ import annotations

import pandas as pd
import streamlit as st


from src.dashboard.charts import barrier_map as _barrier_map_chart, barrier_gex_confluence_chart

from src.edgar.barrier_utils import compute_confluence, detect_clusters

def _tab_barrier_map(barriers: list[dict], snap: dict) -> None:

    spot = snap.get("spot_price") or 0.0

    st.header("🎯 Mappa dei Livelli Critici")
    st.markdown("""
Questa mappa mostra i **livelli di prezzo** dove le banche che hanno emesso
prodotti strutturati su IBIT sono **obbligate** a comprare o vendere Bitcoin
per ribilanciare il loro hedge. Quando il prezzo si avvicina a questi livelli,
aspettati movimenti meccanici e improvvisi — non guidati dal sentiment,
ma dalla gestione del rischio dei dealer.
""")

    # Legenda colori condizionale (con conteggi reali)
    type_counts: dict[str, int] = {}
    type_notional: dict[str, float] = {}
    for b in barriers:
        t = b.get("barrier_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
        type_notional[t] = type_notional.get(t, 0) + (b.get("notional_usd") or 0)

    has_any = False
    if type_counts.get("knock_in", 0) > 0:
        has_any = True
        n = type_counts["knock_in"]
        noz = type_notional.get("knock_in", 0) / 1e6
        st.markdown(
            f"🔴 **Knock-In** ({n} barriere, ${noz:,.0f}M nozionale) — "
            "Rottura al ribasso → dealer vendono aggressivamente."
        )
    if type_counts.get("autocall", 0) > 0:
        has_any = True
        n = type_counts["autocall"]
        noz = type_notional.get("autocall", 0) / 1e6
        st.markdown(
            f"🟢 **Auto-Call** ({n} barriere, ${noz:,.0f}M nozionale) — "
            "Superamento rialzo → rimborso nota + chiusura hedge."
        )
    if type_counts.get("buffer", 0) > 0:
        has_any = True
        n = type_counts["buffer"]
        noz = type_notional.get("buffer", 0) / 1e6
        st.markdown(
            f"🔵 **Buffer Zone** ({n} barriere, ${noz:,.0f}M nozionale) — "
            "Protezione parziale, hedging graduale."
        )
    if type_counts.get("knock_out", 0) > 0:
        has_any = True
        n = type_counts["knock_out"]
        noz = type_notional.get("knock_out", 0) / 1e6
        st.markdown(
            f"🟣 **Knock-Out** ({n} barriere, ${noz:,.0f}M nozionale) — "
            "Estinzione automatica al superamento."
        )
    if not has_any:
        if not barriers:
            st.info(
                "Nessuna barriera attiva nel DB. "
                "Esegui `scripts/run_edgar.py` per scansionare i filing SEC e popolare il database."
            )
        else:
            st.info(
                "Nessuna barriera con prezzo BTC calcolabile — "
                "il ratio IBIT/BTC potrebbe non essere disponibile. "
                "Esegui `make update-edgar` per ricalcolare i prezzi."
            )
        return

    # Grafico barrier map
    fig = _barrier_map_chart(barriers, spot)
    st.plotly_chart(fig, width="stretch")

    # Alert contestuale basato su distanza
    valid_barriers = [b for b in barriers if (b.get("level_price_btc") or 0) > 0 and spot > 0]
    if valid_barriers:
        closest = min(
            valid_barriers,
            key=lambda b: abs((b.get("level_price_btc") or 0) - spot),
        )
        level = closest.get("level_price_btc", 0)
        distance = abs(spot - level) / spot * 100
        btype = closest.get("barrier_type", "barrier")
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

    # ── Overlay confluenza barriere ↔ GEX ──────────────────────────────────────
    if valid_barriers and spot > 0:

        clusters = detect_clusters(barriers, spot)
        confluence = compute_confluence(
            clusters,
            put_wall=snap.get("put_wall"),
            call_wall=snap.get("call_wall"),
            gamma_flip=snap.get("gamma_flip_price"),
        )
        if clusters:
            st.subheader("🔗 Confluenza Barriere ↔ GEX")
            st.markdown("""
Quando un **cluster di barriere** delle note strutturate coincide con un **muro
di gamma** dei dealer di opzioni (put/call wall, gamma flip), due meccanismi di
hedging indipendenti spingono nella **stessa direzione, allo stesso prezzo**:
l'effetto meccanico si **amplifica**.
""")
            st.plotly_chart(
                barrier_gex_confluence_chart(clusters, snap, spot, confluence),
                width="stretch",
            )
            if confluence:
                for c in confluence:
                    ctype = c.get("confluence_type")
                    px = c.get("gex_level_price", 0)
                    gex_name = c.get("gex_level_name", "")
                    notional = c.get("cluster_notional_usd", 0)
                    if ctype == "bearish_reinforced":
                        st.error(
                            f"🔴 **Ribasso rinforzato** a ${px:,.0f}: cluster di barriere "
                            f"ribassiste sul {gex_name} (nozionale ${notional:,.0f}). "
                            "Rottura al ribasso potenzialmente accelerata."
                        )
                    elif ctype == "bullish_reinforced":
                        st.success(
                            f"🟢 **Rialzo rinforzato** a ${px:,.0f}: cluster di barriere "
                            f"rialziste sul {gex_name} (nozionale ${notional:,.0f}). "
                            "Resistenza/soft-cap rafforzata."
                        )
            else:
                st.info(
                    "Nessuna confluenza diretta tra cluster di barriere e wall GEX "
                    "ai livelli di prezzo attuali."
                )

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
        lvl_btc = b.get("level_price_btc")
        dist_str = f"{abs(lvl_btc - spot) / spot * 100:.1f}%" if lvl_btc and spot else "—"
        rows.append(
            {
                "Tipo": b.get("barrier_type", ""),
                "Emittente": b.get("issuer", ""),
                "Prodotto": b.get("product_type", ""),
                "Livello %": f"{b.get('level_pct', 0):.0f}%" if b.get("level_pct") else "—",
                "Prezzo BTC": f"${lvl_btc:,.0f}" if lvl_btc else "—",
                "Distanza": dist_str,
                "Scadenza": b.get("maturity_date", ""),
                "Status": b.get("status", ""),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch")
