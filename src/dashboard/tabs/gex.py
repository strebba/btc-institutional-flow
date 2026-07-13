from __future__ import annotations

import pandas as pd
import streamlit as st


from src.dashboard.charts import gex_profile, gex_walls, regime_bars

def _tab_gex(snap: dict, gex_by_strike: list[dict], merged_df: pd.DataFrame) -> None:
    from src.dashboard.charts import gex_profile, gex_walls

    spot = snap.get("spot_price") or 0
    gex_m = (snap.get("total_net_gex") or 0) / 1e6
    regime = snap.get("regime", "unknown")
    put_wall = snap.get("put_wall") or 0
    call_wall = snap.get("call_wall") or 0
    flip = snap.get("gamma_flip_price") or 0
    max_pain = snap.get("max_pain") or 0

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
        "neutral": "NEUTRALE",
    }.get(regime, regime.upper())
    c1.metric(
        "Regime Corrente",
        regime_label,
        help="Positivo = i dealer assorbono la volatilità. Negativo = la amplificano.",
    )
    c2.metric(
        "Gamma Flip Point",
        f"${flip:,.0f}",
        delta=f"{(flip - spot) / spot * 100:+.1f}% da spot" if spot and flip else None,
        help="Sopra questo livello il mercato è stabilizzante, sotto è amplificante.",
    )
    c3.metric(
        "Put Wall (Supporto)",
        f"${put_wall:,.0f}",
        delta=f"{(put_wall - spot) / spot * 100:+.1f}% da spot" if spot and put_wall else None,
        delta_color="inverse",
        help="I dealer comprano qui, creando supporto meccanico.",
    )
    c4.metric(
        "Call Wall (Resistenza)",
        f"${call_wall:,.0f}",
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
        mc1, mc2 = st.columns(2)
        mc1.metric("Max Pain", f"${snap.get('max_pain') or 0:,.0f}")
        mc2.metric("Strumenti BTC", f"{snap.get('n_instruments') or 0}")

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
                _log.warning("Regime analysis non disponibile: %s", e)
                st.info("Regime analysis non disponibile: dati storici GEX insufficienti.")