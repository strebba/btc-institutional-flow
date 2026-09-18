from __future__ import annotations

import pandas as pd
import streamlit as st


from src.config import setup_logging
from src.dashboard.charts import gex_profile, gex_walls, regime_bars
from src.dashboard.data_loader import run_regime
from src.gex.pine_export import build_pine_indicator

_log = setup_logging("dashboard.tabs.gex")


def _tab_gex(snap: dict, gex_by_strike: list[dict], merged_df: pd.DataFrame) -> None:
    spot = snap.get("spot_price") or 0
    gex_m = (snap.get("total_net_gex") or 0) / 1e6
    regime = snap.get("regime", "unknown")
    max_pain = snap.get("max_pain") or 0

    st.header("Gamma Exposure (GEX)", icon=":material/candlestick_chart:")
    st.caption(
        "Dove e quanto i market maker sono obbligati a comprare o vendere BTC per "
        "restare coperti: la pressione meccanica nascosta nel mercato delle opzioni."
    )

    _regime_callout(regime, gex_m, max_pain)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(gex_profile(gex_by_strike, spot), width="stretch")
        with st.expander("Come leggere l'istogramma"):
            st.markdown(
                "Ogni barra è il GEX netto a uno strike price. **Verde** = zona "
                "stabilizzante (il dealer compra sui cali, vende sui rialzi), "
                "**rossa** = amplificante. L'altezza indica l'intensità; la linea "
                "verticale è lo spot."
            )
    with col2:
        st.plotly_chart(gex_walls(snap), width="stretch")
        mc1, mc2 = st.columns(2)
        mc1.metric("Max Pain", f"${snap.get('max_pain') or 0:,.0f}")
        mc2.metric("Strumenti BTC", f"{snap.get('n_instruments') or 0}")

    with st.expander("Indicatore TradingView (Pine Script)", icon=":material/code:"):
        st.caption(
            "Congela i livelli GEX correnti in un indicatore Pine v6 da incollare "
            "nel Pine Editor. Per aggiornare i livelli va rigenerato."
        )
        if st.button("Genera indicatore", key="gen_pine_indicator"):
            pine_code = build_pine_indicator(snap)
            st.code(pine_code, language="text")
            st.download_button(
                "Scarica gex_levels.pine",
                data=pine_code,
                file_name="gex_levels.pine",
                mime="text/plain",
                key="dl_pine_indicator",
            )

    with st.expander("Come calcoliamo il GEX", icon=":material/settings:"):
        st.markdown(
            "**Fonte**: API pubblica Deribit (opzioni BTC).\n\n"
            "**Formula**: `GEX = Gamma × Open Interest × Spot² × 0.01`\n\n"
            "- **Call** → GEX positivo (il dealer è tipicamente short, assorbe)\n"
            "- **Put** → GEX negativo (il dealer amplifica i movimenti al ribasso)\n\n"
            "**Limiti**: assumiamo che il dealer sia sempre la controparte; le opzioni "
            "IBIT su CBOE non sono incluse. Affidabilità stimata ~80% del segnale "
            "rispetto ai modelli professionali."
        )

    if not merged_df.empty:
        gex_today = snap.get("total_net_gex") or 0.0
        with st.spinner("Calcolo regime analysis..."):
            try:
                regime_result = run_regime(merged_df, gex_today)
                if regime_result.positive_stats or regime_result.negative_stats:
                    st.subheader("Regime: gamma positiva vs negativa")
                    st.plotly_chart(regime_bars(regime_result), width="stretch")
                    if regime_result.gex_vol_correlation is not None:
                        corr_mean = regime_result.gex_vol_correlation.dropna().mean()
                        st.info(
                            f"Correlazione media GEX ↔ BTC Vol (rolling 30d): **{corr_mean:.3f}**",
                            icon=":material/query_stats:",
                        )
            except Exception as e:
                _log.warning("Regime analysis non disponibile: %s", e)
                st.info("Regime analysis non disponibile: dati storici GEX insufficienti.")


def _regime_callout(regime: str, gex_m: float, max_pain: float) -> None:
    """Una riga di sintesi + dettaglio in expander, al posto del vecchio muro di testo."""
    if regime == "positive_gamma":
        st.success(
            "**Gamma positiva** — i dealer assorbono gli shock, volatilità contenuta.",
            icon=":material/check_circle:",
        )
        dettaglio = (
            f"Total GEX ${gex_m:+.1f}M. I dealer sono long gamma: vendono sui rialzi, "
            f"comprano sui cali. Aspettati mean-reversion e breakout che falliscono; "
            f"il prezzo tende verso il Max Pain a ${max_pain:,.0f}."
        )
    elif regime == "negative_gamma":
        st.error(
            "**Gamma negativa** — i dealer amplificano ogni movimento.",
            icon=":material/error:",
        )
        dettaglio = (
            f"Total GEX ${gex_m:+.1f}M. I dealer sono short gamma: comprano sui rialzi, "
            f"vendono sui cali. Possibili cascate, supporti/resistenze bucati con forza. "
            f"Riduci la leva e allarga gli stop."
        )
    else:
        st.info(
            "**Gamma neutrale** — prossimità del gamma flip, regime instabile.",
            icon=":material/info:",
        )
        dettaglio = (
            f"Total GEX ${gex_m:+.1f}M, vicino a zero. Il regime può cambiare rapidamente."
        )
    with st.expander("Implicazione operativa"):
        st.markdown(dettaglio)
