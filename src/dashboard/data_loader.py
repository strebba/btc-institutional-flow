from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import get_settings, setup_logging

_log = setup_logging("dashboard.data_loader")
_settings = get_settings()
_REFRESH = _settings["dashboard"]["refresh_interval_s"]


def _get_backtest_context(days: int = 365):
    """Helper condiviso: GexDB series + StructuredNotesDB barrier_history.

    Sostituisce i 4 blocchi duplicati in run_backtest, run_walk_forward,
    run_factor_decomp e run_sensitivity.
    """
    from src.edgar.structured_notes_db import StructuredNotesDB
    from src.gex.gex_db import GexDB

    gex_series = GexDB().get_series(days=days)
    barrier_history = None
    try:
        bh = StructuredNotesDB().get_barrier_history(days=days)
        if not bh.empty:
            barrier_history = bh
    except Exception:
        pass
    return gex_series, barrier_history


@st.cache_data(ttl=_REFRESH, show_spinner=False)
def load_prices_and_flows() -> pd.DataFrame:
    """Carica prezzi BTC/IBIT e flussi ETF aggregati."""
    from src.flows.correlation import FlowCorrelation
    from src.flows.price_fetcher import PriceFetcher
    from src.flows.scraper import FarsideScraper

    pf = PriceFetcher()
    prices = pf.get_all_prices()
    if prices.empty:
        # FIX: PriceFetcher non ha fetch_and_store — usa fetch() direttamente
        pf.fetch("BTC-USD")
        pf.fetch("IBIT")
        prices = pf.get_all_prices()

    scraper = FarsideScraper()
    flows = scraper.fetch()
    agg_flows = scraper.aggregate(flows)

    corr = FlowCorrelation()
    return corr.merge(agg_flows, prices)


@st.cache_data(ttl=_REFRESH, show_spinner=False)
def load_gex() -> tuple[dict, list[dict]]:
    """Carica snapshot GEX live da Deribit e salva nel DB storico."""
    from src.gex.deribit_client import DeribitClient
    from src.gex.gex_calculator import GexCalculator
    from src.gex.gex_db import GexDB
    from src.gex.regime_detector import RegimeDetector

    client = DeribitClient()
    calc = GexCalculator()
    db = GexDB()
    detector = RegimeDetector()

    # Pre-popola storico per percentile GEX corretto
    detector.load_history_from_db(db.get_latest_n(90))

    spot = client.get_spot_price()
    options = client.fetch_all_options("BTC")
    snap = calc.calculate_gex(options, spot)
    state = detector.detect(snap)

    # Persiste snapshot nel DB
    try:
        db.insert_snapshot(snap, state.regime)
    except Exception as _e:
        _log.warning("GEX DB insert fallito: %s", _e)

    snap_dict = calc.gex_to_dict(snap)
    snap_dict["regime"] = state.regime
    snap_dict["alerts"] = state.alerts
    snap_dict["gex_percentile"] = state.gex_percentile

    by_strike = [
        {
            "strike": g.strike,
            "net_gex": g.net_gex,
            "call_gex": g.call_gex,
            "put_gex": g.put_gex,
        }
        for g in snap.gex_by_strike
    ]
    return snap_dict, by_strike


@st.cache_data(ttl=_REFRESH, show_spinner=False)
def load_barriers() -> list[dict]:
    """Carica barriere attive dal DB EDGAR, arricchite con level_price_btc calcolato.

    compute_btc_prices() nel DB richiede un run manuale dello script EDGAR.
    Qui calcoliamo il prezzo BTC al volo dal ratio IBIT/BTC corrente:
      btc = level_price_ibit / ibit_btc_ratio
    o, se manca level_price_ibit ma abbiamo initial_level + level_pct:
      btc = (initial_level × level_pct/100) / ibit_btc_ratio
    """
    from src.edgar.structured_notes_db import StructuredNotesDB
    from src.flows.price_fetcher import PriceFetcher

    db = StructuredNotesDB()
    barriers = db.get_active_barriers()

    # Ottieni ratio IBIT/BTC corrente per la conversione
    try:
        ratio = PriceFetcher().get_ibit_btc_ratio() or 0.0
    except Exception:
        ratio = 0.0

    enriched: list[dict] = []
    for b in barriers:
        b = dict(b)  # copia mutabile

        if not (b.get("level_price_btc") or 0):
            ibit_price = b.get("level_price_ibit") or 0.0
            # Fallback: calcola da initial_level × level_pct
            if not ibit_price:
                init = b.get("initial_level") or 0.0
                pct = b.get("level_pct") or 0.0
                if init > 0 and pct > 0:
                    ibit_price = init * pct / 100.0
            if ibit_price > 0 and ratio > 0:
                b["level_price_btc"] = ibit_price / ratio
                b["level_price_ibit"] = b.get("level_price_ibit") or ibit_price

        # Salta barriere per cui non riusciamo a ricavare un prezzo BTC
        if (b.get("level_price_btc") or 0) > 0:
            enriched.append(b)
        else:
            _log.debug("Barrier id=%s senza prezzo BTC calcolabile — skipped", b.get("id"))

    _log.info("Barriere caricate: %d totali, %d con BTC price", len(barriers), len(enriched))
    return enriched


@st.cache_data(ttl=_REFRESH, show_spinner=False)
def load_db_summary() -> dict:
    """Carica statistiche aggregate dal DB EDGAR."""
    from src.edgar.structured_notes_db import StructuredNotesDB

    db = StructuredNotesDB()
    return db.summary()


@st.cache_data(ttl=_REFRESH, show_spinner=False)
def load_macro() -> dict:
    """Carica i dati macro CoinGlass (funding/OI/L-S/liquidazioni) per il pilastro Macro.

    Best-effort: se CoinGlass non è disponibile ritorna {} e il pilastro Macro
    risulterà 'n/d' (i pesi vengono riscalati sugli altri).

    Usa src.flows.macro_fetcher.fetch_macro_data() — stessa fonte di /api/signals.
    """
    from src.flows.macro_fetcher import fetch_macro_data

    try:
        macro = fetch_macro_data()
        return macro.to_dict()
    except Exception as e:
        _log.warning("Macro CoinGlass non disponibile per la dashboard: %s", e)
        return {}


@st.cache_data(ttl=_REFRESH, show_spinner=False)
def run_granger(merged_df: pd.DataFrame) -> tuple[dict, pd.DataFrame, str]:
    """Esegue il test di Granger causality."""
    from src.analytics.granger import GrangerAnalysis

    analyzer = GrangerAnalysis()
    results = analyzer.run(merged_df)
    df = analyzer.to_dataframe(results)
    interp = analyzer.interpret(results)
    return results, df, interp


@st.cache_data(ttl=_REFRESH, show_spinner=False)
def run_regime(merged_df: pd.DataFrame, gex_today: float):
    """Esegue la regime analysis."""
    from src.analytics.regime_analysis import RegimeAnalysis

    analyzer = RegimeAnalysis()

    if "total_gex" not in merged_df.columns and gex_today != 0:
        today = pd.Timestamp.today().normalize()
        gex_series = pd.Series({today: gex_today}, name="total_gex")
        return analyzer.analyze(merged_df, gex_series=gex_series)

    return analyzer.analyze(merged_df)


@st.cache_data(ttl=_REFRESH, show_spinner=False)
def run_backtest(merged_df: pd.DataFrame, barriers: list[dict]):
    """Esegue il backtest a 4 pilastri con serie GEX storica dal DB e storico barriere."""
    from src.analytics.backtest import Backtest
    from src.analytics.pillars import CompositeSignal

    gex_series, barrier_history = _get_backtest_context(days=365)

    bt = Backtest()
    return bt, bt.run(
        merged_df,
        gex_series=gex_series if not gex_series.empty else None,
        active_barriers=barriers if barriers else None,
        barrier_history=barrier_history,
        composite=CompositeSignal(),
    )


@st.cache_data(ttl=_REFRESH, show_spinner=False)
def run_event_study(barriers: list[dict], merged_df: pd.DataFrame) -> list:
    """Esegue l'event study sui barrier levels."""
    from src.analytics.event_study import EventStudy

    if not barriers or "btc_close" not in merged_df.columns:
        return []
    study = EventStudy()
    prices_df = merged_df[["btc_close"]].rename(columns={"btc_close": "close"}).dropna()
    return study.run(barriers, prices_df)


# ──────────────────────────────────────────────────────────────────────────────
# Composite signal
# ──────────────────────────────────────────────────────────────────────────────


def compute_composite(
    snap: dict,
    merged_df: pd.DataFrame,
    barriers: list[dict],
    macro: dict | None = None,
):
    """Calcola il segnale composito a 4 pilastri (GEX, Barrier, ETF Flows, Macro).

    Riusa l'unica sorgente di verità `CompositeSignal` (stessa logica di /api/signals),
    così dashboard e API non divergono.

    Returns:
        CompositeResult con score, signal, pillars[], reason.
    """
    from src.analytics.pillars import CompositeSignal, CompositeInputs

    macro = macro or {}

    ibit_3d = 0.0
    if not merged_df.empty and "ibit_flow_3d" in merged_df.columns:
        last = merged_df["ibit_flow_3d"].dropna()
        if not last.empty:
            ibit_3d = float(last.iloc[-1])

    inputs = CompositeInputs(
        gex_usd=snap.get("total_net_gex"),
        gamma_flip_price=snap.get("gamma_flip_price"),
        put_wall=snap.get("put_wall"),
        call_wall=snap.get("call_wall"),
        active_barriers=barriers or None,
        etf_flow_3d_usd=ibit_3d,
        flow_history_df=merged_df if not merged_df.empty else None,
        put_call_ratio=snap.get("put_call_ratio"),
        spot_price=snap.get("spot_price"),
        funding_rate_annualized_pct=macro.get("funding_rate_annualized_pct"),
        oi_change_7d_pct=macro.get("oi_change_7d_pct"),
        long_short_ratio=macro.get("long_short_ratio"),
        liquidations_long_24h_usd=macro.get("liquidations_long_24h_usd"),
        liquidations_short_24h_usd=macro.get("liquidations_short_24h_usd"),
    )
    return CompositeSignal().compute(inputs)


# ──────────────────────────────────────────────────────────────────────────────
# Quantitative validation (nuovi moduli)
# ──────────────────────────────────────────────────────────────────────────────


@st.cache_data(ttl=_REFRESH, show_spinner=False)
def run_walk_forward(merged_df: pd.DataFrame, barriers: list[dict]) -> dict | None:
    """Walk-forward backtest del CompositeSignal a 4 pilastri.

    Returns:
        dict da WalkForwardBacktest.analyze() o None se dati insufficienti.
    """
    from src.analytics.walk_forward import WalkForwardBacktest

    if merged_df.empty or "btc_return" not in merged_df.columns:
        return None

    gex_series, barrier_history = _get_backtest_context(days=730)

    wfb = WalkForwardBacktest()
    results = wfb.run(
        merged_df,
        train_days=504,
        test_days=63,
        step_days=63,
        gex_series=gex_series if not gex_series.empty else None,
        active_barriers=barriers if barriers else None,
        barrier_history=barrier_history,
    )
    return wfb.analyze(results)


@st.cache_data(ttl=_REFRESH, show_spinner=False)
def run_factor_decomp(merged_df: pd.DataFrame, barriers: list[dict]) -> dict | None:
    """Factor decomposition dei rendimenti della strategia.

    Returns:
        dict con decomposition e exposures, o None.
    """
    from src.analytics.backtest import Backtest
    from src.analytics.factor_decomposition import FactorDecomposition
    from src.analytics.pillars import CompositeSignal

    if merged_df.empty or "btc_return" not in merged_df.columns:
        return None

    bt = Backtest()
    composite = CompositeSignal()

    gex_series, barrier_history = _get_backtest_context(days=365)

    df = merged_df.copy()
    if gex_series is not None and not gex_series.empty:
        df = df.join(gex_series.rename("_gex"), how="left")
        df["_gex"] = df["_gex"].ffill().fillna(0.0)

    signals = bt._generate_signals(
        df,
        gex_series=gex_series if not gex_series.empty else None,
        active_barriers=barriers if barriers else None,
        barrier_history=barrier_history,
        composite=composite,
    )
    signals_lagged = signals.shift(1).fillna(0.0)
    rets = df["btc_return"].dropna()
    strat_rets = rets * signals_lagged.reindex(rets.index).fillna(0.0)

    fd = FactorDecomposition()
    factors = fd.build_default_factors(rets)

    if factors.empty:
        return None

    exposures = fd.calculate_factor_exposures(strat_rets, factors)
    decomposition = fd.decompose_strategy_returns(strat_rets, factors)

    return {"exposures": exposures, "decomposition": decomposition}


@st.cache_data(ttl=_REFRESH, show_spinner=False)
def run_sensitivity(merged_df: pd.DataFrame, barriers: list[dict]) -> dict | None:
    """Parameter sensitivity dei pesi dei pilastri.

    Returns:
        dict da ParameterSensitivity.pillar_sensitivity() o None.
    """
    from src.analytics.sensitivity import ParameterSensitivity

    if merged_df.empty or "btc_return" not in merged_df.columns:
        return None

    gex_series, barrier_history = _get_backtest_context(days=365)

    ps = ParameterSensitivity()
    return ps.pillar_sensitivity(
        merged_df,
        delta=0.20,
        gex_series=gex_series if not gex_series.empty else None,
        active_barriers=barriers if barriers else None,
        barrier_history=barrier_history,
    )
