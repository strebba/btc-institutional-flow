"""CLI per il modulo 4: Statistical Analysis.

Esegue Granger causality, event study, regime analysis e backtest
usando i dati reali presenti nel database SQLite.

Uso:
    python scripts/run_analytics.py [--backtest] [--granger] [--regime] [--events]
    python scripts/run_analytics.py  # esegue tutto
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Aggiunge il root al sys.path per import locali
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.analytics.backtest import Backtest
from src.analytics.event_study import EventStudy
from src.analytics.granger import GrangerAnalysis
from src.analytics.regime_analysis import RegimeAnalysis
from src.config import get_settings, setup_logging
from src.edgar.structured_notes_db import StructuredNotesDB
from src.flows.correlation import FlowCorrelation
from src.flows.price_fetcher import PriceFetcher
from src.flows.scraper import FarsideScraper
from src.gex.deribit_client import DeribitClient
from src.gex.gex_calculator import GexCalculator
from src.gex.regime_detector import RegimeDetector

_log = setup_logging("scripts.run_analytics")


def _load_merged_df(settings: dict) -> pd.DataFrame:
    """Carica e unisce prezzi BTC + flussi IBIT dal DB locale."""
    pf = PriceFetcher()
    prices = pf.get_all_prices()
    if prices.empty:
        _log.warning("Prezzi non disponibili nel DB — scarico da yfinance...")
        pf.fetch_and_store(tickers=["BTC-USD", "IBIT"], days_back=365)
        prices = pf.get_all_prices()

    scraper = FarsideScraper()
    flows = scraper.fetch()
    agg_flows = scraper.aggregate(flows)

    corr = FlowCorrelation()
    merged = corr.merge(agg_flows, prices)
    _log.info("Merged DataFrame: %d righe, colonne=%s", len(merged), list(merged.columns))
    return merged


def _load_gex_series() -> pd.Series | None:
    """Ottieni snapshot GEX dal live Deribit."""
    try:
        client = DeribitClient()
        calc = GexCalculator()
        spot = client.get_spot_price()
        options = client.fetch_all_options("BTC")
        snap = calc.calculate_gex(options, spot)
        detector = RegimeDetector()
        state = detector.detect(snap)
        print(f"\n[GEX Live] Regime: {state.regime}, GEX: ${snap.total_net_gex/1e6:.1f}M")
        if state.alerts:
            for alert in state.alerts:
                print(f"  Alert: {alert}")
        # Crea serie con la data odierna
        import pandas as pd
        today = pd.Timestamp.today().normalize()
        return pd.Series({today: snap.total_net_gex})
    except Exception as e:
        _log.warning("GEX non disponibile: %s", e)
        return None


def _load_barriers(settings: dict) -> list[dict]:
    """Carica barriere attive dal DB EDGAR."""
    try:
        db = StructuredNotesDB()
        barriers = db.get_active_barriers()
        _log.info("Barriere attive: %d", len(barriers))
        return barriers
    except Exception as e:
        _log.warning("Barriere non disponibili: %s", e)
        return []


def run_granger(merged_df: pd.DataFrame) -> None:
    """Test di Granger causality: IBIT flows ↔ BTC returns."""
    print("\n" + "=" * 60)
    print("GRANGER CAUSALITY TEST")
    print("=" * 60)
    analyzer = GrangerAnalysis()
    try:
        results = analyzer.run(merged_df)
        print(analyzer.interpret(results))
        frame = analyzer.to_dataframe(results)
        sig_rows = frame[frame["significant"]]
        if not sig_rows.empty:
            print(f"\nLag significativi trovati:\n{sig_rows.to_string()}")
        else:
            print("\nNessun lag significativo trovato.")
    except Exception as e:
        _log.error("Granger test fallito: %s", e)


def run_regime(merged_df: pd.DataFrame, gex_series: pd.Series | None) -> None:
    """Analisi regime GEX e differenze di rendimento tra regimi."""
    print("\n" + "=" * 60)
    print("REGIME ANALYSIS")
    print("=" * 60)
    analyzer = RegimeAnalysis()
    try:
        result = analyzer.analyze(merged_df, gex_series)
        print(result.interpretation)
        if result.gex_vol_correlation is not None:
            corr_mean = result.gex_vol_correlation.dropna().mean()
            print(f"\nCorrelazione media GEX ↔ BTC Vol (rolling 30d): {corr_mean:.3f}")
    except Exception as e:
        _log.error("Regime analysis fallita: %s", e)


def run_events(barriers: list[dict], merged_df: pd.DataFrame) -> None:
    """Event study sui barrier levels IBIT."""
    print("\n" + "=" * 60)
    print("EVENT STUDY — BARRIER LEVELS")
    print("=" * 60)

    if not barriers:
        print("Nessuna barriera attiva nel DB.")
        return

    study = EventStudy()
    # Costruisci DataFrame prezzi BTC nel formato richiesto
    if "btc_close" not in merged_df.columns:
        print("Colonna btc_close mancante — event study saltato.")
        return

    prices_df = merged_df[["btc_close"]].rename(columns={"btc_close": "close"}).dropna()

    try:
        results = study.run(barriers, prices_df)
        for res in results:
            sig_tag = "*** SIGNIFICATIVO" if res.significant else ""
            print(
                f"\n  Barriera: {res.barrier_type} | eventi: {res.n_events}"
                f" | CAR={res.car_mean*100:+.2f}% | p={res.p_value:.4f} {sig_tag}"
            )
            if res.n_events >= 2:
                print(f"  CI 95%: [{res.ci_lower*100:.2f}%, {res.ci_upper*100:.2f}%]")
    except Exception as e:
        _log.error("Event study fallito: %s", e)


def run_backtest(
    merged_df: pd.DataFrame,
    gex_series: pd.Series | None,
    barriers: list[dict],
) -> None:
    """Backtest della strategia GEX + flussi ETF."""
    print("\n" + "=" * 60)
    print("BACKTEST: GEX + ETF FLOWS STRATEGY vs BUY & HOLD BTC")
    print("=" * 60)
    bt = Backtest()
    try:
        results = bt.run(merged_df, gex_series, barriers or None)
        if not results:
            print("Backtest non eseguito: dati insufficienti.")
            return
        table = bt.summary_table(results)
        print(f"\n{table.to_string()}")

        strat = results["strategy"]
        bah   = results["buy_and_hold"]
        delta_sharpe = strat.sharpe_ratio - bah.sharpe_ratio
        print(f"\nDelta Sharpe (strategia - B&H): {delta_sharpe:+.2f}")
        print(
            f"Posizioni: long={strat.days_long}d, "
            f"short={strat.days_short}d, flat={strat.days_flat}d | "
            f"trades={strat.n_trades}"
        )
    except Exception as e:
        _log.error("Backtest fallito: %s", e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Esegue gli analytics GEX + ETF flows.")
    parser.add_argument("--granger", action="store_true", help="Solo Granger causality")
    parser.add_argument("--regime",  action="store_true", help="Solo regime analysis")
    parser.add_argument("--events",  action="store_true", help="Solo event study")
    parser.add_argument("--backtest",action="store_true", help="Solo backtest")
    args = parser.parse_args()

    run_all = not any([args.granger, args.regime, args.events, args.backtest])

    settings = get_settings()

    print("Carico dati merged (prezzi + flussi)...")
    merged_df = _load_merged_df(settings)
    if merged_df.empty:
        print("ERRORE: DataFrame merged vuoto. Verificare i dati nel DB.")
        sys.exit(1)
    print(f"  → {len(merged_df)} righe disponibili")

    print("Carico snapshot GEX live da Deribit...")
    gex_series = _load_gex_series()

    print("Carico barriere attive dal DB EDGAR...")
    barriers = _load_barriers(settings)

    if run_all or args.granger:
        run_granger(merged_df)

    if run_all or args.regime:
        run_regime(merged_df, gex_series)

    if run_all or args.events:
        run_events(barriers, merged_df)

    if run_all or args.backtest:
        run_backtest(merged_df, gex_series, barriers)

    print("\nAnalytics completati.")


if __name__ == "__main__":
    main()
