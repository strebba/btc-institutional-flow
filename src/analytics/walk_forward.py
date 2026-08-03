"""Walk-forward backtest del CompositeSignal.

Applica la metodologia walk-forward (rolling train/test) al CompositeSignal
a 4 pilastri per validare il potere predittivo out-of-sample. Senza walk-forward,
il backtest full-series non distingue tra overfitting in-sample e performance reale.

Reference: patterns.md "Proper Backtest Framework" — walk-forward è il MINIMO
per la validazione di una strategia.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.config import setup_logging
from src.forecast.validation import walk_forward_windows

_log = setup_logging("analytics.walk_forward")


@dataclass
class WalkForwardResult:
    """Risultato di una singola finestra walk-forward.

    Attributes:
        train_sharpe: Sharpe ratio sulla finestra di training (in-sample).
        test_sharpe: Sharpe ratio sulla finestra di test (out-of-sample).
        train_start: data inizio training.
        train_end: data fine training.
        test_start: data inizio test.
        test_end: data fine test.
        n_trades: numero di cambi di segnale nel periodo di test.
        test_total_return: rendimento totale nel periodo di test.
    """

    train_sharpe: float
    test_sharpe: float
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_trades: int
    test_total_return: float


class WalkForwardBacktest:
    """Esegue il walk-forward backtest per il CompositeSignal a 4 pilastri.

    Per ogni finestra rolling train→test:
      1. Calcola i segnali CompositeSignal sui dati di train (per calibrare
         i pesi) e di test.
      2. Valuta la performance SOLO sui dati di test (out-of-sample).
      3. Aggrega le metriche OOS su tutte le finestre.

    Args:
        cfg: configurazione backtest (da settings.yaml).
    """

    def __init__(self, cfg: dict | None = None) -> None:
        from src.config import get_settings

        self._cfg = cfg or get_settings()["backtest"]

    def run(
        self,
        merged_df: pd.DataFrame,
        train_days: int = 504,
        test_days: int = 63,
        step_days: int = 63,
        gex_series: Optional[pd.Series] = None,
        active_barriers: Optional[list[dict]] = None,
        barrier_history: Optional[pd.DataFrame] = None,
    ) -> list[WalkForwardResult]:
        """Esegue il walk-forward backtest su finestre rolling train→test.

        Args:
            merged_df: DataFrame con btc_return, btc_close, ibit_flow_3d,
                total_net_gex, total_flow_usd, btc_vol_7d, e colonne macro.
            train_days: giorni per la finestra di training (default 504 = ~2 anni).
            test_days: giorni per la finestra di test (default 63 = ~3 mesi).
            step_days: giorni di step tra finestre consecutive (default 63).
            gex_series: serie GEX storica opzionale (altrimenti dal df).
            active_barriers: barriere attive correnti.
            barrier_history: storico barriere da StructuredNotesDB.

        Returns:
            list[WalkForwardResult]: risultati per ogni finestra, in ordine
            temporale crescente.
        """
        if "btc_return" not in merged_df.columns:
            _log.error("btc_return mancante nel DataFrame")
            return []

        from src.analytics.backtest import Backtest
        from src.analytics.pillars import CompositeSignal

        df = merged_df.copy()
        bt = Backtest(self._cfg)
        composite = CompositeSignal()

        results: list[WalkForwardResult] = []

        windows = list(
            walk_forward_windows(df.index, train_size=train_days, test_size=test_days, step=step_days)
        )
        _log.info("Walk-forward: %d finestre (train=%d, test=%d, step=%d)",
                  len(windows), train_days, test_days, step_days)

        for w in windows:
            train_df = df.loc[w.train_idx]
            test_df = df.loc[w.test_idx]

            if len(train_df) < 30 or len(test_df) < 10:
                continue

            block_df = pd.concat([train_df, test_df])

            block_gex = None
            if gex_series is not None:
                block_gex = gex_series.reindex(block_df.index)

            block_barrier_history = None
            if barrier_history is not None and not barrier_history.empty:
                train_start_d = pd.Timestamp(w.train_idx[0]).date()
                test_end_d = pd.Timestamp(w.test_idx[-1]).date()
                block_barrier_history = barrier_history[
                    (barrier_history["snapshot_date"] >= train_start_d)
                    & (barrier_history["snapshot_date"] <= test_end_d)
                ]
                if block_barrier_history.empty:
                    block_barrier_history = None

            signals = bt._generate_signals(
                block_df, block_gex, active_barriers, block_barrier_history,
                composite=composite,
            )

            train_signals = signals.loc[w.train_idx]
            test_signals = signals.loc[w.test_idx]

            test_signals_lagged = test_signals.shift(1).fillna(0)

            test_rets = test_df["btc_return"].dropna()
            test_strat_rets = test_rets * test_signals_lagged.reindex(test_rets.index).fillna(0)

            train_signals_lagged = train_signals.shift(1).fillna(0)
            train_rets = train_df["btc_return"].dropna()
            train_strat_rets = train_rets * train_signals_lagged.reindex(train_rets.index).fillna(0)

            train_metrics = bt._compute_metrics(train_strat_rets, "train")
            test_metrics = bt._compute_metrics(test_strat_rets, "test")

            results.append(WalkForwardResult(
                train_sharpe=train_metrics.sharpe_ratio,
                test_sharpe=test_metrics.sharpe_ratio,
                train_start=pd.Timestamp(w.train_idx[0]),
                train_end=pd.Timestamp(w.train_idx[-1]),
                test_start=pd.Timestamp(w.test_idx[0]),
                test_end=pd.Timestamp(w.test_idx[-1]),
                n_trades=test_metrics.n_trades,
                test_total_return=test_metrics.total_return,
            ))

        _log.info("Walk-forward completato: %d periodi", len(results))
        return results

    def analyze(self, results: list[WalkForwardResult]) -> dict:
        """Analizza i risultati walk-forward per determinare la viability.

        Reference: patterns.md "analyze_walk_forward".

        Args:
            results: lista da run().

        Returns:
            dict con metriche aggregate e flag is_viable.
        """
        if not results:
            return {
                "avg_train_sharpe": 0.0,
                "avg_test_sharpe": 0.0,
                "sharpe_degradation": 1.0,
                "test_sharpe_std": 0.0,
                "pct_profitable_periods": 0.0,
                "worst_test_sharpe": 0.0,
                "total_periods": 0,
                "is_viable": False,
            }

        train_sharpes = [r.train_sharpe for r in results]
        test_sharpes = [r.test_sharpe for r in results]

        avg_train = float(np.mean(train_sharpes))
        avg_test = float(np.mean(test_sharpes))
        degradation = 1.0 - avg_test / avg_train if avg_train > 0 else 1.0
        test_std = float(np.std(test_sharpes))
        pct_prof = sum(1 for s in test_sharpes if s > 0) / len(test_sharpes)
        worst = min(test_sharpes)

        is_viable = bool(
            avg_test > 0.3
            and (avg_train > 0 and avg_test / avg_train > 0.4)
            and pct_prof >= 0.5
        )

        return {
            "avg_train_sharpe": round(avg_train, 3),
            "avg_test_sharpe": round(avg_test, 3),
            "sharpe_degradation": round(degradation, 3),
            "test_sharpe_std": round(test_std, 3),
            "pct_profitable_periods": round(pct_prof, 3),
            "worst_test_sharpe": round(worst, 3),
            "total_periods": len(results),
            "is_viable": is_viable,
        }

    def summary_table(self, analysis: dict) -> pd.DataFrame:
        """Crea DataFrame riepilogativo dell'analisi walk-forward.

        Args:
            analysis: dict da analyze().

        Returns:
            pd.DataFrame con una riga di metriche aggregate.
        """
        return pd.DataFrame([{
            "Train Sharpe Avg": analysis["avg_train_sharpe"],
            "Test Sharpe Avg": analysis["avg_test_sharpe"],
            "Sharpe Degradation": f"{analysis['sharpe_degradation']:.1%}",
            "Test Sharpe Std": analysis["test_sharpe_std"],
            "% Profitable": f"{analysis['pct_profitable_periods']:.0%}",
            "Worst Test Sharpe": analysis["worst_test_sharpe"],
            "Periods": analysis["total_periods"],
            "Viable": "YES" if analysis["is_viable"] else "NO",
        }])
