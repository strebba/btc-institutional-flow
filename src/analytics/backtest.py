"""Backtest della strategia combinata GEX + flussi ETF (multi-fattore).

Supporta tre modalità di generazione segnali:
  1. CompositeSignal (default dashboard): 4 pilastri pesati.
  2. SignalModel: scoring 0-100 su 8 fattori ponderati.
  3. Regole legacy (fallback): GEX > 0 AND flow_3d > 100M → LONG.

Null models per validazione statistica:
  - Random: segnale casuale ±1 con probabilità 50%.
  - Always Long: posizione long permanente.
  - Momentum 20d: long se BTC return 20d > 0, altrimenti short.

Metriche: Sharpe ratio, max drawdown, win rate, profit factor.
Confronto vs buy-and-hold BTC + null models. Equity curve serializzata per frontend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from src.config import get_settings, setup_logging

if TYPE_CHECKING:
    from src.analytics.factor_scorers import SignalModel
    from src.analytics.pillars import CompositeSignal

_log = setup_logging("analytics.backtest")


@dataclass
class BacktestMetrics:
    """Risultati del backtest.

    Attributes:
        strategy_name: nome della strategia.
        total_return: rendimento totale (es. 0.35 = +35%).
        annualized_return: rendimento annualizzato.
        sharpe_ratio: Sharpe ratio annualizzato.
        max_drawdown: massimo drawdown (negativo, es. -0.30 = -30%).
        win_rate: percentuale di giorni positivi.
        profit_factor: rapporto guadagni/perdite.
        n_trades: numero di cambiamenti di posizione.
        days_long: giorni in posizione long.
        days_short: giorni in posizione short.
        days_flat: giorni flat.
        equity_curve: serie temporale della curva di equity (base=1.0).
        daily_returns: rendimenti giornalieri della strategia.
    """

    strategy_name: str
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    n_trades: int
    days_long: int
    days_short: int
    days_flat: int
    equity_curve: pd.Series = field(default_factory=pd.Series)
    daily_returns: pd.Series = field(default_factory=pd.Series)


class Backtest:
    """Esegue il backtest della strategia combinata GEX + flussi.

    Args:
        cfg: configurazione backtest (da settings.yaml).
    """

    def __init__(self, cfg: dict | None = None) -> None:
        settings  = get_settings()
        self._cfg = cfg or settings["backtest"]

    # ──────────────────────────────────────────────────────────────────────────
    # Signal generation
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_signals(
        self,
        merged_df: pd.DataFrame,
        gex_series: Optional[pd.Series] = None,
        active_barriers: Optional[list[dict]] = None,
        barrier_history: Optional[pd.DataFrame] = None,
        signal_model: Optional["SignalModel"] = None,
        composite: Optional["CompositeSignal"] = None,
    ) -> pd.Series:
        """Genera i segnali di trading (+1 long, -1 short, 0 flat).

        Priorità modalità:
          1. composite (4 pilastri): scoring 0-100 con barriere come pilastro pesato.
          2. signal_model (7 fattori): scoring 0-100, barriere come veto/override.
          3. regole legacy (GEX + flow threshold).

        Args:
            merged_df: DataFrame con ibit_flow_3d, btc_close, btc_return.
            gex_series: serie GEX totale (DatetimeIndex), opzionale.
            active_barriers: lista barriere attive dal DB, opzionale.
            signal_model: SignalModel per scoring multi-fattore, opzionale.
            composite: CompositeSignal a 4 pilastri, opzionale (ha priorità).

        Returns:
            pd.Series: segnali con DatetimeIndex.
        """
        cfg = self._cfg
        df  = merged_df.copy()

        # GEX: da serie fornita o da colonna nel df
        if gex_series is not None and not gex_series.empty:
            df = df.join(gex_series.rename("_gex"), how="left")
            df["_gex"] = df["_gex"].ffill().fillna(0.0)
        elif "total_gex" in df.columns:
            df["_gex"] = df["total_gex"]
        else:
            _log.warning("GEX non disponibile — segnali basati solo sui flussi")
            df["_gex"] = 0.0

        # Flussi 3 giorni
        if "ibit_flow_3d" not in df.columns:
            df["ibit_flow_3d"] = df["ibit_flow"].rolling(3, min_periods=1).sum() \
                if "ibit_flow" in df.columns else 0.0

        # Barriere attive
        barrier_excl = cfg.get("barrier_exclusion_pct", 5.0) / 100.0
        barrier_prices: set[float] = set()
        if active_barriers:
            for b in active_barriers:
                p = b.get("level_price_btc") or 0.0
                if p > 0:
                    barrier_prices.add(p)

        # ── Modalità Composite (4 pilastri) ────────────────────────────────────
        # Le barriere sono un PILASTRO pesato: nessun override/veto separato.
        if composite is not None:
            scores = composite.compute_series(
                df, active_barriers=active_barriers, barrier_history=barrier_history,
            )["composite_score"]
            signals = composite.signals_from_scores(scores)
            _log.info(
                "Composite — long=%d, risk_off=%d, flat=%d",
                (signals == 1).sum(), (signals == -1).sum(), (signals == 0).sum(),
            )
            return signals

        # ── Modalità SignalModel (multi-fattore) ───────────────────────────────
        if signal_model is not None:
            scores = signal_model.compute_series(df)
            signals = signal_model.signals_from_scores(scores)
            # Override: blocca LONG se vicino a barriera
            if barrier_prices:
                for ts, row in df.iterrows():
                    price = row.get("btc_close", 0.0)
                    if price > 0:
                        for bp in barrier_prices:
                            if abs(price - bp) / price < barrier_excl:
                                if signals[ts] == 1.0:
                                    signals[ts] = 0.0
                                break
            _log.info(
                "SignalModel — long=%d, risk_off=%d, flat=%d",
                (signals == 1).sum(), (signals == -1).sum(), (signals == 0).sum(),
            )
            return signals

        # ── Modalità legacy (fallback) ─────────────────────────────────────────
        long_gex_th   = cfg.get("long_gex_threshold",   0)
        long_flow_th  = cfg.get("long_flow_threshold_usd_m",  100) * 1e6
        short_gex_th  = cfg.get("short_gex_threshold",  0)
        short_flow_th = cfg.get("short_flow_threshold_usd_m", -200) * 1e6

        signals = pd.Series(0, index=df.index, dtype=float)
        for ts, row in df.iterrows():
            gex   = row.get("_gex", 0.0)
            flow3 = row.get("ibit_flow_3d", 0.0)
            price = row.get("btc_close", 0.0)
            near_barrier = False
            if price > 0 and barrier_prices:
                for bp in barrier_prices:
                    if abs(price - bp) / price < barrier_excl:
                        near_barrier = True
                        break
            if gex > long_gex_th and flow3 > long_flow_th and not near_barrier:
                signals[ts] = 1.0
            elif gex < short_gex_th and flow3 < short_flow_th:
                signals[ts] = -1.0

        _log.info(
            "Legacy — long=%d, short=%d, flat=%d",
            (signals == 1).sum(), (signals == -1).sum(), (signals == 0).sum(),
        )
        return signals

    # ──────────────────────────────────────────────────────────────────────────
    # Backtest execution
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_metrics(
        self,
        strategy_returns: pd.Series,
        strategy_name: str,
        signals: Optional[pd.Series] = None,
    ) -> BacktestMetrics:
        """Calcola le metriche di performance da una serie di rendimenti.

        Args:
            strategy_returns: rendimenti giornalieri (log returns).
            strategy_name: nome della strategia.
            signals: segnali opzionali per contare long/short/flat e applicare costi.

        Returns:
            BacktestMetrics.
        """
        rets = strategy_returns.dropna()
        if rets.empty:
            return BacktestMetrics(
                strategy_name=strategy_name,
                total_return=0.0, annualized_return=0.0,
                sharpe_ratio=0.0, max_drawdown=0.0,
                win_rate=0.0, profit_factor=0.0,
                n_trades=0, days_long=0, days_short=0, days_flat=0,
            )

        # Transaction costs: applicati solo sui giorni di cambio posizione
        cost_bps = float(self._cfg.get("transaction_cost_bps", 80))
        if cost_bps > 0 and signals is not None:
            cost_pct = cost_bps / 10_000
            position_changes = signals.diff().fillna(0).abs()
            rets = rets.copy()
            rets = rets - (position_changes.reindex(rets.index).fillna(0) * cost_pct)

        # Equity curve — log returns richiedono exp(cumsum), non (1+r).cumprod()
        equity = np.exp(rets.cumsum())

        # Rendimenti
        total_ret = float(equity.iloc[-1] - 1)
        n_years   = len(rets) / 365  # BTC tratta 365 giorni/anno (no weekend off)
        ann_ret   = float((1 + total_ret) ** (1 / n_years) - 1) if n_years > 0 else 0.0

        # Sharpe (rf=0, annualizzato) — sqrt(365) per crypto
        ann_vol   = float(rets.std() * (365 ** 0.5))
        sharpe    = ann_ret / ann_vol if ann_vol > 0 else 0.0

        # ── Sharpe sanity check ────────────────────────────────────────────
        # Reference: sharp_edges.md "overfitting-certainty" — Haircut Rule
        if sharpe > 5.0:
            _log.warning(
                "%s: Sharpe=%.2f — SEVERELY OVERFIT (expect ≤0.5 live). "
                "Reduce parameters or increase out-of-sample validation.",
                strategy_name, sharpe,
            )
        elif sharpe > 3.0:
            _log.warning(
                "%s: Sharpe=%.2f — LIKELY OVERFIT (Haircut Rule: expect ≤1.5 live). "
                "Verify with walk-forward backtest.",
                strategy_name, sharpe,
            )
        elif sharpe > 2.0:
            _log.info(
                "%s: Sharpe=%.2f — above typical. Walk-forward validation recommended.",
                strategy_name, sharpe,
            )

        # Max drawdown
        rolling_max = equity.cummax()
        drawdowns   = (equity - rolling_max) / rolling_max
        max_dd      = float(drawdowns.min())

        # Win rate
        win_rate = float((rets > 0).sum() / len(rets))

        # Profit factor
        gains  = rets[rets > 0].sum()
        losses = abs(rets[rets < 0].sum())
        pf     = float(gains / losses) if losses > 0 else float("inf")

        # Conteggio long/short/flat
        days_long = days_short = days_flat = 0
        n_trades  = 0
        if signals is not None:
            days_long  = int((signals == 1).sum())
            days_short = int((signals == -1).sum())
            days_flat  = int((signals == 0).sum())
            n_trades   = int(signals.diff().fillna(0).abs().gt(0).sum())

        return BacktestMetrics(
            strategy_name=strategy_name,
            total_return=total_ret,
            annualized_return=ann_ret,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=pf,
            n_trades=n_trades,
            days_long=days_long,
            days_short=days_short,
            days_flat=days_flat,
            equity_curve=equity,
            daily_returns=rets,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Null models
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _null_models(merged_df: pd.DataFrame, seed: int = 42) -> dict[str, pd.Series]:
        """Genera segnali nulli per confronto statistico.

        Reference: patterns.md "Proper Backtest Framework" — una strategia
        deve battere TUTTI i null model per essere considerata viable.

        Args:
            merged_df: DataFrame con btc_return e btc_close.
            seed: seed per riproducibilità del random.

        Returns:
            dict con chiavi "random_signal", "always_long", "momentum_20d"
            e valori pd.Series allineate all'indice di merged_df.
        """
        idx = merged_df.index
        n = len(merged_df)
        rng = np.random.default_rng(seed)

        random_signal = pd.Series(
            rng.choice([1.0, -1.0], size=n, p=[0.5, 0.5]),
            index=idx, dtype=float,
        )

        always_long = pd.Series(np.ones(n), index=idx, dtype=float)

        btc_close = merged_df.get("btc_close", pd.Series())
        if btc_close.empty or len(btc_close) < 21:
            momentum_20d = pd.Series(np.zeros(n), index=idx, dtype=float)
        else:
            ret_20d = btc_close.pct_change(20).shift(1)
            momentum_20d = pd.Series(
                np.where(ret_20d.fillna(0) > 0, 1.0, -1.0),
                index=idx,
            )

        return {
            "random_signal": random_signal,
            "always_long": always_long,
            "momentum_20d": momentum_20d,
        }

    def run(
        self,
        merged_df: pd.DataFrame,
        gex_series: Optional[pd.Series] = None,
        active_barriers: Optional[list[dict]] = None,
        barrier_history: Optional[pd.DataFrame] = None,
        signal_model: Optional["SignalModel"] = None,
        composite: Optional["CompositeSignal"] = None,
        include_null_models: bool = False,
    ) -> dict[str, BacktestMetrics]:
        """Esegue il backtest, benchmark buy-and-hold e opzionalmente null models.

        Args:
            merged_df: DataFrame con btc_return, ibit_flow, btc_close.
            gex_series: serie GEX totale (opzionale).
            active_barriers: barriere attive dal DB (opzionale, fallback).
            barrier_history: storico barriere da StructuredNotesDB (opzionale,
                prioritario su active_barriers per backtest accurato).
            signal_model: SignalModel per scoring multi-fattore (opzionale).
            composite: CompositeSignal a 4 pilastri (opzionale, ha priorità).
                Se né composite né signal_model sono forniti usa le regole legacy.
            include_null_models: se True, aggiunge 3 null models al risultato
                (random signal, always_long, momentum_20d).

        Returns:
            dict: "strategy" e "buy_and_hold" sempre presenti. Se
            include_null_models=True, anche "random_signal",
            "always_long", "momentum_20d".
        """
        df = merged_df.copy()
        if "btc_return" not in df.columns:
            _log.error("btc_return mancante nel DataFrame")
            return {}

        n_days = len(df)
        n_years_bh = n_days / 365
        if n_years_bh < 1:
            _log.warning(
                "Backtest su < 1 anno di dati (%.1f mesi) — statisticamente "
                "insufficiente per validazione. Richiesti ≥2 anni con più regimi.",
                n_years_bh * 12,
            )
        elif n_years_bh < 2:
            _log.info(
                "Backtest su %.1f anni di dati — copertura minima. "
                "Walk-forward validation raccomandata.",
                n_years_bh,
            )

        # Genera segnali (lag=1 per evitare look-ahead bias)
        signals = self._generate_signals(
            df, gex_series, active_barriers, barrier_history, signal_model, composite
        )
        signals_lagged = signals.shift(1).fillna(0)

        # Rendimenti strategia
        btc_rets   = df["btc_return"].dropna()
        strat_rets = btc_rets * signals_lagged.reindex(btc_rets.index).fillna(0)

        name = (
            "4-Pillar Strategy" if composite
            else ("Multi-Factor Strategy" if signal_model else "GEX+Flows Strategy")
        )
        strategy = self._compute_metrics(strat_rets, name, signals_lagged)
        bah      = self._compute_metrics(btc_rets, "Buy & Hold BTC")

        self._log_comparison(strategy, bah)

        results: dict[str, BacktestMetrics] = {
            "strategy": strategy,
            "buy_and_hold": bah,
        }

        if include_null_models:
            null_signals = self._null_models(df)
            null_names = {
                "random_signal": "Null: Random Signal",
                "always_long": "Null: Always Long",
                "momentum_20d": "Null: Momentum 20d",
            }
            for key, null_sig in null_signals.items():
                null_lagged = null_sig.shift(1).fillna(0)
                null_rets = btc_rets * null_lagged.reindex(btc_rets.index).fillna(0)
                results[key] = self._compute_metrics(
                    null_rets, null_names[key], null_lagged,
                )

        return results

    def _log_comparison(
        self,
        strategy: BacktestMetrics,
        bah: BacktestMetrics,
    ) -> None:
        """Logga il confronto tra strategia e buy-and-hold."""
        _log.info(
            "\nConfronto:\n"
            "%-25s %8s  %8s  %8s  %8s\n"
            "%-25s %7.1f%%  %7.1f%%  %7.2f   %7.1f%%\n"
            "%-25s %7.1f%%  %7.1f%%  %7.2f   %7.1f%%",
            "", "TotRet", "AnnRet", "Sharpe", "MaxDD",
            strategy.strategy_name,
            strategy.total_return * 100,
            strategy.annualized_return * 100,
            strategy.sharpe_ratio,
            strategy.max_drawdown * 100,
            bah.strategy_name,
            bah.total_return * 100,
            bah.annualized_return * 100,
            bah.sharpe_ratio,
            bah.max_drawdown * 100,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Regime coverage
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def regime_coverage(merged_df: pd.DataFrame) -> dict:
        """Verifica che il dataset copra regimi di mercato noti.

        Reference: sharp_edges.md "regime-blindness" — una strategia
        testata su un solo regime non è validata.

        Args:
            merged_df: DataFrame con DatetimeIndex.

        Returns:
            dict: periods_covered (bool per periodo), coverage_pct,
            is_adequate (bool, True se ≥2 periodi coperti).
        """
        required_periods = {
            "2020_covid": ("2020-02-15", "2020-04-15"),
            "2021_bull": ("2021-01-01", "2021-04-15"),
            "2022_bear": ("2022-01-01", "2022-12-31"),
            "2023_recovery": ("2023-01-01", "2023-06-30"),
            "2024_etf_launch": ("2024-01-10", "2024-03-31"),
        }

        coverage: dict[str, bool] = {}

        for period_name, (start, end) in required_periods.items():
            try:
                period_data = merged_df.loc[start:end]
                coverage[period_name] = len(period_data) > 5
            except Exception:
                coverage[period_name] = False

        n_covered = sum(coverage.values())
        coverage_pct = n_covered / len(required_periods) if required_periods else 0.0

        is_adequate = n_covered >= 2

        if not is_adequate:
            _log.warning(
                "Regime coverage insufficiente: %d/%d periodi coperti. "
                "Il backtest potrebbe riflettere un solo regime di mercato.",
                n_covered, len(required_periods),
            )
        else:
            _log.info("Regime coverage: %d/%d periodi — adeguato.", n_covered, len(required_periods))

        return {
            "periods_covered": coverage,
            "coverage_pct": round(coverage_pct, 2),
            "n_covered": n_covered,
            "is_adequate": is_adequate,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Plot
    # ──────────────────────────────────────────────────────────────────────────

    def plot(self, results: dict[str, BacktestMetrics]):
        """Genera grafico Plotly con equity curve + statistiche.

        Args:
            results: dict da run().

        Returns:
            plotly.graph_objects.Figure | None.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            from src.config import get_settings
            theme = get_settings()["dashboard"]["theme"]
        except ImportError:
            return None

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=["Equity Curve", "Daily Returns"],
            vertical_spacing=0.1,
        )

        colors = {"strategy": theme["positive"], "buy_and_hold": theme["neutral"]}

        for key, metrics in results.items():
            if metrics.equity_curve.empty:
                continue
            fig.add_trace(go.Scatter(
                x=metrics.equity_curve.index,
                y=metrics.equity_curve,
                name=metrics.strategy_name,
                line=dict(color=colors.get(key, "white"), width=2),
            ), row=1, col=1)
            fig.add_trace(go.Bar(
                x=metrics.daily_returns.index,
                y=metrics.daily_returns,
                name=f"{metrics.strategy_name} daily",
                marker_color=colors.get(key, "white"),
                opacity=0.5,
                showlegend=False,
            ), row=2, col=1)

        fig.update_layout(
            paper_bgcolor=theme["background"],
            plot_bgcolor=theme["background"],
            font=dict(color=theme["text"]),
            title="Backtest: GEX+Flows Strategy vs Buy & Hold BTC",
            height=700,
        )
        return fig

    def summary_table(self, results: dict[str, BacktestMetrics]) -> pd.DataFrame:
        """Crea un DataFrame comparativo delle metriche.

        Args:
            results: dict da run().

        Returns:
            pd.DataFrame: tabella con righe=strategie, colonne=metriche.
        """
        rows = []
        for key, m in results.items():
            rows.append({
                "Strategia":       m.strategy_name,
                "Return Totale":   f"{m.total_return*100:.1f}%",
                "Return Annuo":    f"{m.annualized_return*100:.1f}%",
                "Sharpe":          f"{m.sharpe_ratio:.2f}",
                "Max Drawdown":    f"{m.max_drawdown*100:.1f}%",
                "Win Rate":        f"{m.win_rate*100:.1f}%",
                "Profit Factor":   f"{m.profit_factor:.2f}" if m.profit_factor < 100 else "∞",
                "N Trades":        m.n_trades,
                "Giorni Long":     m.days_long,
                "Giorni Short":    m.days_short,
                "Giorni Flat":     m.days_flat,
            })
        return pd.DataFrame(rows).set_index("Strategia")
