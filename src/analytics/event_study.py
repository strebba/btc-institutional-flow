"""Event study sui barrier levels delle note strutturate IBIT.

Per ogni barriera estratta da EDGAR:
  - Identifica le date in cui BTC è entro ±2% del livello
  - Calcola rendimenti anormali (vs media mobile 30gg)
  - Cumula in CAR per finestra [-5, +5] giorni
  - Test t su H0: CAR = 0

Output: tabella con CAR medio per tipo di barriera (knock_in, autocall, buffer).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.config import get_settings, setup_logging

_log = setup_logging("analytics.event_study")


@dataclass
class EventStudyResult:
    """Risultato dell'event study per un tipo di barriera.

    Attributes:
        barrier_type: tipo di barriera analizzata.
        n_events: numero di eventi rilevati.
        car_mean: CAR medio su tutti gli eventi.
        car_std: deviazione standard del CAR.
        t_stat: t-statistic del test H0: CAR=0.
        p_value: p-value del t-test.
        significant: True se p < 0.05.
        car_by_day: CAR medio per giorno nella finestra [-window, +window].
        ci_lower: lower 95% confidence interval.
        ci_upper: upper 95% confidence interval.
    """

    barrier_type: str
    n_events: int
    car_mean: float
    car_std: float
    t_stat: float
    p_value: float
    significant: bool
    car_by_day: dict[int, float] = field(default_factory=dict)
    ci_lower: float = 0.0
    ci_upper: float = 0.0


class EventStudy:
    """Analisi event study sui barrier levels delle note strutturate IBIT.

    Args:
        cfg: configurazione analytics (da settings.yaml).
    """

    def __init__(self, cfg: dict | None = None) -> None:
        settings = get_settings()
        self._cfg = cfg or settings["analytics"]
        self._window = self._cfg.get("event_window_days", 5)
        self._proximity_pct = self._cfg.get("barrier_proximity_pct", 2.0)
        self._ibit_btc_ratio = self._cfg.get("ibit_btc_ratio", 0.0006)

    # ──────────────────────────────────────────────────────────────────────────
    # Abnormal returns
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_abnormal_returns(
        self,
        returns: pd.Series,
        benchmark_window: int = 30,
    ) -> pd.Series:
        """Calcola i rendimenti anormali rispetto alla media mobile.

        AR_t = R_t - rolling_mean(R, window)

        Args:
            returns: serie di rendimenti giornalieri (log).
            benchmark_window: finestra per la media mobile (default 30gg).

        Returns:
            pd.Series: rendimenti anormali.
        """
        rolling_mean = returns.rolling(benchmark_window, min_periods=10).mean()
        return returns - rolling_mean

    # ──────────────────────────────────────────────────────────────────────────
    # Event identification
    # ──────────────────────────────────────────────────────────────────────────

    def _find_event_dates(
        self,
        btc_prices: pd.Series,
        barrier_price_btc: float,
    ) -> list[pd.Timestamp]:
        """Trova le date in cui BTC è entro ±proximity_pct% della barriera.

        Args:
            btc_prices: serie prezzi BTC (DatetimeIndex).
            barrier_price_btc: prezzo BTC corrispondente alla barriera.

        Returns:
            list[pd.Timestamp]: date degli eventi.
        """
        if barrier_price_btc <= 0:
            return []

        proximity = self._proximity_pct / 100.0
        near = btc_prices[
            (btc_prices >= barrier_price_btc * (1 - proximity)) &
            (btc_prices <= barrier_price_btc * (1 + proximity))
        ]
        return near.index.tolist()

    # ──────────────────────────────────────────────────────────────────────────
    # CAR calculation
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_car(
        self,
        event_date: pd.Timestamp,
        abnormal_returns: pd.Series,
    ) -> Optional[dict[int, float]]:
        """Calcola il CAR cumulativo per la finestra intorno a un evento.

        Args:
            event_date: data dell'evento.
            abnormal_returns: serie dei rendimenti anormali.

        Returns:
            dict[int, float] | None: CAR per ogni giorno nella finestra, o None se
                la finestra non è completamente coperta.
        """
        w = self._window
        idx  = abnormal_returns.index
        pos  = idx.get_loc(event_date) if event_date in idx else None

        if pos is None:
            # Cerca la data più vicina
            diffs = abs(idx - event_date)
            pos   = diffs.argmin()
            if diffs[pos].days > 3:
                return None

        start = pos - w
        end   = pos + w + 1

        if start < 0 or end > len(idx):
            return None

        window_ar = abnormal_returns.iloc[start:end]
        car: dict[int, float] = {}
        cumulative = 0.0
        for day_offset, (_, ar_val) in zip(range(-w, w + 1), window_ar.items()):
            if pd.notna(ar_val):
                cumulative += ar_val
            car[day_offset] = cumulative

        return car

    # ──────────────────────────────────────────────────────────────────────────
    # Main analysis
    # ──────────────────────────────────────────────────────────────────────────

    def run(
        self,
        barriers: list[dict],
        btc_prices: pd.DataFrame,
    ) -> list[EventStudyResult]:
        """Esegue l'event study su tutti i barrier levels.

        Args:
            barriers: lista di dict dal DB (get_active_barriers()).
                Ogni dict deve avere: barrier_type, level_price_btc.
            btc_prices: DataFrame con DatetimeIndex e colonna "close" (BTC prices).

        Returns:
            list[EventStudyResult]: un risultato per ogni tipo di barriera.
        """
        if btc_prices.empty or not barriers:
            _log.warning("Dati insufficienti per event study")
            return []

        # Calcola rendimenti anormali
        returns = btc_prices["close"].pct_change().apply(np.log1p)
        ab_ret  = self._compute_abnormal_returns(returns)

        # Raggruppa barriere per tipo
        by_type: dict[str, list[dict]] = {}
        for b in barriers:
            btype = b.get("barrier_type", "unknown")
            by_type.setdefault(btype, []).append(b)

        results: list[EventStudyResult] = []

        for btype, barrier_list in by_type.items():
            all_cars: list[dict[int, float]] = []

            for barrier in barrier_list:
                price_btc = barrier.get("level_price_btc") or 0.0
                if price_btc <= 0:
                    # Stima da IBIT se disponibile
                    price_ibit = barrier.get("level_price_ibit") or 0.0
                    # ratio configurabile in settings.yaml → analytics.ibit_btc_ratio
                    price_btc = price_ibit / self._ibit_btc_ratio if price_ibit > 0 else 0.0

                if price_btc <= 0:
                    continue

                event_dates = self._find_event_dates(btc_prices["close"], price_btc)
                _log.debug(
                    "Barriera %s @ $%.0f BTC: %d eventi trovati",
                    btype, price_btc, len(event_dates),
                )

                for event_date in event_dates:
                    car = self._compute_car(event_date, ab_ret)
                    if car:
                        all_cars.append(car)

            if not all_cars:
                _log.warning("Nessun evento trovato per %s", btype)
                results.append(EventStudyResult(
                    barrier_type=btype,
                    n_events=0,
                    car_mean=0.0,
                    car_std=0.0,
                    t_stat=0.0,
                    p_value=1.0,
                    significant=False,
                ))
                continue

            # Aggregazione CAR
            w = self._window
            car_finals = [c.get(w, 0.0) for c in all_cars]  # CAR al giorno +window
            car_mean   = float(np.mean(car_finals))
            car_std    = float(np.std(car_finals, ddof=1)) if len(car_finals) > 1 else 0.0

            # T-test H0: CAR = 0
            if len(car_finals) >= 2 and car_std > 0:
                t_stat, p_value = stats.ttest_1samp(car_finals, 0.0)
            else:
                t_stat, p_value = 0.0, 1.0

            # CAR medio per giorno
            car_by_day: dict[int, float] = {}
            for day in range(-w, w + 1):
                vals = [c.get(day, float("nan")) for c in all_cars]
                valid = [v for v in vals if not np.isnan(v)]
                car_by_day[day] = float(np.mean(valid)) if valid else 0.0

            # Confidence interval (95%)
            n = len(car_finals)
            se = car_std / (n ** 0.5) if n > 1 else 0.0
            ci_lower = car_mean - 1.96 * se
            ci_upper = car_mean + 1.96 * se

            results.append(EventStudyResult(
                barrier_type=btype,
                n_events=len(all_cars),
                car_mean=car_mean,
                car_std=car_std,
                t_stat=float(t_stat),
                p_value=float(p_value),
                significant=p_value < 0.05,
                car_by_day=car_by_day,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
            ))

            _log.info(
                "Event study %s: n=%d, CAR=%.4f, t=%.2f, p=%.4f %s",
                btype, len(all_cars), car_mean, t_stat, p_value,
                "***" if p_value < 0.05 else "",
            )

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Synthetic event study (quando non ci sono barriere reali nel range)
    # ──────────────────────────────────────────────────────────────────────────

    def run_on_price_levels(
        self,
        price_levels: list[float],
        level_type: str,
        btc_prices: pd.DataFrame,
    ) -> Optional[EventStudyResult]:
        """Esegue l'event study su una lista di prezzi BTC arbitrari.

        Utile per testare round numbers, livelli tecnici, ecc.

        Args:
            price_levels: lista di prezzi BTC da testare come eventi.
            level_type: nome del tipo di livello (per il report).
            btc_prices: DataFrame con DatetimeIndex e colonna "close".

        Returns:
            EventStudyResult | None.
        """
        barriers = [
            {"barrier_type": level_type, "level_price_btc": p, "level_price_ibit": None}
            for p in price_levels
        ]
        results = self.run(barriers, btc_prices)
        return results[0] if results else None

    # ──────────────────────────────────────────────────────────────────────────
    # Plot
    # ──────────────────────────────────────────────────────────────────────────

    def plot(self, results: list[EventStudyResult]):
        """Genera grafico Plotly con CAR medio ± CI per ogni tipo di barriera.

        Args:
            results: lista di EventStudyResult.

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

        w = self._window
        days = list(range(-w, w + 1))

        fig = go.Figure()
        colors = [theme["positive"], theme["negative"], theme["neutral"], "orange", "purple"]

        for i, res in enumerate(results):
            if res.n_events == 0:
                continue
            color = colors[i % len(colors)]
            car_vals = [res.car_by_day.get(d, 0.0) for d in days]
            ci_upper_vals = [v + (res.ci_upper - res.car_mean) for v in car_vals]
            ci_lower_vals = [v + (res.ci_lower - res.car_mean) for v in car_vals]

            label = f"{res.barrier_type} (n={res.n_events})"
            if res.significant:
                label += " ***"

            fig.add_trace(go.Scatter(
                x=days, y=car_vals,
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
            ))
            fig.add_trace(go.Scatter(
                x=days + days[::-1],
                y=ci_upper_vals + ci_lower_vals[::-1],
                fill="toself",
                fillcolor=color,
                opacity=0.15,
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
            ))

        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=0, line_dash="dot",  line_color="white", annotation_text="Event")
        fig.update_layout(
            title="Cumulative Abnormal Returns intorno ai Barrier Levels",
            xaxis_title="Giorni dall'evento",
            yaxis_title="CAR (log return cumulativo)",
            paper_bgcolor=theme["background"],
            plot_bgcolor=theme["background"],
            font=dict(color=theme["text"]),
        )
        return fig
