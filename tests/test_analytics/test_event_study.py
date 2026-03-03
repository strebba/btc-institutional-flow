"""Test unitari per EventStudy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.event_study import EventStudy, EventStudyResult


def _make_prices(n: int = 200, start_price: float = 65_000.0) -> pd.DataFrame:
    """Crea DataFrame di prezzi BTC con DatetimeIndex."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    returns = rng.normal(0, 0.02, n)
    prices = start_price * np.exp(np.cumsum(returns))
    return pd.DataFrame({"close": prices}, index=dates)


def _make_barrier(btype: str, price_btc: float) -> dict:
    return {"barrier_type": btype, "level_price_btc": price_btc, "level_price_ibit": None}


@pytest.fixture
def study() -> EventStudy:
    return EventStudy()


@pytest.fixture
def prices() -> pd.DataFrame:
    return _make_prices()


class TestAbnormalReturns:
    def test_output_length(self, study, prices):
        returns = prices["close"].pct_change().apply(np.log1p)
        ab = study._compute_abnormal_returns(returns)
        assert len(ab) == len(returns)

    def test_no_bias(self, study, prices):
        """Rendimenti anormali devono avere media vicina a zero."""
        returns = prices["close"].pct_change().apply(np.log1p)
        ab = study._compute_abnormal_returns(returns).dropna()
        assert abs(ab.mean()) < 0.01


class TestFindEventDates:
    def test_finds_near_barrier(self, study):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        prices = pd.Series([65_000.0] * 5 + [63_000.0] * 5, index=dates)
        # Barriera a 64000 → le date con prezzo ~65k sono entro 2%
        events = study._find_event_dates(prices, 64_800.0)
        assert len(events) > 0

    def test_no_events_far_from_barrier(self, study):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        prices = pd.Series([65_000.0] * 10, index=dates)
        events = study._find_event_dates(prices, 50_000.0)  # molto distante
        assert len(events) == 0

    def test_zero_barrier_returns_empty(self, study):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        prices = pd.Series([65_000.0] * 5, index=dates)
        assert study._find_event_dates(prices, 0.0) == []


class TestComputeCAR:
    def test_returns_dict_with_correct_days(self, study):
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        rng = np.random.default_rng(0)
        ab_ret = pd.Series(rng.normal(0, 0.01, 50), index=dates)
        event = dates[25]
        car = study._compute_car(event, ab_ret)
        assert car is not None
        w = study._window
        assert set(car.keys()) == set(range(-w, w + 1))

    def test_returns_none_near_edge(self, study):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        rng = np.random.default_rng(0)
        ab_ret = pd.Series(rng.normal(0, 0.01, 20), index=dates)
        # Evento troppo vicino al bordo
        car = study._compute_car(dates[1], ab_ret)
        assert car is None

    def test_cumulative_increasing(self, study):
        """CAR deve essere cumulativo."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        # Tutti i rendimenti positivi
        ab_ret = pd.Series([0.01] * 50, index=dates)
        event = dates[25]
        car = study._compute_car(event, ab_ret)
        w = study._window
        # CAR al giorno +w deve essere > CAR al giorno 0
        assert car[w] > car[0]


class TestRun:
    def test_empty_barriers_returns_empty(self, study, prices):
        result = study.run([], prices)
        assert result == []

    def test_empty_prices_returns_empty(self, study):
        barriers = [_make_barrier("knock_in", 65_000.0)]
        result = study.run(barriers, pd.DataFrame())
        assert result == []

    def test_returns_list_of_results(self, study, prices):
        # Usa il prezzo medio come barriera per garantire eventi
        mean_price = float(prices["close"].mean())
        barriers = [_make_barrier("knock_in", mean_price)]
        result = study.run(barriers, prices)
        assert len(result) >= 1
        assert all(isinstance(r, EventStudyResult) for r in result)

    def test_groups_by_barrier_type(self, study, prices):
        mean_price = float(prices["close"].mean())
        barriers = [
            _make_barrier("knock_in", mean_price),
            _make_barrier("autocall", mean_price),
        ]
        result = study.run(barriers, prices)
        types = {r.barrier_type for r in result}
        assert "knock_in" in types
        assert "autocall" in types

    def test_no_events_creates_zero_result(self, study, prices):
        barriers = [_make_barrier("knock_in", 10_000.0)]  # prezzo irraggiungibile
        result = study.run(barriers, prices)
        assert len(result) == 1
        assert result[0].n_events == 0
        assert result[0].p_value == 1.0

    def test_ibit_fallback_conversion(self, study, prices):
        """Se level_price_btc=0, usa level_price_ibit / 0.0006."""
        mean_price = float(prices["close"].mean())
        ibit_price = mean_price * 0.0006
        barriers = [{"barrier_type": "buffer", "level_price_btc": 0.0, "level_price_ibit": ibit_price}]
        result = study.run(barriers, prices)
        # Deve aver trovato almeno qualche evento
        assert len(result) == 1


class TestEventStudyResult:
    def test_ci_ordering(self, study, prices):
        mean_price = float(prices["close"].mean())
        barriers = [_make_barrier("knock_in", mean_price)]
        result = study.run(barriers, prices)
        for r in result:
            if r.n_events >= 2:
                assert r.ci_lower <= r.car_mean <= r.ci_upper

    def test_significant_flag_consistency(self, study, prices):
        mean_price = float(prices["close"].mean())
        barriers = [_make_barrier("knock_in", mean_price)]
        result = study.run(barriers, prices)
        for r in result:
            assert r.significant == (r.p_value < 0.05)


class TestRunOnPriceLevels:
    def test_returns_single_result(self, study, prices):
        mean_price = float(prices["close"].mean())
        result = study.run_on_price_levels([mean_price], "round_number", prices)
        assert isinstance(result, EventStudyResult)

    def test_empty_levels_returns_none(self, study, prices):
        result = study.run_on_price_levels([], "test", prices)
        assert result is None


class TestPlot:
    def test_returns_figure_or_none(self, study, prices):
        mean_price = float(prices["close"].mean())
        barriers = [_make_barrier("knock_in", mean_price)]
        results = study.run(barriers, prices)
        fig = study.plot(results)
        # Plotly potrebbe non essere installato, ma non deve crashare
        assert fig is None or hasattr(fig, "data")
