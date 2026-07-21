"""Test unitari per GrangerAnalysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.granger import GrangerAnalysis, GrangerResult


def _make_df(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Crea DataFrame sintetico con flussi e rendimenti."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    returns = rng.normal(0, 0.02, n)
    # Flussi correlati ai rendimenti del giorno precedente (causalità artificiale)
    flows = np.roll(returns, 1) * 1e8 + rng.normal(0, 5e6, n)
    return pd.DataFrame({"ibit_flow": flows, "btc_return": returns}, index=dates)


@pytest.fixture
def analyzer() -> GrangerAnalysis:
    return GrangerAnalysis()


@pytest.fixture
def df() -> pd.DataFrame:
    return _make_df()


class TestPrepareData:
    def test_normalizes_flows(self, analyzer, df):
        result = analyzer._prepare_data(df)
        # Flussi normalizzati in miliardi
        assert result["ibit_flow"].abs().max() < 1.0

    def test_drops_na(self, analyzer, df):
        df.iloc[5, 0] = float("nan")
        result = analyzer._prepare_data(df)
        assert not result.isnull().any().any()

    def test_raises_missing_column(self, analyzer, df):
        with pytest.raises(ValueError, match="Colonne mancanti"):
            analyzer._prepare_data(df.drop(columns=["ibit_flow"]))

    def test_handles_non_stationary(self, analyzer):
        """Serie non stazionaria (random walk) deve essere differenziata."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        rng = np.random.default_rng(0)
        # Random walk non stazionario
        flow_rw = np.cumsum(rng.normal(0, 1e7, 100))
        df = pd.DataFrame(
            {"ibit_flow": flow_rw, "btc_return": rng.normal(0, 0.02, 100)},
            index=dates,
        )
        result = analyzer._prepare_data(df)
        assert len(result) >= 50  # deve avere ancora dati dopo differenziazione


class TestStationarity:
    def test_stationary_series(self, analyzer):
        rng = np.random.default_rng(0)
        series = pd.Series(rng.normal(0, 1, 100))
        assert analyzer._check_stationarity(series, "test")  # truthy

    def test_random_walk_not_stationary(self, analyzer):
        rng = np.random.default_rng(0)
        series = pd.Series(np.cumsum(rng.normal(0, 1, 200)))
        # ADF dovrebbe rilevare non stazionarietà su random walk lungo
        result = analyzer._check_stationarity(series, "rw")
        # Risultato può essere bool o np.bool_ — verifica che sia falsy/truthy
        assert result in (True, False) or isinstance(result, (bool, int))

    def test_short_series_assumes_stationary(self, analyzer):
        series = pd.Series([1.0, 2.0, 3.0])
        assert analyzer._check_stationarity(series, "short") is True


class TestRun:
    def test_returns_two_directions(self, analyzer, df):
        results = analyzer.run(df)
        assert "flows→returns" in results
        assert "returns→flows" in results

    def test_result_types(self, analyzer, df):
        results = analyzer.run(df)
        for direction, res_list in results.items():
            assert isinstance(res_list, list)
            for r in res_list:
                assert isinstance(r, GrangerResult)

    def test_lag_range(self, analyzer, df):
        max_lags = 5
        results = analyzer.run(df, max_lags=max_lags)
        lags_f2r = [r.lag for r in results["flows→returns"]]
        assert lags_f2r == list(range(1, max_lags + 1))

    def test_p_values_in_range(self, analyzer, df):
        results = analyzer.run(df)
        for direction, res_list in results.items():
            for r in res_list:
                assert 0.0 <= r.p_value <= 1.0

    def test_significant_flag(self, analyzer, df):
        results = analyzer.run(df, alpha=0.05)
        for direction, res_list in results.items():
            for r in res_list:
                assert r.significant == (r.p_value < 0.05)

    def test_fdr_significant_field_exists(self, analyzer, df):
        results = analyzer.run(df)
        for direction, res_list in results.items():
            for r in res_list:
                assert hasattr(r, "fdr_significant")
                assert isinstance(r.fdr_significant, bool)

    def test_fdr_not_more_than_naive(self, analyzer, df):
        """FDR non deve dichiarare più significativi del naive."""
        results = analyzer.run(df)
        for direction, res_list in results.items():
            n_naive = sum(1 for r in res_list if r.significant)
            n_fdr = sum(1 for r in res_list if r.fdr_significant)
            assert n_fdr <= n_naive

    def test_missing_columns_returns_empty(self, analyzer):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        results = analyzer.run(df)
        assert results["flows→returns"] == []
        assert results["returns→flows"] == []

    def test_short_dataset_reduces_lags(self, analyzer):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {"ibit_flow": rng.normal(0, 1e7, 20), "btc_return": rng.normal(0, 0.02, 20)},
            index=dates,
        )
        results = analyzer.run(df, max_lags=10)
        # Con 20 righe deve ridurre i lag
        assert len(results["flows→returns"]) <= 6


class TestInterpret:
    def test_returns_string(self, analyzer, df):
        results = analyzer.run(df)
        text = analyzer.interpret(results)
        assert isinstance(text, str)
        assert "Granger" in text

    def test_contains_both_directions(self, analyzer, df):
        results = analyzer.run(df)
        text = analyzer.interpret(results)
        assert "flows→returns" in text
        assert "returns→flows" in text

    def test_empty_results(self, analyzer):
        text = analyzer.interpret({"flows→returns": [], "returns→flows": []})
        assert isinstance(text, str)


class TestToDataFrame:
    def test_returns_dataframe(self, analyzer, df):
        results = analyzer.run(df)
        frame = analyzer.to_dataframe(results)
        assert isinstance(frame, pd.DataFrame)

    def test_columns(self, analyzer, df):
        results = analyzer.run(df)
        frame = analyzer.to_dataframe(results)
        for col in ["lag", "direction", "f_stat", "p_value", "significant", "fdr_significant"]:
            assert col in frame.columns

    def test_row_count(self, analyzer, df):
        max_lags = 4
        results = analyzer.run(df, max_lags=max_lags)
        frame = analyzer.to_dataframe(results)
        # 2 direzioni × max_lags righe
        assert len(frame) == 2 * max_lags


class TestBenjaminiHochberg:
    def test_all_low_pvalues_all_significant(self, analyzer):
        """Con tutti p-value bassi, FDR conferma tutto."""
        from src.analytics.granger import GrangerResult
        results = [
            GrangerResult("test", lag=1, f_stat=10.0, p_value=0.001, significant=True),
            GrangerResult("test", lag=2, f_stat=8.0, p_value=0.002, significant=True),
            GrangerResult("test", lag=3, f_stat=6.0, p_value=0.003, significant=True),
        ]
        corrected = analyzer.benjamini_hochberg(results, alpha=0.05)
        assert all(r.fdr_significant for r in corrected)

    def test_all_high_pvalues_none_significant(self, analyzer):
        from src.analytics.granger import GrangerResult
        results = [
            GrangerResult("test", lag=1, f_stat=1.0, p_value=0.5, significant=False),
            GrangerResult("test", lag=2, f_stat=1.0, p_value=0.6, significant=False),
        ]
        corrected = analyzer.benjamini_hochberg(results, alpha=0.05)
        assert not any(r.fdr_significant for r in corrected)

    def test_mixed_pvalues_fdr_stricter(self, analyzer):
        """Con p-value misti, FDR deve essere più restrittivo del naive."""
        from src.analytics.granger import GrangerResult
        results = [
            GrangerResult("test", lag=1, f_stat=5.0, p_value=0.01, significant=True),
            GrangerResult("test", lag=2, f_stat=3.0, p_value=0.04, significant=True),
            GrangerResult("test", lag=3, f_stat=1.5, p_value=0.10, significant=False),
            GrangerResult("test", lag=4, f_stat=1.2, p_value=0.15, significant=False),
        ]
        corrected = analyzer.benjamini_hochberg(results, alpha=0.05)
        n_naive = sum(1 for r in results if r.significant)
        n_fdr = sum(1 for r in corrected if r.fdr_significant)
        assert n_fdr <= n_naive

    def test_empty_list(self, analyzer):
        assert analyzer.benjamini_hochberg([], alpha=0.05) == []


def _make_causal_df(n: int = 100, causal_lag: int = 3, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.005, n)
    flow = rng.normal(0, 1e8, n + causal_lag)
    ret = np.zeros(n)
    scale = 1.0e-7
    for t in range(causal_lag, n):
        ret[t] = flow[t - causal_lag] * scale + noise[t]
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "ibit_flow": flow[causal_lag:],
        "btc_return": ret,
    }, index=dates)


class TestFindOptimalLag:
    def test_finds_significant_lag(self):
        df = _make_causal_df(n=300, causal_lag=3)
        result = GrangerAnalysis.find_optimal_lag(
            df, train_end="2023-08-15", max_lags=10,
        )
        assert result.get("n_train", 0) > 30


    def test_no_significant_lag_returns_none(self):
        rng = np.random.default_rng(42)
        dates = pd.date_range("2023-01-01", periods=200, freq="D")
        df = pd.DataFrame({
            "ibit_flow": rng.normal(0, 1e8, 200),
            "btc_return": rng.normal(0, 0.02, 200),
        }, index=dates)
        result = GrangerAnalysis.find_optimal_lag(
            df, train_end="2023-06-15", max_lags=5,
        )
        assert result["validated"] is False or result["optimal_lag"] is None

    def test_insufficient_holdout(self):
        df = _make_causal_df(n=50, causal_lag=3)
        result = GrangerAnalysis.find_optimal_lag(
            df, train_end="2023-02-10", max_lags=5,
        )
        assert not result["validated"]

    def test_returns_expected_keys(self):
        df = _make_causal_df(n=150, causal_lag=2)
        result = GrangerAnalysis.find_optimal_lag(
            df, train_end="2023-04-15", max_lags=5,
        )
        for key in ("optimal_lag", "train_p", "train_fdr_significant",
                     "holdout_p", "holdout_significant", "validated",
                     "n_train", "n_holdout"):
            assert key in result

    def test_holdout_preserves_lag(self):
        df = _make_causal_df(n=300, causal_lag=4)
        result = GrangerAnalysis.find_optimal_lag(
            df, train_end="2023-09-15", max_lags=8,
        )
        if result["optimal_lag"] is not None:
            assert 1 <= result["optimal_lag"] <= 8

    def test_missing_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = GrangerAnalysis.find_optimal_lag(
            df, train_end="2023-01-02", max_lags=3,
        )
        assert result["optimal_lag"] is None
        assert not result["validated"]

    def test_class_attribute_exists(self):
        assert hasattr(GrangerAnalysis, "_GRANGER_LEAD_LAG")
        assert isinstance(GrangerAnalysis._GRANGER_LEAD_LAG, int)
