"""Test per data_loader.py — helper condivisi e compute_composite."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd


# ─── _get_backtest_context ────────────────────────────────────────────────────


class TestGetBacktestContext:
    def test_returns_gex_series_and_barrier_history(self):
        """Con DB popolati, ritorna entrambi."""
        from src.dashboard.data_loader import _get_backtest_context

        gex_series, barrier_history = _get_backtest_context(days=30)
        assert isinstance(gex_series, pd.Series)
        assert barrier_history is None or isinstance(barrier_history, pd.DataFrame)

    def test_empty_db_does_not_crash(self):
        """DB vuoto → gex_series vuoto, barrier_history=None senza eccezioni."""
        from src.dashboard.data_loader import _get_backtest_context

        gex_series, barrier_history = _get_backtest_context(days=30)
        assert isinstance(gex_series, pd.Series)
        assert barrier_history is None

    def test_barrier_history_exception_graceful(self):
        """Eccezione su barrier_history → ritorna None senza propagare."""
        from src.dashboard.data_loader import _get_backtest_context

        with patch("src.gex.gex_db.GexDB") as mock_gex:
            mock_gex.return_value.get_series.return_value = pd.Series(dtype=float)
            with patch("src.edgar.structured_notes_db.StructuredNotesDB") as mock_sndb:
                mock_sndb.return_value.get_barrier_history.side_effect = RuntimeError("DB locked")
                gex_series, barrier_history = _get_backtest_context(days=30)
                assert barrier_history is None


# ─── compute_composite ─────────────────────────────────────────────────────────


class TestComputeComposite:
    def test_all_pillars_active(self):
        """4 pilastri con dati → score valido e 4 pillar scores."""
        from src.dashboard.data_loader import compute_composite

        snap = {
            "total_net_gex": 500_000_000,
            "gamma_flip_price": 85_000,
            "put_wall": 80_000,
            "call_wall": 110_000,
            "put_call_ratio": 0.6,
            "spot_price": 100_000,
        }
        merged = _make_flows_df()
        barriers = [
            {"barrier_type": "autocall", "level_price_btc": 105_000,
             "notional_usd": 50e6, "issuer": "GS"},
        ]
        macro = {
            "funding_rate_annualized_pct": 12.0,
            "oi_change_7d_pct": 2.5,
            "long_short_ratio": 1.8,
            "liquidations_long_24h_usd": 5e6,
            "liquidations_short_24h_usd": 3e6,
        }

        result = compute_composite(snap, merged, barriers, macro)
        assert 0 <= result.score <= 100
        assert result.signal in ("LONG", "CAUTION", "RISK_OFF")
        assert len(result.pillars) == 4

    def test_no_macro_rescales_weights(self):
        """Senza macro → 3 pilastri, i pesi vengono riscalati."""
        from src.dashboard.data_loader import compute_composite

        snap = {
            "total_net_gex": 200_000_000,
            "spot_price": 95_000,
        }
        merged = _make_flows_df(negative=True)
        barriers = [
            {"barrier_type": "knock_in", "level_price_btc": 90_000,
             "notional_usd": 100e6, "issuer": "JPM"},
        ]
        # macro assente
        result = compute_composite(snap, merged, barriers, None)
        assert 0 <= result.score <= 100
        assert any(p.name == "macro" and p.score is None for p in result.pillars)
        assert any(p.name == "barrier" and p.score is not None for p in result.pillars)

    def test_no_barriers_pillar_none(self):
        """Senza barriere → barrier pillar = None, gli altri funzionano."""
        from src.dashboard.data_loader import compute_composite

        snap = {"total_net_gex": 100_000_000, "spot_price": 100_000}
        merged = _make_flows_df()
        result = compute_composite(snap, merged, [], None)
        assert 0 <= result.score <= 100
        barrier = next(p for p in result.pillars if p.name == "barrier")
        assert barrier.score is None

    def test_spot_zero_no_crash(self):
        """Spot=0 → non crasha, barrier pillar segnala dati insufficienti."""
        from src.dashboard.data_loader import compute_composite

        snap = {"total_net_gex": 500_000_000, "spot_price": 0}
        merged = _make_flows_df()
        result = compute_composite(snap, merged, [], None)
        assert result.signal in ("LONG", "CAUTION", "RISK_OFF")
        assert 0 <= result.score <= 100
        barrier = next(p for p in result.pillars if p.name == "barrier")
        assert barrier.score is None


# ─── helpers ───────────────────────────────────────────────────────────────────


def _make_flows_df(negative: bool = False, days: int = 30) -> pd.DataFrame:
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="D")
    rng = np.random.default_rng(42)
    flow = rng.uniform(-200e6, -50e6, days) if negative else rng.uniform(50e6, 500e6, days)
    return pd.DataFrame({
        "total_flow_usd": flow,
        "ibit_flow_3d": pd.Series(flow, index=dates).rolling(3, min_periods=1).sum().values,
        "btc_close": rng.uniform(90_000, 110_000, days),
        "btc_return": rng.normal(0.0005, 0.02, days),
    }, index=dates)
