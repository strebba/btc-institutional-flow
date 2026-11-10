"""Risk map: niente lookahead, niente score, SQLite e Postgres (stesso PredictionDB)."""
from __future__ import annotations

import inspect
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.forecast.models import Prediction, TARGET_DIRECTION
from src.forecast.prediction_db import PredictionDB
from src.forecast.risk_map import (
    build_risk_map,
    expected_log_range,
    range_cdf,
    reflection_touch_prob,
)


def _bars(n: int = 180, sigma: float = 0.02, seed: int = 1, end: date | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = end or date(2024, 12, 31)
    rets = rng.normal(0.0, sigma, n)
    close = 100_000.0 * np.exp(np.cumsum(rets))
    span = np.abs(rng.normal(sigma * 0.6, sigma * 0.15, n))
    high = close * np.exp(span)
    low = close * np.exp(-span)
    dates = [end - timedelta(days=(n - 1 - i)) for i in range(n)]
    return pd.DataFrame({
        "date": dates,
        "open": close,
        "high": np.maximum(high, close),
        "low": np.minimum(low, close),
        "close": close,
    })


def _asof_after(df: pd.DataFrame) -> datetime:
    last = df["date"].iloc[-1]
    return datetime(last.year, last.month, last.day, 7, 30, tzinfo=timezone.utc) + timedelta(days=1)


class TestBinanceFetch:
    def test_uses_the_public_mirror_when_the_main_host_is_blocked(self, monkeypatch):
        import requests

        candle = [1_704_067_200_000, "100", "110", "90", "105", "1", 1_704_153_599_999]

        class Resp:
            def __init__(self, status, body):
                self.status_code = status
                self._body = body
                self.text = ""

            def json(self):
                return self._body

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise requests.HTTPError(f"{self.status_code}", response=self)

        def fake_get(url, params=None, timeout=None):
            if "api.binance.com" in url:
                return Resp(451, {})
            return Resp(200, [candle])

        monkeypatch.setattr(requests, "get", fake_get)
        from src.forecast.risk_map import fetch_binance_daily
        df = fetch_binance_daily(start="2024-01-01", timeout=1)
        assert list(df["close"]) == [105.0]
        assert df["date"].iloc[0].isoformat() == "2024-01-01"


class TestClosedForm:
    def test_reflection_matches_two_tailed_normal(self):
        assert reflection_touch_prob(0.0) == 1.0
        # 2 (1 − Φ(1)) ≈ 0.3173
        assert reflection_touch_prob(1.0) == pytest.approx(0.31731050786291415, abs=1e-9)
        assert reflection_touch_prob(8.0) == 0.0
        assert 0.0 < reflection_touch_prob(0.5) < 1.0

    def test_range_cdf_is_a_proper_distribution(self):
        assert range_cdf(0.0) == 0.0
        assert range_cdf(10.0) == 1.0
        assert range_cdf(2.0) == pytest.approx(0.8189, abs=1e-3)
        zs = np.linspace(0.5, 4.5, 40)
        vals = [range_cdf(float(z)) for z in zs]
        assert vals == sorted(vals)
        # E[R]/σ = √(8/π) ≈ 1.5958, integrando la sopravvivenza sulla griglia.
        grid = np.linspace(0.0, 6.0, 601)
        surv = np.array([1.0 - range_cdf(float(z)) for z in grid])
        assert float(np.trapezoid(surv, grid)) == pytest.approx(math.sqrt(8.0 / math.pi), rel=0.02)

    def test_expected_range_scales_with_sigma(self):
        assert expected_log_range(0.04) == pytest.approx(0.04 * math.sqrt(8.0 / math.pi))


class TestBuild:
    def test_signature_has_no_score_or_weights(self):
        names = set(inspect.signature(build_risk_map).parameters)
        assert "score" not in names
        assert "weights" not in names

    def test_higher_vol_widens_the_5d_sigma(self):
        calm = build_risk_map(_bars(160, sigma=0.008, seed=2), ref_price=100_000.0,
                              asof=_asof_after(_bars(160, sigma=0.008, seed=2)))
        wild_df = _bars(160, sigma=0.04, seed=2)
        wild = build_risk_map(wild_df, ref_price=100_000.0, asof=_asof_after(wild_df))
        assert wild["vol_method"] in {"garch", "ewma"}
        assert wild["sigma_5d_log"] > calm["sigma_5d_log"] * 2
        assert wild["expected_range_5d_usd"] > calm["expected_range_5d_usd"]
        assert "score" not in wild

    def test_ewma_when_the_fit_fails(self):
        df = _bars(120, seed=3)
        with patch("src.forecast.risk_map.minimize", side_effect=RuntimeError("fit")):
            rm = build_risk_map(df, ref_price=100_000.0, asof=_asof_after(df))
        assert rm["vol_method"] == "ewma"
        assert rm["sigma_5d_log"] > 0
        assert rm["prob_range_gt_median_90d"] is not None

    def test_touch_prob_falls_as_the_level_moves_away(self):
        df = _bars(200, seed=4)
        ref = float(df["close"].iloc[-1])
        asof = _asof_after(df)
        rm = build_risk_map(
            df, ref_price=ref, asof=asof,
            levels=[
                {"name": "max_pain", "price": ref * 1.005},
                {"name": "call_wall", "price": ref * 1.08},
            ],
        )
        by_name = {row["name"]: row for row in rm["levels"]}
        near, far = by_name["max_pain"], by_name["call_wall"]
        assert near["distance_vol"] < far["distance_vol"]
        assert near["touch_prob_5d"] > far["touch_prob_5d"]
        assert near["touch_method"] in {"reflection", "empirical_excursion"}
        assert 0.0 <= far["touch_prob_reflection"] <= near["touch_prob_reflection"] <= 1.0
        json.dumps(rm)

    def test_partial_bar_and_future_bars_do_not_change_the_map(self):
        df = _bars(220, seed=5)
        cut = df["date"].iloc[160]
        asof = datetime(cut.year, cut.month, cut.day, 7, 30, tzinfo=timezone.utc)
        ref = float(df.loc[df["date"] < cut, "close"].iloc[-1])
        level = [{"name": "put_wall", "price": ref * 0.97}]
        base = build_risk_map(df, ref_price=ref, levels=level, asof=asof)
        shocked = df.copy()
        shocked.loc[shocked["date"] >= cut, "high"] = shocked.loc[shocked["date"] >= cut, "close"] * 3
        shocked.loc[shocked["date"] >= cut, "low"] = shocked.loc[shocked["date"] >= cut, "close"] * 0.4
        shocked.loc[shocked["date"] >= cut, "close"] = shocked.loc[shocked["date"] >= cut, "close"] * 1.5
        shocked["partial"] = [1 if d >= cut else 0 for d in shocked["date"]]
        other = build_risk_map(shocked, ref_price=ref, levels=level, asof=asof)
        for key in ("sigma_5d_log", "prob_range_gt_median_90d", "expected_range_5d_pct", "last_bar_date"):
            assert other[key] == base[key]
        assert other["levels"][0]["touch_prob_5d"] == base["levels"][0]["touch_prob_5d"]
        assert base["last_bar_date"] < cut.isoformat()

    def test_a_wild_window_enters_the_median_only_after_it_has_closed(self):
        df = _bars(180, sigma=0.01, seed=6)
        # Più di metà della finestra a 90 giorni: altrimenti la mediana non si sposta.
        wild_start = df["date"].iloc[-70]
        df = df.copy()
        df.loc[df["date"] >= wild_start, "high"] = df.loc[df["date"] >= wild_start, "close"] * 1.6
        df.loc[df["date"] >= wild_start, "low"] = df.loc[df["date"] >= wild_start, "close"] * 0.55
        before = build_risk_map(
            df, ref_price=100_000.0,
            asof=datetime(wild_start.year, wild_start.month, wild_start.day, 7, 30, tzinfo=timezone.utc),
        )
        after = build_risk_map(df, ref_price=100_000.0, asof=_asof_after(df))
        assert before["last_bar_date"] < wild_start.isoformat()
        assert after["range_median_90d_log"] > before["range_median_90d_log"] * 1.5


class TestPersistence:
    def test_new_metadata_does_not_rewrite_old_rows_or_outcomes(self, tmp_path: Path):
        db = PredictionDB(db_path=tmp_path / "runtime.db")
        old = Prediction(
            source="dealer_flow", asset="BTC", target_type=TARGET_DIRECTION,
            target_spec={"direction": "up", "ref_price": 100.0, "flat_band_pct": 1.0},
            horizon_days=5, confidence=0.2, created_at="2026-01-02T07:30:00",
            score_ref=61.0, metadata=None,
        )
        old_id = db.insert_prediction(old)
        from src.forecast.models import Outcome
        db.insert_outcome(Outcome(
            prediction_id=old_id, hit=True, realized_return=0.01,
            realized_price=101.0, ref_price=100.0, brier=0.2, detail="invariato",
        ))
        fresh = Prediction(
            source="dealer_flow", asset="BTC", target_type=TARGET_DIRECTION,
            target_spec={"direction": "up", "ref_price": 100.0, "flat_band_pct": 1.0},
            horizon_days=5, confidence=0.2, created_at="2026-02-02T07:30:00",
            score_ref=61.0,
            metadata={"risk_map": {"sigma_5d_log": 0.04}, "levels_pit": {"put_wall": 90.0}},
        )
        db.insert_prediction(fresh)

        stored_old = db.predictions_on_date("2026-01-02", "dealer_flow")[0]
        assert stored_old.metadata is None
        assert stored_old.score_ref == 61.0
        scored = db.scored_rows()
        assert len(scored) == 1
        assert scored[0]["outcome_detail"] == "invariato"
        assert scored[0]["brier"] == pytest.approx(0.2)
        stored_new = db.predictions_on_date("2026-02-02", "dealer_flow")[0]
        assert stored_new.metadata["risk_map"]["sigma_5d_log"] == 0.04
        assert stored_new.metadata["levels_pit"]["put_wall"] == 90.0
