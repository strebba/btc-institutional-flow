"""Test di integrazione per forecast/jobs.py — orchestrazione con componenti reali.

Complementa tests/test_forecast/test_jobs.py (che mocka tutto, kill-switch/data-unavailable
inclusi): qui `PredictionDB` e `build_dealer_flow_predictions` girano per davvero, su un
DB temporaneo. L'unico confine mockato è `gather_dealer_flow_context` (richiede rete:
Deribit/Farside/CoinGlass). Verifica che l'orchestrazione reale in `run_daily_predict`
componga correttamente gli oggetti tra i moduli — un mismatch di campi qui sfuggirebbe a
una suite interamente mockata.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.analytics.factor_scorers import SignalInputs, SignalResult
from src.forecast.context import DealerFlowContext
from src.forecast.jobs import run_daily_predict
from src.forecast.prediction_db import PredictionDB
from src.gex.models import GexSnapshot


def _bars() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 160
    end = date(2024, 6, 30)
    close = 90_000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, n)))
    span = np.abs(rng.normal(0.012, 0.003, n))
    dates = [end - timedelta(days=(n - 1 - i)) for i in range(n)]
    return pd.DataFrame({
        "date": dates,
        "open": close,
        "high": close * np.exp(span),
        "low": close * np.exp(-span),
        "close": close,
    })

pytestmark = pytest.mark.integration


def _real_context() -> DealerFlowContext:
    snapshot = GexSnapshot(
        timestamp=datetime.now(timezone.utc),
        spot_price=90_000.0,
        total_net_gex=500e6,
        gamma_flip_price=85_000.0,
        put_wall=80_000.0,
        call_wall=95_000.0,
        max_pain=88_000.0,
    )
    inputs = SignalInputs(gex_usd=500e6, etf_flow_3d_usd=120e6, spot_price=90_000.0)
    result = SignalResult(
        score=72.0, signal="LONG",
        components={"gex": 1.0, "etf_flows": 0.8},
        weights_used={"gex": 0.4, "etf_flows": 0.3},
        reason="test",
    )
    return DealerFlowContext(
        result=result, snapshot=snapshot, spot=90_000.0, inputs=inputs,
        ibit_flow_3d=120e6, near_barrier=False,
    )


class TestRunDailyPredictOrchestration:
    def test_persiste_predizioni_reali_nel_db(self, tmp_path: Path):
        db = PredictionDB(db_path=tmp_path / "predictions.db")

        with patch("src.forecast.jobs._governance", return_value={"kill_switch": False}), \
             patch("src.forecast.context.gather_dealer_flow_context", return_value=_real_context()), \
             patch("src.forecast.jobs.load_risk_map_context", return_value=(_bars(), [])):
            result = run_daily_predict(db=db)

        assert result["status"] == "ok"
        assert result["signal"] == "LONG"
        assert result["inserted"] == result["total"] > 0

        rows = db.get_recent(limit=20)
        assert len(rows) == result["inserted"]
        assert all(r.asset == "BTC" for r in rows)
        assert all(r.created_at for r in rows)
        # v1 e l'ombra v2 portano la stessa mappa; lo score non viene tolto.
        maps = [r.metadata["risk_map"]["sigma_5d_log"] for r in rows]
        assert len(set(maps)) == 1 and maps[0] > 0
        assert all(r.score_ref == 72.0 for r in rows)
        names = {lvl["name"] for lvl in rows[0].metadata["risk_map"]["levels"]}
        assert {"put_wall", "call_wall", "gamma_flip", "max_pain"} <= names
        pit = rows[0].metadata["levels_pit"]
        assert pit["put_wall"] == 80_000.0
        assert pit["total_gex"] == 500e6
        assert pit["gex_sign"] == 1
        assert "score" not in rows[0].metadata["risk_map"]

    def test_inserimento_duplicato_non_raddoppia(self, tmp_path: Path):
        """Stesso contesto lanciato due volte nello stesso giorno: insert_prediction
        ignora i duplicati (constraint reale sul DB), l'orchestrazione non deve fallire."""
        db = PredictionDB(db_path=tmp_path / "predictions.db")

        with patch("src.forecast.jobs._governance", return_value={"kill_switch": False}), \
             patch("src.forecast.context.gather_dealer_flow_context", return_value=_real_context()), \
             patch("src.forecast.jobs.load_risk_map_context", return_value=(_bars(), [])):
            first = run_daily_predict(db=db)
            second = run_daily_predict(db=db)

        assert first["inserted"] > 0
        assert second["inserted"] == 0
        assert len(db.get_recent(limit=50)) == first["inserted"]
