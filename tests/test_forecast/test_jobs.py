"""Test per forecast/jobs.py — run_daily_predict, run_daily_verify, run_weekly_calibrate."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.forecast.jobs import run_daily_predict, run_daily_verify, run_weekly_calibrate


class TestRunDailyPredict:
    def test_kill_switch_skips(self):
        with patch("src.forecast.jobs._governance", return_value={"kill_switch": True}):
            result = run_daily_predict()
            assert result["status"] == "skipped_kill_switch"
            assert result["inserted"] == 0

    def test_data_unavailable_handled(self):
        from src.forecast.context import DataUnavailable
        with patch("src.forecast.jobs._governance", return_value={"kill_switch": False}), \
             patch("src.forecast.jobs.PredictionDB") as mock_db, \
             patch("src.forecast.jobs.load_weights_config", return_value={}), \
             patch("src.forecast.context.gather_dealer_flow_context") as mock_gather:
            mock_db.return_value.get_active_weights.return_value = (1, {})
            mock_gather.side_effect = DataUnavailable("network down")
            result = run_daily_predict()
            assert result["status"] == "data_unavailable"

    def test_success_path(self):
        mock_ctx = MagicMock()
        mock_ctx.result.signal = "LONG"
        mock_ctx.result.score = 72.0
        mock_ctx.spot = 90000.0
        mock_snap = MagicMock()
        mock_snap.gamma_flip_price = 85000.0
        mock_snap.max_pain = 88000.0
        mock_snap.total_net_gex = 500e6
        mock_ctx.snapshot = mock_snap

        with patch("src.forecast.jobs._governance", return_value={"kill_switch": False}), \
             patch("src.forecast.jobs.PredictionDB") as mock_db, \
             patch("src.forecast.jobs.load_weights_config", return_value={}), \
             patch("src.forecast.context.gather_dealer_flow_context", return_value=mock_ctx), \
             patch("src.forecast.jobs.build_dealer_flow_predictions") as mock_build:

            mock_pred = MagicMock()
            mock_build.return_value = [mock_pred]
            mock_db.return_value.get_active_weights.return_value = (1, {})
            mock_db.return_value.insert_prediction.return_value = 1

            result = run_daily_predict()
            assert result["status"] == "ok"
            assert result["inserted"] == 1
            assert result["signal"] == "LONG"


class TestRunDailyVerify:
    def test_no_due_returns_zero(self):
        with patch("src.forecast.jobs.PredictionDB") as mock_db:
            mock_db.return_value.get_due.return_value = []
            with patch("src.forecast.verifier.score_due_predictions", return_value=[]):
                result = run_daily_verify()
                assert result["status"] == "ok"
                assert result["due"] == 0
                assert result["verified"] == 0

    def test_verifies_due_predictions(self):
        mock_outcome = MagicMock()
        mock_outcome.hit = True

        with patch("src.forecast.jobs.PredictionDB") as mock_db, \
             patch("src.flows.price_fetcher.PriceFetcher"), \
             patch("src.forecast.verifier.score_due_predictions") as mock_score:
            mock_db.return_value.get_due.return_value = [MagicMock(), MagicMock()]
            mock_score.return_value = [mock_outcome, mock_outcome]

            result = run_daily_verify()
            assert result["due"] == 2
            assert result["verified"] == 2
            assert result["hit"] == 2
            assert result["miss"] == 0


class TestRunWeeklyCalibrate:
    def test_returns_calibration_report(self):
        mock_report = MagicMock()
        mock_report.gate_ok = True
        mock_report.metrics = {"total_scored": 50}

        with patch("src.forecast.jobs.PredictionDB"), \
             patch("src.forecast.jobs.run_calibration", return_value=mock_report):
            result = run_weekly_calibrate()
            assert result.gate_ok is True
            assert result.metrics["total_scored"] == 50
