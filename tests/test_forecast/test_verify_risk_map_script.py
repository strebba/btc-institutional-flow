"""Lo script di verifica gira su un CSV, senza rete e senza scrivere sul DB."""
from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd

from scripts.verify_risk_map import main


def _csv(tmp_path, n: int = 140) -> tuple[str, date]:
    rng = np.random.default_rng(9)
    end = date(2024, 6, 1)
    close = 60_000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, n)))
    span = np.abs(rng.normal(0.01, 0.002, n))
    dates = [end - timedelta(days=(n - 1 - i)) for i in range(n)]
    pd.DataFrame({
        "date": dates,
        "open": close,
        "high": close * np.exp(span),
        "low": close * np.exp(-span),
        "close": close,
    }).to_csv(tmp_path / "btc.csv", index=False)
    return str(tmp_path / "btc.csv"), end + timedelta(days=1)


def test_sample_date_prints_a_risk_map(tmp_path, capsys):
    path, day = _csv(tmp_path)
    code = main(["--csv", path, "--sample-date", day.isoformat(), "--ref-price", "60000"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["last_bar_date"] < day.isoformat()
    assert payload["expected_range_5d_pct"] > 0
    assert payload["prob_range_gt_median_90d"] is not None
    assert payload["levels"] == []
    assert "score" not in payload


def test_walk_forward_report_has_brier_and_auc(tmp_path, capsys):
    path, _day = _csv(tmp_path, n=160)
    code = main(["--csv", path, "--min-returns", "50"])
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    metrics = report["metrics"]
    assert metrics["range_model"]["n"] > 20
    assert 0.0 <= metrics["range_model"]["brier"] <= 1.0
    assert metrics["range_model"]["brier_constant_expanding"] is not None
    touch = metrics["touch_used"]
    assert touch["n"] > 50
    assert touch["auc"] is not None and touch["auc"] > 0.7
    assert "calibration" in touch
