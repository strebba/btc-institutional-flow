"""Il passo giornaliero ``scripts/run_flows.py`` non deve morire senza ``ibit_close``.

Riproduce il KeyError visto il 2026-11-09: ``get_all_prices`` può restituire i
close BTC senza la colonna IBIT (fetch vuoto, giorno senza borsa, cache assente).
Il passo deve loggare il buco e arrivare in fondo, senza stampare un close stimato.
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.flows.models import AggregateFlows

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_flows.py"
_spec = importlib.util.spec_from_file_location("run_flows_script", _SCRIPT)
run_flows = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = run_flows
_spec.loader.exec_module(run_flows)


def _prices(*, ibit: str) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "btc_close": np.cumsum(rng.normal(0, 1000, 40)) + 50_000,
            "btc_return": rng.normal(0.001, 0.03, 40),
            "btc_vol_7d": rng.uniform(0.3, 0.8, 40),
        },
        index=idx,
    )
    if ibit == "all-nan":
        frame["ibit_close"] = np.nan
    return frame


class _FakeScraper:
    def fetch(self):
        return [object()]

    def aggregate(self, _flows):
        base = date(2024, 1, 1)
        return [
            AggregateFlows(
                date=base + timedelta(days=i),
                total_flow_usd=1e8,
                ibit_flow_usd=5e7,
                flows_by_ticker={"IBIT": 5e7},
            )
            for i in range(40)
        ]

    def to_dataframe(self, _flows):
        idx = pd.date_range("2024-01-01", periods=40, freq="D")
        return pd.DataFrame({"IBIT": [5e7] * 40, "total": [1e8] * 40}, index=idx)


class _FakeFetcher:
    def __init__(self, prices: pd.DataFrame) -> None:
        self._prices = prices

    def get_all_prices(self, **_kwargs):
        return self._prices

    def get_ibit_btc_ratio(self):
        return None


@pytest.mark.parametrize("ibit", ["missing", "all-nan"])
def test_run_flows_completes_when_ibit_close_is_missing(monkeypatch, capsys, ibit):
    warnings: list[str] = []

    def _warning(msg, *args, **_kwargs):
        warnings.append(msg % args if args else str(msg))

    monkeypatch.setattr(run_flows._log, "warning", _warning)
    monkeypatch.setattr(run_flows, "FarsideScraper", _FakeScraper)
    monkeypatch.setattr(run_flows, "PriceFetcher", lambda: _FakeFetcher(_prices(ibit=ibit)))
    monkeypatch.setattr(sys, "argv", ["run_flows.py", "--no-fetch"])

    run_flows.main()

    out = capsys.readouterr().out
    assert "Completato" in out
    assert "IBIT ultimo close non disponibile" in out
    assert "IBIT ultimo close:" not in out
    assert warnings
    assert any("ibit_close" in message for message in warnings)
    assert any("Nessun prezzo inventato" in message for message in warnings)
    if ibit == "missing":
        assert any("assente" in message for message in warnings)
    else:
        assert any("senza close osservati" in message for message in warnings)
