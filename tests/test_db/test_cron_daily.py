"""Test di scripts/cron_daily.py (nessuna chiamata di rete: i passi sono simulati)."""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "cron_daily.py"
_spec = importlib.util.spec_from_file_location("cron_daily", _SCRIPT)
cron = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cron
_spec.loader.exec_module(cron)

MONDAY = datetime(2026, 10, 26, 7, 15, tzinfo=timezone.utc)
THURSDAY = datetime(2026, 10, 29, 7, 15, tzinfo=timezone.utc)


def test_calibrate_only_on_monday():
    assert "calibrate" in cron.select_steps(None, False, MONDAY)
    assert "calibrate" not in cron.select_steps(None, False, THURSDAY)
    assert "calibrate" in cron.select_steps(None, True, THURSDAY)


def test_explicit_steps_keep_canonical_order():
    assert cron.select_steps("predict,gex", False, THURSDAY) == ["gex", "predict"]


def test_unknown_step_rejected():
    with pytest.raises(SystemExit):
        cron.select_steps("gex,boh", False, THURSDAY)


def test_refuses_to_run_without_database_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "")
    with pytest.raises(SystemExit) as exc:
        cron.main([])
    assert "DATABASE_URL" in str(exc.value)


def test_step_failure_does_not_stop_others(monkeypatch):
    calls = []

    def ok(name):
        def _f():
            calls.append(name)
            return "ok"
        return _f

    def boom():
        calls.append("gex")
        raise RuntimeError("deribit giu'")

    monkeypatch.setitem(cron.STEP_FUNCS, "gex", boom)
    monkeypatch.setitem(cron.STEP_FUNCS, "predict", ok("predict"))
    res = cron.run(["gex", "predict"])
    assert calls == ["gex", "predict"]
    assert res["gex"][0] is False and res["predict"][0] is True


def test_exit_code_ignores_probe_failures(monkeypatch, pg_url):
    monkeypatch.setenv("DATABASE_URL", pg_url)

    def fail():
        raise RuntimeError("x")

    monkeypatch.setitem(cron.STEP_FUNCS, "coinglass", fail)
    monkeypatch.setitem(cron.STEP_FUNCS, "verify", lambda: "ok")
    assert cron.main(["--steps", "coinglass,verify"]) == 0
    monkeypatch.setitem(cron.STEP_FUNCS, "verify", fail)
    assert cron.main(["--steps", "coinglass,verify"]) == 1
