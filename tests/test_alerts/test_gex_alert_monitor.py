"""Test per GexAlertMonitor — soglie conservative, dedup, cooldown."""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from src.alerts.alert_db import AlertDB
from src.alerts.gex_alert_monitor import (
    GexAlertMonitor,
    evaluate_etf_flow_triggers,
    summarize_flows,
)
from src.alerts.telegram_client import TelegramClient
from src.alerts.templates import FlowsSummary
from src.flows.models import AggregateFlows
from src.gex.gex_db import GexDB
from src.gex.models import GexSnapshot


# ─── Fixtures ────────────────────────────────────────────────────────────────


def _snap(
    d: date,
    gex: float = 300e6,
    spot: float = 100_000.0,
) -> GexSnapshot:
    return GexSnapshot(
        timestamp=datetime(d.year, d.month, d.day, 10, 0, tzinfo=timezone.utc),
        spot_price=spot,
        total_net_gex=gex,
        gamma_flip_price=98_000.0,
        put_wall=92_000.0,
        call_wall=108_000.0,
        max_pain=100_000.0,
    )


def _agg(d: date, total: float, by_ticker: dict[str, float] | None = None) -> AggregateFlows:
    return AggregateFlows(
        date=d,
        total_flow_usd=total,
        ibit_flow_usd=(by_ticker or {}).get("IBIT", total),
        flows_by_ticker=by_ticker or {"IBIT": total},
    )


class FakeTelegram:
    """Stub async di TelegramClient che raccoglie i messaggi."""

    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.sent: list[str] = []

    async def send_message(self, text: str, *, disable_web_page_preview: bool = True) -> bool:
        self.sent.append(text)
        return self.ok


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture
def gex_db(db_path: Path) -> GexDB:
    return GexDB(db_path=db_path)


@pytest.fixture
def alert_db(db_path: Path) -> AlertDB:
    return AlertDB(db_path=db_path)


def run(coro):
    return asyncio.run(coro)


# ─── summarize_flows ─────────────────────────────────────────────────────────


class TestSummarizeFlows:
    def test_empty_returns_none(self) -> None:
        assert summarize_flows([]) is None

    def test_single_day_summary(self) -> None:
        s = summarize_flows([_agg(date(2026, 4, 22), 600e6, {"IBIT": 400e6, "FBTC": 200e6})])
        assert s is not None
        assert s.last_day_total_usd == 600e6
        assert s.cumul_7d_usd == 600e6
        assert s.streak_days == 1
        assert s.last_day_by_ticker == {"IBIT": 400e6, "FBTC": 200e6}

    def test_cumul_7d_sums_correct_window(self) -> None:
        aggs = [_agg(date(2026, 4, 22 - i), 100e6) for i in range(10)]
        s = summarize_flows(aggs)
        # Ultimi 7 giorni → 700M
        assert s is not None
        assert s.cumul_7d_usd == pytest.approx(700e6)

    def test_streak_counts_consecutive_inflows(self) -> None:
        aggs = [
            _agg(date(2026, 4, 15), -50e6),
            _agg(date(2026, 4, 16), 100e6),
            _agg(date(2026, 4, 17), 200e6),
            _agg(date(2026, 4, 18), 150e6),
        ]
        s = summarize_flows(aggs)
        assert s is not None
        assert s.streak_days == 3

    def test_streak_counts_consecutive_outflows_as_negative(self) -> None:
        aggs = [
            _agg(date(2026, 4, 15), 50e6),
            _agg(date(2026, 4, 16), -100e6),
            _agg(date(2026, 4, 17), -200e6),
        ]
        s = summarize_flows(aggs)
        assert s is not None
        assert s.streak_days == -2

    def test_streak_breaks_on_zero(self) -> None:
        aggs = [
            _agg(date(2026, 4, 15), 100e6),
            _agg(date(2026, 4, 16), 100e6),
            _agg(date(2026, 4, 17), 0.0),
            _agg(date(2026, 4, 18), 100e6),
        ]
        s = summarize_flows(aggs)
        # Zero non rompe in direzione positiva — streak conta solo dalla coda
        assert s is not None
        assert s.streak_days == 1


# ─── evaluate_etf_flow_triggers ──────────────────────────────────────────────


class TestEvaluateTriggers:
    def _summary(
        self,
        last_day: float,
        cumul_7d: float,
        streak: int,
    ) -> FlowsSummary:
        return FlowsSummary(
            last_day_total_usd=last_day,
            last_day_by_ticker={"IBIT": last_day},
            cumul_7d_usd=cumul_7d,
            streak_days=streak,
            last_day_date="2026-04-22",
        )

    def test_conservative_profile_no_trigger_on_small_flow(self) -> None:
        events = evaluate_etf_flow_triggers(
            self._summary(last_day=300e6, cumul_7d=1.5e9, streak=2),
            single_day_threshold=500e6,
            cumulative_7d_threshold=2e9,
            streak_min_days=3,
        )
        assert events == []

    def test_single_day_inflow_above_threshold_triggers(self) -> None:
        events = evaluate_etf_flow_triggers(
            self._summary(last_day=600e6, cumul_7d=1.2e9, streak=2),
            single_day_threshold=500e6,
            cumulative_7d_threshold=2e9,
            streak_min_days=3,
        )
        assert len(events) == 1
        assert events[0].trigger == "single_day"

    def test_single_day_outflow_above_threshold_triggers(self) -> None:
        events = evaluate_etf_flow_triggers(
            self._summary(last_day=-700e6, cumul_7d=-1e9, streak=-2),
            single_day_threshold=500e6,
            cumulative_7d_threshold=2e9,
            streak_min_days=3,
        )
        assert len(events) == 1
        assert events[0].trigger == "single_day"

    def test_cumulative_triggers_independently(self) -> None:
        events = evaluate_etf_flow_triggers(
            self._summary(last_day=100e6, cumul_7d=2.3e9, streak=2),
            single_day_threshold=500e6,
            cumulative_7d_threshold=2e9,
            streak_min_days=3,
        )
        triggers = {e.trigger for e in events}
        assert "cumulative_7d" in triggers
        assert "single_day" not in triggers

    def test_streak_triggers_at_min_days(self) -> None:
        events = evaluate_etf_flow_triggers(
            self._summary(last_day=100e6, cumul_7d=500e6, streak=3),
            single_day_threshold=500e6,
            cumulative_7d_threshold=2e9,
            streak_min_days=3,
        )
        triggers = {e.trigger for e in events}
        assert "streak" in triggers

    def test_all_three_can_fire_together(self) -> None:
        events = evaluate_etf_flow_triggers(
            self._summary(last_day=800e6, cumul_7d=3e9, streak=5),
            single_day_threshold=500e6,
            cumulative_7d_threshold=2e9,
            streak_min_days=3,
        )
        assert len(events) == 3


# ─── Daily recap integration ─────────────────────────────────────────────────


class TestDailyRecapIntegration:
    def test_skips_when_telegram_missing(self, gex_db: GexDB, alert_db: AlertDB) -> None:
        gex_db.insert_snapshot(_snap(date(2026, 4, 22)), regime="positive_gamma")
        mon = GexAlertMonitor(
            telegram=None,
            gex_db=gex_db,
            alert_db=alert_db,
            fetch_flows=lambda: [],
            config={"cooldown_hours": 24},
        )
        assert run(mon.send_daily_recap()) is False

    def test_skips_when_no_gex_snapshots(self, gex_db: GexDB, alert_db: AlertDB) -> None:
        tg = FakeTelegram()
        mon = GexAlertMonitor(
            telegram=tg, gex_db=gex_db, alert_db=alert_db,
            fetch_flows=lambda: [], config={"cooldown_hours": 24},
        )
        assert run(mon.send_daily_recap()) is False
        assert tg.sent == []

    def test_sends_when_snapshot_exists(self, gex_db: GexDB, alert_db: AlertDB) -> None:
        gex_db.insert_snapshot(_snap(date(2026, 4, 22)), regime="positive_gamma")
        tg = FakeTelegram()
        mon = GexAlertMonitor(
            telegram=tg, gex_db=gex_db, alert_db=alert_db,
            fetch_flows=lambda: [_agg(date(2026, 4, 21), 120e6)],
            config={"cooldown_hours": 24},
        )
        assert run(mon.send_daily_recap()) is True
        assert len(tg.sent) == 1
        assert "BTC Institutional Flow" in tg.sent[0]
        assert "$100,000" in tg.sent[0]

    def test_respects_cooldown(self, gex_db: GexDB, alert_db: AlertDB) -> None:
        gex_db.insert_snapshot(_snap(date(2026, 4, 22)), regime="positive_gamma")
        tg = FakeTelegram()
        mon = GexAlertMonitor(
            telegram=tg, gex_db=gex_db, alert_db=alert_db,
            fetch_flows=lambda: [], config={"cooldown_hours": 24},
        )
        assert run(mon.send_daily_recap()) is True
        assert run(mon.send_daily_recap()) is False  # cooldown attivo
        assert len(tg.sent) == 1

    def test_flow_fetch_failure_does_not_block_recap(
        self, gex_db: GexDB, alert_db: AlertDB
    ) -> None:
        gex_db.insert_snapshot(_snap(date(2026, 4, 22)), regime="neutral")
        tg = FakeTelegram()

        def broken_fetch() -> list:
            raise RuntimeError("Farside 500")

        mon = GexAlertMonitor(
            telegram=tg, gex_db=gex_db, alert_db=alert_db,
            fetch_flows=broken_fetch, config={"cooldown_hours": 24},
        )
        assert run(mon.send_daily_recap()) is True
        assert "non disponibili" in tg.sent[0]


# ─── ETF flow event integration ──────────────────────────────────────────────


class TestFlowCheckIntegration:
    def _mon(self, gex_db, alert_db, flows: list[AggregateFlows]) -> tuple[GexAlertMonitor, FakeTelegram]:
        tg = FakeTelegram()
        mon = GexAlertMonitor(
            telegram=tg,
            gex_db=gex_db,
            alert_db=alert_db,
            fetch_flows=lambda: flows,
            config={
                "cooldown_hours": 24,
                "etf_flow_check": {
                    "single_day_usd_threshold": 500_000_000,
                    "cumulative_7d_usd_threshold": 2_000_000_000,
                    "streak_min_days": 3,
                },
            },
        )
        return mon, tg

    def test_no_alert_for_small_flow(self, gex_db: GexDB, alert_db: AlertDB) -> None:
        flows = [_agg(date(2026, 4, 22), 300e6)]
        mon, tg = self._mon(gex_db, alert_db, flows)
        assert run(mon.check_etf_flows()) == 0
        assert tg.sent == []

    def test_alert_for_large_single_day(self, gex_db: GexDB, alert_db: AlertDB) -> None:
        flows = [_agg(date(2026, 4, 22), 620e6, {"IBIT": 410e6, "FBTC": 210e6})]
        mon, tg = self._mon(gex_db, alert_db, flows)
        assert run(mon.check_etf_flows()) == 1
        assert "Large Inflow" in tg.sent[0]

    def test_cooldown_blocks_second_run(self, gex_db: GexDB, alert_db: AlertDB) -> None:
        flows = [_agg(date(2026, 4, 22), 620e6)]
        mon, tg = self._mon(gex_db, alert_db, flows)
        run(mon.check_etf_flows())
        n = run(mon.check_etf_flows())
        assert n == 0  # cooldown attivo
        assert len(tg.sent) == 1

    def test_streak_alert_fires_at_threshold(self, gex_db: GexDB, alert_db: AlertDB) -> None:
        flows = [_agg(date(2026, 4, 20 + i), 100e6) for i in range(3)]
        mon, tg = self._mon(gex_db, alert_db, flows)
        assert run(mon.check_etf_flows()) >= 1
        assert any("Streak" in m for m in tg.sent)

    def test_enriches_with_gex_context(self, gex_db: GexDB, alert_db: AlertDB) -> None:
        gex_db.insert_snapshot(_snap(date(2026, 4, 22), spot=103_500.0), regime="positive_gamma")
        flows = [_agg(date(2026, 4, 22), 600e6)]
        mon, tg = self._mon(gex_db, alert_db, flows)
        run(mon.check_etf_flows())
        assert len(tg.sent) == 1
        assert "$103,500" in tg.sent[0]
        assert "positive_gamma" in tg.sent[0]
