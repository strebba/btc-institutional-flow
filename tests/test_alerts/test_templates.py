"""Test per templates HTML — daily recap e flow event."""
from __future__ import annotations

from datetime import datetime, timezone

from src.alerts.templates import (
    EtfFlowEvent,
    FlowsSummary,
    format_daily_recap,
    format_etf_flow_alert,
)
from src.gex.models import GexSnapshot, RegimeState


def _snap(
    gex: float = 300_000_000.0,
    spot: float = 100_000.0,
    put_wall: float = 92_000.0,
    call_wall: float = 108_000.0,
    flip: float = 98_000.0,
) -> GexSnapshot:
    return GexSnapshot(
        timestamp=datetime(2026, 4, 22, 7, 0, tzinfo=timezone.utc),
        spot_price=spot,
        total_net_gex=gex,
        gamma_flip_price=flip,
        put_wall=put_wall,
        call_wall=call_wall,
        max_pain=100_000.0,
    )


def _regime(
    label: str = "positive_gamma",
    alerts: list[str] | None = None,
    percentile: float | None = 55.0,
) -> RegimeState:
    return RegimeState(
        timestamp=datetime(2026, 4, 22, 7, 0, tzinfo=timezone.utc),
        regime=label,
        total_net_gex=300e6,
        spot_price=100_000.0,
        put_wall=92_000.0,
        call_wall=108_000.0,
        gamma_flip=98_000.0,
        alerts=alerts or [],
        gex_percentile=percentile,
    )


NOW = datetime(2026, 4, 22, 7, 0, tzinfo=timezone.utc)


class TestDailyRecap:
    def test_contains_regime_and_spot(self) -> None:
        msg = format_daily_recap(_snap(), None, _regime(), None, now=NOW)
        assert "POSITIVE GAMMA" in msg
        assert "$100,000" in msg
        assert "🟢" in msg

    def test_red_emoji_for_negative_regime(self) -> None:
        msg = format_daily_recap(_snap(gex=-400e6), None, _regime("negative_gamma"), None, now=NOW)
        assert "🔴" in msg
        assert "NEGATIVE GAMMA" in msg

    def test_shows_delta_when_prev_available(self) -> None:
        cur = _snap(gex=400e6)
        prev = _snap(gex=250e6)
        msg = format_daily_recap(cur, prev, _regime(), None, now=NOW)
        # Δ = (400-250)/250 * 100 = 60%
        assert "Δ +60%" in msg

    def test_omits_delta_when_prev_missing(self) -> None:
        msg = format_daily_recap(_snap(), None, _regime(), None, now=NOW)
        assert "Δ" not in msg

    def test_flows_block_includes_top3_and_streak(self) -> None:
        flows = FlowsSummary(
            last_day_total_usd=412e6,
            last_day_by_ticker={"IBIT": 285e6, "FBTC": 78e6, "ARKB": 41e6, "BITB": 8e6},
            cumul_7d_usd=1.8e9,
            streak_days=4,
            last_day_date="2026-04-21",
        )
        msg = format_daily_recap(_snap(), None, _regime(), flows, now=NOW)
        assert "+$412M" in msg
        assert "IBIT" in msg and "FBTC" in msg and "ARKB" in msg
        assert "+$1.80B" in msg
        assert "4 gg consecutivi inflow" in msg

    def test_flows_block_handles_none(self) -> None:
        msg = format_daily_recap(_snap(), None, _regime(), None, now=NOW)
        assert "non disponibili" in msg

    def test_alerts_section_appears_when_present(self) -> None:
        msg = format_daily_recap(
            _snap(),
            None,
            _regime(alerts=["GAMMA FLIP: positivo → negativo"]),
            None,
            now=NOW,
        )
        assert "GAMMA FLIP" in msg
        assert "⚠" in msg

    def test_alerts_section_hidden_when_empty(self) -> None:
        msg = format_daily_recap(_snap(), None, _regime(alerts=[]), None, now=NOW)
        assert "⚠" not in msg

    def test_dashboard_link_present(self) -> None:
        msg = format_daily_recap(_snap(), None, _regime(), None, now=NOW)
        assert "https://seashell-app-h7hc4.ondigitalocean.app/btc" in msg


class TestFlowAlert:
    def test_single_day_inflow_message(self) -> None:
        ev = EtfFlowEvent(
            trigger="single_day",
            last_day_total_usd=620e6,
            last_day_by_ticker={"IBIT": 410e6, "FBTC": 135e6, "ARKB": 75e6},
            cumul_7d_usd=2.4e9,
            streak_days=4,
            threshold_usd=500e6,
            spot_price=102_980.0,
            gex_regime="positive_gamma",
            event_date="2026-04-22",
        )
        msg = format_etf_flow_alert(ev, now=NOW)
        assert "Large Inflow" in msg
        assert "Single-Day Move" in msg
        assert "+$620M" in msg
        assert "IBIT: +$410M" in msg
        assert "±$500M" in msg
        assert "$102,980" in msg
        assert "positive_gamma" in msg

    def test_outflow_message(self) -> None:
        ev = EtfFlowEvent(
            trigger="cumulative_7d",
            last_day_total_usd=-300e6,
            last_day_by_ticker={"IBIT": -300e6},
            cumul_7d_usd=-2.1e9,
            streak_days=-5,
            threshold_usd=2e9,
        )
        msg = format_etf_flow_alert(ev, now=NOW)
        assert "Large Outflow" in msg
        assert "7-Day Cumulative" in msg
        assert "-$2.10B" in msg
        assert "5 gg consecutivi di outflow" in msg
