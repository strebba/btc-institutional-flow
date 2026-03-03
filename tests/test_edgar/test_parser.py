"""Test unitari per il parser dei prospetti SEC."""
from __future__ import annotations

import pytest
from src.edgar.parser import (
    _parse_date,
    _parse_notional,
    _detect_product_type,
    _detect_issuer,
    _extract_barrier_levels,
)


class TestParseDate:
    def test_month_name(self):
        d = _parse_date("January 15, 2024")
        assert d.year == 2024
        assert d.month == 1
        assert d.day == 15

    def test_iso_format(self):
        d = _parse_date("2024-03-20")
        assert d.year == 2024
        assert d.month == 3

    def test_invalid(self):
        assert _parse_date("not a date") is None

    def test_december(self):
        d = _parse_date("December 31, 2025")
        assert d.month == 12
        assert d.day == 31


class TestParseNotional:
    def test_million(self):
        text = "The aggregate principal amount of the notes is $5,000,000."
        n = _parse_notional(text)
        assert n == 5_000_000.0

    def test_million_suffix(self):
        text = "Total offering: $15 million"
        n = _parse_notional(text)
        assert n == 15_000_000.0

    def test_billion(self):
        text = "notional amount of $1.5 billion"
        n = _parse_notional(text)
        assert n == 1_500_000_000.0

    def test_no_match(self):
        assert _parse_notional("no amount here") is None


class TestDetectProductType:
    def test_autocallable(self):
        text = "These are Auto-Callable notes linked to IBIT."
        assert _detect_product_type(text) == "autocallable"

    def test_barrier(self):
        text = "The notes feature a knock-in level at 70%."
        assert _detect_product_type(text) == "barrier_note"

    def test_buffered(self):
        text = "A buffered note with 15% principal buffer protection."
        assert _detect_product_type(text) == "buffered_note"

    def test_none(self):
        text = "Some unrelated text about bonds."
        assert _detect_product_type(text) is None


class TestDetectIssuer:
    ISSUERS = ["JPMorgan", "Goldman Sachs", "Barclays", "Morgan Stanley"]

    def test_jpmorgan(self):
        text = "Issued by JPMorgan Chase Bank, N.A."
        assert _detect_issuer(text, self.ISSUERS) == "JPMorgan"

    def test_case_insensitive(self):
        text = "goldman sachs & co. llc"
        assert _detect_issuer(text, self.ISSUERS) == "Goldman Sachs"

    def test_not_found(self):
        assert _detect_issuer("Unknown Bank Ltd.", self.ISSUERS) is None


class TestExtractBarrierLevels:
    def test_knock_in(self):
        text = (
            "The knock-in level is set at 70% of the Initial Value. "
            "If IBIT falls below this knock-in level, principal is at risk."
        )
        barriers = _extract_barrier_levels(text, initial_level=50.0)
        assert len(barriers) == 1
        assert barriers[0].barrier_type == "knock_in"
        assert barriers[0].level_pct == 70.0
        assert abs(barriers[0].level_price_ibit - 35.0) < 0.01

    def test_autocall(self):
        text = (
            "The auto-call trigger level is 100% of the Starting Value. "
            "On each observation date, if the closing price equals or exceeds "
            "the auto-call trigger level..."
        )
        barriers = _extract_barrier_levels(text, initial_level=50.0)
        autocalls = [b for b in barriers if b.barrier_type == "autocall"]
        assert len(autocalls) >= 1
        assert autocalls[0].level_pct == 100.0

    def test_no_barriers(self):
        barriers = _extract_barrier_levels("No barrier levels mentioned.", None)
        assert barriers == []

    def test_multiple_barriers(self):
        text = (
            "The knock-in level is 65% of the Initial Value. "
            "The auto-call trigger is 105% of the Initial Value."
        )
        barriers = _extract_barrier_levels(text, initial_level=40.0)
        types = {b.barrier_type for b in barriers}
        assert "knock_in" in types
        assert "autocall" in types

    def test_no_initial_level(self):
        text = "The barrier is at 75% of the Initial Value."
        barriers = _extract_barrier_levels(text, initial_level=None)
        assert len(barriers) == 1
        assert barriers[0].level_price_ibit is None
