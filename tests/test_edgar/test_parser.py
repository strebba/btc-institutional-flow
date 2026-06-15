"""Test unitari per il parser dei prospetti SEC."""
from __future__ import annotations

from datetime import date

from src.edgar.models import StructuredNote
from src.edgar.parser import (
    ProspectusParser,
    _parse_date,
    _parse_notional,
    _detect_product_type,
    _detect_issuer,
    _canonicalize_issuer,
    _known_issuer_or_none,
    _detect_underlying,
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

    def test_aggregate_wins(self):
        text = "Aggregate principal amount: $5,000,000  Payment at maturity: ..."
        assert _parse_notional(text) == 5_000_000.0

    def test_rejects_per_note_denomination(self):
        # "$1,000 stated principal amount" è la denominazione, non il notional.
        text = "for each $1,000 stated principal amount of the notes"
        assert _parse_notional(text) is None


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


class TestCanonicalizeIssuer:
    def test_jpmorgan_entity_variant(self):
        # La variante grezza di entity_name EDGAR va unificata a "JPMorgan"
        assert _canonicalize_issuer("JPMorgan Chase Financial Co. LLC") == "JPMorgan"

    def test_hsbc_state_suffix(self):
        assert _canonicalize_issuer("HSBC USA INC /MD/") == "HSBC"

    def test_already_canonical(self):
        assert _canonicalize_issuer("Morgan Stanley") == "Morgan Stanley"

    def test_case_insensitive(self):
        assert _canonicalize_issuer("goldman sachs & co. llc") == "Goldman Sachs"

    def test_unknown_passthrough(self):
        # Emittente non mappato: ritorna il nome ripulito, non None
        assert _canonicalize_issuer("Some Tiny Bank Ltd.") == "Some Tiny Bank Ltd."

    def test_none_input(self):
        assert _canonicalize_issuer(None) is None

    def test_empty_input(self):
        assert _canonicalize_issuer("") is None


class TestKnownIssuerOrNone:
    def test_gs_finance_corp(self):
        assert _known_issuer_or_none("GS Finance Corp.") == "Goldman Sachs"

    def test_jefferies(self):
        assert _known_issuer_or_none("Jefferies Financial Group Inc.  (JEF)") == "Jefferies"

    def test_jpmorgan_filer(self):
        assert _known_issuer_or_none("JPMorgan Chase Financial Co. LLC") == "JPMorgan"

    def test_etf_trust_rejected(self):
        # Il prospetto della trust IBIT stessa NON è una nota → None
        assert _known_issuer_or_none("iShares Bitcoin Trust  (IBIT)") is None

    def test_franklin_rejected(self):
        assert _known_issuer_or_none("Franklin Templeton Digital Holdings Trust  (EZBC)") is None

    def test_random_company_rejected(self):
        assert _known_issuer_or_none("Flybondi Ltd") is None

    def test_none(self):
        assert _known_issuer_or_none(None) is None


class TestDetectUnderlying:
    def test_ibit_ticker(self):
        assert _detect_underlying("linked to the performance of IBIT shares") == "IBIT"

    def test_ibit_fund_name(self):
        assert _detect_underlying("the iShares® Bitcoin Trust ETF") == "IBIT"

    def test_fbtc_ticker(self):
        assert _detect_underlying("notes linked to FBTC") == "FBTC"

    def test_fbtc_fund_name(self):
        assert _detect_underlying("the Fidelity® Wise Origin® Bitcoin Fund") == "FBTC"

    def test_bitb(self):
        assert _detect_underlying("linked to the Bitwise Bitcoin ETF (BITB)") == "BITB"

    def test_arkb(self):
        assert _detect_underlying("the ARK 21Shares Bitcoin ETF") == "ARKB"

    def test_multi_underlier_prefers_ibit(self):
        # Nota multi-asset Goldman: si traccia la gamba IBIT.
        text = "linked to the lesser performing of FBTC and the iShares Bitcoin Trust"
        assert _detect_underlying(text) == "IBIT"

    def test_no_match_falls_back_to_ibit(self):
        assert _detect_underlying("generic prospectus text") == "IBIT"

    def test_ibit_substring_not_matched(self):
        # "prohibited" contiene "ibit" ma non deve matchare (word boundary).
        assert _detect_underlying("transfer is prohibited under FBTC terms") == "FBTC"


class TestParseBatchInceptionFilter:
    """parse_batch deve scartare i falsi positivi pre-inception IBIT."""

    def _parser_with_stub(self, monkeypatch, notes_by_url):
        parser = ProspectusParser()

        def fake_parse(filing_meta):
            return notes_by_url[filing_meta["url"]]

        monkeypatch.setattr(parser, "parse", fake_parse)
        return parser

    def test_drops_pre_inception(self, monkeypatch):
        notes = {
            "u_old": StructuredNote(filing_url="u_old", issuer="UBS",
                                    issue_date=date(2008, 4, 30)),
            "u_new": StructuredNote(filing_url="u_new", issuer="JPMorgan",
                                    issue_date=date(2026, 2, 2)),
        }
        parser = self._parser_with_stub(monkeypatch, notes)
        out = parser.parse_batch([{"url": "u_old"}, {"url": "u_new"}])
        urls = [n.filing_url for n in out]
        assert urls == ["u_new"]

    def test_keeps_note_without_date(self, monkeypatch):
        # Senza issue_date non possiamo affermare che sia un falso positivo: si tiene.
        notes = {"u": StructuredNote(filing_url="u", issuer="JPMorgan", issue_date=None)}
        parser = self._parser_with_stub(monkeypatch, notes)
        out = parser.parse_batch([{"url": "u"}])
        assert len(out) == 1

    def test_drops_unknown_issuer(self, monkeypatch):
        # Filer non-emittente (issuer=None) → falso positivo, scartato.
        notes = {
            "junk": StructuredNote(filing_url="junk", issuer=None,
                                   product_type="leveraged_note"),
            "ok":   StructuredNote(filing_url="ok", issuer="Goldman Sachs"),
        }
        parser = self._parser_with_stub(monkeypatch, notes)
        out = parser.parse_batch([{"url": "junk", "entity_name": "iShares Bitcoin Trust"},
                                  {"url": "ok"}])
        assert [n.filing_url for n in out] == ["ok"]


class TestParsePreliminaryVsFinal:
    """Test end-to-end di parse() con _fetch monkeypatchato (no rete)."""

    def _parse_html(self, monkeypatch, html, filing_date="2026-02-02"):
        parser = ProspectusParser()
        monkeypatch.setattr(parser, "_fetch", lambda url: html)
        return parser.parse({"url": "https://example.com/x.htm",
                             "entity_name": "JPMorgan Chase Financial Co. LLC",
                             "filing_date": filing_date})

    def test_preliminary_no_invented_values(self, monkeypatch):
        html = (
            "<html><body>PRELIMINARY PRICING SUPPLEMENT "
            "Subject to completion dated February 2, 2026. "
            "Auto Callable Notes Linked to the iShares Bitcoin Trust ETF. "
            "Contingent Interest if the closing price is at least 70.00% of the "
            "Initial Value. For each $1,000 stated principal amount of the notes."
            "</body></html>"
        )
        note = self._parse_html(monkeypatch, html)
        assert note.is_preliminary is True
        assert note.notional_usd is None       # denominazione non inventata
        assert note.initial_level is None       # Initial Value non ancora fissato
        # la barriera percentuale resta estratta
        assert any(b.level_pct == 70.0 for b in note.barriers)

    def test_final_extracts_values(self, monkeypatch):
        html = (
            "<html><body>PRICING SUPPLEMENT January 9, 2026. "
            "Auto Callable Notes Linked to the iShares Bitcoin Trust ETF. "
            "Aggregate principal amount: $5,000,000. "
            "Initial Value: $51.16. Barrier at 70.00% of the Initial Value."
            "</body></html>"
        )
        note = self._parse_html(monkeypatch, html, filing_date="2026-01-09")
        assert note.is_preliminary is False
        assert note.notional_usd == 5_000_000.0
        assert note.initial_level == 51.16
        assert note.issue_date.isoformat() == "2026-01-09"

    def test_initial_from_closing_price(self, monkeypatch):
        # Formato Goldman: "Initial ETF price: $42.73, which is the closing price …".
        # Il $100.00 ipotetico NON deve essere preso.
        html = (
            "<html><body>PRICING SUPPLEMENT. Notes Linked to the ETF. "
            "Initial ETF price: $42.73, which is the closing price of the ETF on the "
            "pricing date. The hypothetical initial ETF price of $100.00 has been chosen "
            "for illustrative purposes only.</body></html>"
        )
        note = self._parse_html(monkeypatch, html, filing_date="2026-04-20")
        assert note.initial_level == 42.73

    def test_initial_morgan_stanley_level_beats_denomination(self, monkeypatch):
        # Formato Morgan Stanley: "Initial level: $63.10, which is the closing level
        # of the underlier on the strike date". Il taglio nominale ($100 per security)
        # presente altrove NON deve essere preso (regressione: prima vinceva 100.00).
        html = (
            "<html><body>PRICING SUPPLEMENT August 29, 2025. "
            "Notes Linked to the iShares Bitcoin Trust ETF. "
            "Stated principal amount: $1,000 per security. "
            "Estimated value on the pricing date: $933.20 per security. "
            "Initial level: $63.10, which is the closing level of the underlier on the "
            "strike date. Downside threshold: 70.00% of the initial level."
            "</body></html>"
        )
        note = self._parse_html(monkeypatch, html, filing_date="2025-08-29")
        assert note.initial_level == 63.10

    def test_initial_from_reverse_barrier(self, monkeypatch):
        # Formato Morgan Stanley: "$31.41, 84.90% of the initial underlying value"
        # → initial = 31.41 / 0.849 ≈ 37.0.
        html = (
            "<html><body>PRICING SUPPLEMENT. Notes Linked to the ETF. "
            "Strike value: $31.41, 84.90% of the initial underlying value. "
            "Autocall barrier value: $40.70, 110.00% of the initial underlying value."
            "</body></html>"
        )
        note = self._parse_html(monkeypatch, html, filing_date="2026-03-01")
        assert note.initial_level == 37.0


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
