"""Test unitari per il Flow Scraper."""
from __future__ import annotations

from datetime import date

import pytest
from src.flows.scraper import _parse_flow_value, _parse_farside_date, FarsideScraper
from src.flows.models import EtfFlowData


class TestParseFlowValue:
    def test_positive(self):
        assert _parse_flow_value("123.4") == pytest.approx(123_400_000)

    def test_negative_parentheses(self):
        val = _parse_flow_value("(45.6)")
        assert val == pytest.approx(-45_600_000)

    def test_dash(self):
        assert _parse_flow_value("-") is None

    def test_empty(self):
        assert _parse_flow_value("") is None

    def test_zero(self):
        assert _parse_flow_value("0") == pytest.approx(0.0)

    def test_with_comma(self):
        assert _parse_flow_value("1,234.5") == pytest.approx(1_234_500_000)

    def test_total_header(self):
        assert _parse_flow_value("Total") is None


class TestParseFarsideDate:
    def test_day_month(self):
        d = _parse_farside_date("13 Jan", year_hint=2024)
        assert d == date(2024, 1, 13)

    def test_day_month_year(self):
        d = _parse_farside_date("5 Nov 2024")
        assert d == date(2024, 11, 5)

    def test_invalid(self):
        assert _parse_farside_date("not a date") is None

    def test_december(self):
        d = _parse_farside_date("31 Dec 2024")
        assert d == date(2024, 12, 31)

    def test_uppercase(self):
        d = _parse_farside_date("15 MAR 2025", year_hint=2025)
        assert d == date(2025, 3, 15)


class TestFarSideScraper:
    SAMPLE_HTML = """
    <html><body>
    <table>
      <tr><th>Date</th><th>IBIT</th><th>FBTC</th><th>GBTC</th><th>Total</th></tr>
      <tr><td>13 Jan 2025</td><td>500.1</td><td>200.5</td><td>(150.3)</td><td>550.3</td></tr>
      <tr><td>14 Jan 2025</td><td>-</td><td>100.0</td><td>50.0</td><td>150.0</td></tr>
      <tr><td>15 Jan 2025</td><td>(80.0)</td><td>-</td><td>-</td><td>(80.0)</td></tr>
    </table>
    </body></html>
    """

    def test_parse_table(self):
        scraper = FarsideScraper()
        flows   = scraper._parse_table(self.SAMPLE_HTML)
        assert len(flows) > 0

    def test_ibit_positive(self):
        scraper = FarsideScraper()
        flows   = scraper._parse_table(self.SAMPLE_HTML)
        ibit_13 = [f for f in flows if f.ticker == "IBIT" and f.date == date(2025, 1, 13)]
        assert len(ibit_13) == 1
        assert ibit_13[0].flow_usd == pytest.approx(500_100_000)

    def test_ibit_negative(self):
        scraper = FarsideScraper()
        flows   = scraper._parse_table(self.SAMPLE_HTML)
        ibit_15 = [f for f in flows if f.ticker == "IBIT" and f.date == date(2025, 1, 15)]
        assert len(ibit_15) == 1
        assert ibit_15[0].flow_usd == pytest.approx(-80_000_000)

    def test_gbtc_negative(self):
        scraper = FarsideScraper()
        flows   = scraper._parse_table(self.SAMPLE_HTML)
        gbtc_13 = [f for f in flows if f.ticker == "GBTC" and f.date == date(2025, 1, 13)]
        assert len(gbtc_13) == 1
        assert gbtc_13[0].flow_usd < 0

    def test_dash_skipped(self):
        scraper = FarsideScraper()
        flows   = scraper._parse_table(self.SAMPLE_HTML)
        # Il 14 Jan, IBIT è "-" → non deve essere nel risultato
        ibit_14 = [f for f in flows if f.ticker == "IBIT" and f.date == date(2025, 1, 14)]
        assert len(ibit_14) == 0

    def test_to_dataframe(self):
        scraper = FarsideScraper()
        flows   = scraper._parse_table(self.SAMPLE_HTML)
        df = scraper.to_dataframe(flows)
        assert not df.empty
        assert "IBIT" in df.columns
        assert "total" in df.columns

    def test_aggregate(self):
        scraper = FarsideScraper()
        flows   = scraper._parse_table(self.SAMPLE_HTML)
        agg     = scraper.aggregate(flows)
        assert len(agg) == 3
        # 13 Jan: IBIT=500.1M + FBTC=200.5M + GBTC=-150.3M = 550.3M
        day13 = next(a for a in agg if a.date == date(2025, 1, 13))
        assert day13.ibit_flow_usd == pytest.approx(500_100_000)
        assert day13.total_flow_usd == pytest.approx(550_300_000)
