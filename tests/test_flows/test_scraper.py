"""Test unitari per il Flow Scraper."""
from __future__ import annotations

from datetime import date

import pytest
from src.flows.scraper import _parse_flow_value, _parse_farside_date, FarsideScraper


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


class TestFarsideEth:
    SAMPLE_HTML = """
    <html><body><table>
      <tr><th>Date</th><th>ETHA</th><th>FETH</th><th>ETHE</th><th>Total</th></tr>
      <tr><td>13 Jan 2025</td><td>120.0</td><td>30.0</td><td>(50.0)</td><td>100.0</td></tr>
    </table></body></html>
    """

    def test_aggregate_usa_etha_come_lead(self):
        scraper = FarsideScraper(asset="ETH")
        agg = scraper.aggregate(scraper._parse_table(self.SAMPLE_HTML))
        assert agg[0].lead_flow_usd == pytest.approx(120_000_000)
        assert agg[0].total_flow_usd == pytest.approx(100_000_000)
        assert agg[0].ibit_flow_usd == 0.0

    def test_btc_lead_coincide_con_ibit(self):
        scraper = FarsideScraper()
        agg = scraper.aggregate(scraper._parse_table(TestFarSideScraper.SAMPLE_HTML))
        assert all(a.lead_flow_usd == a.ibit_flow_usd for a in agg)

    def test_to_dataframe_garantisce_la_colonna_etha(self):
        scraper = FarsideScraper(asset="ETH")
        df = scraper.to_dataframe(scraper._parse_table(self.SAMPLE_HTML))
        assert "ETHA" in df.columns and "IBIT" not in df.columns

    def test_url_e_cache_sono_per_asset(self):
        btc, eth = FarsideScraper(), FarsideScraper(asset="ETH")
        assert "ethereum" in eth.farside_url and "bitcoin" in btc.farside_url
        assert btc.cache_file != eth.cache_file

    def test_waterfall_eth_non_usa_le_stime_ibit(self, monkeypatch):
        """SoSoValue, N-PORT e la stima yfinance sono costruiti su IBIT: per ETH darebbero numeri BTC."""
        scraper = FarsideScraper(asset="ETH")
        monkeypatch.setattr("src.flows.coinglass_client.CoinGlassClient.fetch_etf_flows",
                            lambda self, **k: [])
        monkeypatch.setattr(scraper, "_fetch_html", lambda url: "<html></html>")
        monkeypatch.setattr(scraper, "_read_cache", lambda: None)
        monkeypatch.setattr(scraper, "_fetch_yfinance_fallback",
                            lambda: pytest.fail("stima IBIT usata per ETH"))
        assert scraper.fetch() == []
