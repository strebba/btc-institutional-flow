"""Unit test per EdgarNportClient."""
from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.flows.edgar_nport import EdgarNportClient, _avg
from src.flows.models import EtfFlowData


# ─── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<edgarSubmission xmlns="http://www.sec.gov/edgar/nport">
  <genInfo>
    <regName>iShares Bitcoin Trust ETF</regName>
    <repPdDate>2024-01-31</repPdDate>
  </genInfo>
  <fundInfo>
    <totAssets>5000000000.00</totAssets>
    <netAssets>4998000000.00</netAssets>
    <shrOutstanding>100000000</shrOutstanding>
    <navPerShare>49.98</navPerShare>
  </fundInfo>
</edgarSubmission>"""

SAMPLE_XML_NO_NS = """<?xml version="1.0"?>
<root>
  <totAssets>2500000000.00</totAssets>
  <shrOutstanding>50000000</shrOutstanding>
</root>"""

SAMPLE_XML_MISSING_FIELDS = """<?xml version="1.0"?>
<root>
  <someOtherField>123</someOtherField>
</root>"""

MONTHLY_POINTS_2 = [
    {
        "date":               date(2024, 1, 31),
        "total_assets":       5_000_000_000,
        "shares_outstanding": 100_000_000,
        "nav_per_share":      50.0,
    },
    {
        "date":               date(2024, 2, 29),
        "total_assets":       6_000_000_000,
        "shares_outstanding": 110_000_000,
        "nav_per_share":      54.5,
    },
]

MONTHLY_POINTS_3 = MONTHLY_POINTS_2 + [
    {
        "date":               date(2024, 3, 31),
        "total_assets":       5_500_000_000,
        "shares_outstanding": 105_000_000,
        "nav_per_share":      52.4,
    },
]


# ─── Test: _avg helper ────────────────────────────────────────────────────────

class TestAvgHelper:
    def test_both_values(self):
        assert _avg(10.0, 20.0) == pytest.approx(15.0)

    def test_first_none(self):
        assert _avg(None, 20.0) == pytest.approx(20.0)

    def test_second_none(self):
        assert _avg(10.0, None) == pytest.approx(10.0)

    def test_both_none(self):
        assert _avg(None, None) == pytest.approx(0.0)


# ─── Test: _extract_xml_value ─────────────────────────────────────────────────

class TestExtractXmlValue:
    def test_tag_with_namespace(self):
        val = EdgarNportClient._extract_xml_value(SAMPLE_XML, ["totAssets"])
        assert val == pytest.approx(5_000_000_000.0)

    def test_shares_outstanding_with_namespace(self):
        val = EdgarNportClient._extract_xml_value(SAMPLE_XML, ["shrOutstanding"])
        assert val == pytest.approx(100_000_000.0)

    def test_nav_per_share(self):
        val = EdgarNportClient._extract_xml_value(SAMPLE_XML, ["navPerShare"])
        assert val == pytest.approx(49.98)

    def test_tag_without_namespace(self):
        val = EdgarNportClient._extract_xml_value(SAMPLE_XML_NO_NS, ["totAssets"])
        assert val == pytest.approx(2_500_000_000.0)

    def test_missing_tag_returns_none(self):
        val = EdgarNportClient._extract_xml_value(SAMPLE_XML_MISSING_FIELDS, ["totAssets"])
        assert val is None

    def test_multiple_candidate_tags_first_wins(self):
        val = EdgarNportClient._extract_xml_value(
            SAMPLE_XML, ["totAssets", "netAssets"]
        )
        assert val == pytest.approx(5_000_000_000.0)

    def test_fallback_to_second_candidate(self):
        val = EdgarNportClient._extract_xml_value(
            SAMPLE_XML, ["nonExistentTag", "shrOutstanding"]
        )
        assert val == pytest.approx(100_000_000.0)

    def test_invalid_xml_returns_none(self):
        val = EdgarNportClient._extract_xml_value("not valid xml <<<", ["totAssets"])
        assert val is None


# ─── Test: _parse_nport_xml_text ──────────────────────────────────────────────

class TestParseNportXmlText:
    def _client(self) -> EdgarNportClient:
        return EdgarNportClient(cfg={"edgar": {}, "flows": {}})

    def test_full_xml(self):
        client = self._client()
        point = client._parse_nport_xml_text(SAMPLE_XML, date(2024, 1, 31))
        assert point is not None
        assert point["date"] == date(2024, 1, 31)
        assert point["total_assets"] == pytest.approx(5_000_000_000.0)
        assert point["shares_outstanding"] == pytest.approx(100_000_000.0)
        assert point["nav_per_share"] == pytest.approx(49.98)

    def test_xml_no_ns(self):
        client = self._client()
        point = client._parse_nport_xml_text(SAMPLE_XML_NO_NS, date(2024, 2, 28))
        assert point is not None
        assert point["total_assets"] == pytest.approx(2_500_000_000.0)
        assert point["shares_outstanding"] == pytest.approx(50_000_000.0)

    def test_nav_computed_from_assets_and_shares(self):
        """Se navPerShare non c'è, viene calcolato come totalAssets / shares."""
        client = self._client()
        point = client._parse_nport_xml_text(SAMPLE_XML_NO_NS, date(2024, 2, 28))
        assert point is not None
        expected_nav = 2_500_000_000.0 / 50_000_000.0
        assert point["nav_per_share"] == pytest.approx(expected_nav)

    def test_missing_both_fields_returns_none(self):
        client = self._client()
        result = client._parse_nport_xml_text(SAMPLE_XML_MISSING_FIELDS, date(2024, 1, 1))
        assert result is None

    def test_invalid_xml_returns_none(self):
        client = self._client()
        result = client._parse_nport_xml_text("<broken <xml>", date(2024, 1, 1))
        assert result is None


# ─── Test: _interpolate_to_daily ──────────────────────────────────────────────

class TestInterpolateToDaily:
    def _client(self) -> EdgarNportClient:
        return EdgarNportClient(cfg={"edgar": {}, "flows": {}})

    def test_two_points_produce_daily_flows(self):
        client = self._client()
        results = client._interpolate_to_daily(MONTHLY_POINTS_2)
        assert len(results) > 0
        assert all(isinstance(r, EtfFlowData) for r in results)

    def test_source_label(self):
        client = self._client()
        results = client._interpolate_to_daily(MONTHLY_POINTS_2)
        assert all(r.source == "edgar_nport_interpolated" for r in results)

    def test_ticker_is_ibit(self):
        client = self._client()
        results = client._interpolate_to_daily(MONTHLY_POINTS_2)
        assert all(r.ticker == "IBIT" for r in results)

    def test_flow_direction_inflow(self):
        """Δshares > 0 → flusso netto positivo (inflow)."""
        client = self._client()
        results = client._interpolate_to_daily(MONTHLY_POINTS_2)
        # Da 100M a 110M shares → inflow
        assert all(r.flow_usd > 0 for r in results)

    def test_flow_direction_outflow(self):
        """Δshares < 0 → flusso netto negativo (outflow)."""
        client = self._client()
        points = [
            {"date": date(2024, 1, 31), "shares_outstanding": 110_000_000,
             "nav_per_share": 54.5, "total_assets": 6e9},
            {"date": date(2024, 2, 29), "shares_outstanding": 100_000_000,
             "nav_per_share": 50.0, "total_assets": 5e9},
        ]
        results = client._interpolate_to_daily(points)
        assert all(r.flow_usd < 0 for r in results)

    def test_total_flow_equals_delta_shares_times_nav(self):
        """La somma dei flussi giornalieri deve ≈ Δshares × avg_NAV."""
        client = self._client()
        p1, p2 = MONTHLY_POINTS_2[0], MONTHLY_POINTS_2[1]
        delta_shares = p2["shares_outstanding"] - p1["shares_outstanding"]
        avg_nav = (p1["nav_per_share"] + p2["nav_per_share"]) / 2
        expected_total = delta_shares * avg_nav

        results = client._interpolate_to_daily(MONTHLY_POINTS_2)
        actual_total = sum(r.flow_usd for r in results)
        assert actual_total == pytest.approx(expected_total, rel=1e-6)

    def test_daily_flow_uniform(self):
        """Tutti i giorni dello stesso intervallo hanno lo stesso flusso."""
        client = self._client()
        results = client._interpolate_to_daily(MONTHLY_POINTS_2)
        flows = [r.flow_usd for r in results]
        assert all(abs(f - flows[0]) < 1e-6 for f in flows)

    def test_three_points(self):
        """Con 3 punti mensili, si producono due segmenti di interpolazione."""
        client = self._client()
        results = client._interpolate_to_daily(MONTHLY_POINTS_3)
        # Il terzo segmento ha Δshares < 0 (105M < 110M)
        # Quindi ci devono essere sia valori positivi che negativi
        assert any(r.flow_usd > 0 for r in results)
        assert any(r.flow_usd < 0 for r in results)

    def test_single_point_returns_empty(self):
        client = self._client()
        results = client._interpolate_to_daily([MONTHLY_POINTS_2[0]])
        assert results == []

    def test_empty_points_returns_empty(self):
        client = self._client()
        results = client._interpolate_to_daily([])
        assert results == []

    def test_missing_shares_skips_interval(self):
        """Intervallo con shares_outstanding=None viene saltato."""
        client = self._client()
        points = [
            {"date": date(2024, 1, 31), "shares_outstanding": None,
             "nav_per_share": 50.0, "total_assets": None},
            {"date": date(2024, 2, 29), "shares_outstanding": 100_000_000,
             "nav_per_share": 50.0, "total_assets": 5e9},
        ]
        results = client._interpolate_to_daily(points)
        assert results == []

    def test_dates_are_sequential(self):
        """Le date dei flussi giornalieri devono essere consecutive."""
        client = self._client()
        results = client._interpolate_to_daily(MONTHLY_POINTS_2)
        dates = sorted(r.date for r in results)
        for i in range(1, len(dates)):
            assert (dates[i] - dates[i - 1]).days == 1

    def test_no_future_dates(self):
        """Nessun record deve avere data futura."""
        client = self._client()
        results = client._interpolate_to_daily(MONTHLY_POINTS_2)
        today = date.today()
        assert all(r.date <= today for r in results)


# ─── Test: fetch_monthly_flows con mock rete ─────────────────────────────────

class TestFetchMonthlyFlowsMocked:
    def _client(self) -> EdgarNportClient:
        return EdgarNportClient(cfg={"edgar": {}, "flows": {}})

    def test_returns_empty_on_network_error(self):
        client = self._client()
        with patch.object(client._session, "get", side_effect=Exception("timeout")):
            result = client.fetch_monthly_flows()
        assert result == []

    def test_returns_empty_when_no_nport_filings(self):
        client = self._client()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "filings": {"recent": {"form": ["10-K", "8-K"], "filingDate": ["2024-01-01", "2024-02-01"],
                                   "accessionNumber": ["0001-01-01", "0001-01-02"]}}
        }
        with patch.object(client._session, "get", return_value=mock_resp):
            result = client.fetch_monthly_flows()
        assert result == []
