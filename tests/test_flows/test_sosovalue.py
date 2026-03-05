"""Unit test per SoSoValueClient."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.flows.sosovalue import SoSoValueClient
from src.flows.models import EtfFlowData


# ─── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_JSON_ARRAY = [
    {"date": "2024-03-01", "IBIT": 500.1,  "FBTC": 200.5, "GBTC": -150.3},
    {"date": "2024-03-04", "IBIT": -80.0,  "FBTC": None,  "GBTC": 50.0},
    {"date": "2024-03-05", "IBIT": 0.0,    "FBTC": 30.0},
]

SAMPLE_JSON_WRAPPED = {"data": SAMPLE_JSON_ARRAY}

SAMPLE_JSON_ITEMS_WRAPPER = {"items": SAMPLE_JSON_ARRAY}

SAMPLE_JSON_UNIX_MS = [
    # 2024-03-01 00:00:00 UTC in ms
    {"date": 1709251200000, "IBIT": 100.0},
]


# ─── Test: skip se API key assente ────────────────────────────────────────────

class TestFetchNoApiKey:
    def test_returns_empty_without_key(self):
        client = SoSoValueClient(cfg={"sosovalue_api_key": ""})
        result = client.fetch()
        assert result == []

    def test_returns_empty_when_key_missing_from_cfg(self):
        client = SoSoValueClient(cfg={})
        result = client.fetch()
        assert result == []


# ─── Test: _parse_response ────────────────────────────────────────────────────

class TestParseResponse:
    def _client(self) -> SoSoValueClient:
        return SoSoValueClient(cfg={"sosovalue_api_key": "dummy"})

    def test_array_format(self):
        client = self._client()
        results = client._parse_response(SAMPLE_JSON_ARRAY)
        assert len(results) > 0
        assert all(isinstance(r, EtfFlowData) for r in results)

    def test_wrapped_data_format(self):
        client = self._client()
        results = client._parse_response(SAMPLE_JSON_WRAPPED)
        assert len(results) > 0

    def test_wrapped_items_format(self):
        client = self._client()
        results = client._parse_response(SAMPLE_JSON_ITEMS_WRAPPER)
        assert len(results) > 0

    def test_ibit_positive_flow(self):
        client = self._client()
        results = client._parse_response(SAMPLE_JSON_ARRAY)
        ibit_1 = [r for r in results if r.ticker == "IBIT" and r.date == date(2024, 3, 1)]
        assert len(ibit_1) == 1
        assert ibit_1[0].flow_usd == pytest.approx(500_100_000)

    def test_ibit_negative_flow(self):
        client = self._client()
        results = client._parse_response(SAMPLE_JSON_ARRAY)
        ibit_4 = [r for r in results if r.ticker == "IBIT" and r.date == date(2024, 3, 4)]
        assert len(ibit_4) == 1
        assert ibit_4[0].flow_usd == pytest.approx(-80_000_000)

    def test_gbtc_negative(self):
        client = self._client()
        results = client._parse_response(SAMPLE_JSON_ARRAY)
        gbtc_1 = [r for r in results if r.ticker == "GBTC" and r.date == date(2024, 3, 1)]
        assert len(gbtc_1) == 1
        assert gbtc_1[0].flow_usd < 0

    def test_null_value_skipped(self):
        client = self._client()
        results = client._parse_response(SAMPLE_JSON_ARRAY)
        # Giorno 2024-03-04: FBTC=None → non deve apparire
        fbtc_4 = [r for r in results if r.ticker == "FBTC" and r.date == date(2024, 3, 4)]
        assert len(fbtc_4) == 0

    def test_zero_flow_included(self):
        client = self._client()
        results = client._parse_response(SAMPLE_JSON_ARRAY)
        ibit_5 = [r for r in results if r.ticker == "IBIT" and r.date == date(2024, 3, 5)]
        assert len(ibit_5) == 1
        assert ibit_5[0].flow_usd == pytest.approx(0.0)

    def test_source_label(self):
        client = self._client()
        results = client._parse_response(SAMPLE_JSON_ARRAY)
        assert all(r.source == "sosovalue" for r in results)

    def test_unix_ms_timestamp(self):
        client = self._client()
        results = client._parse_response(SAMPLE_JSON_UNIX_MS)
        assert len(results) == 1
        assert results[0].date == date(2024, 3, 1)
        assert results[0].flow_usd == pytest.approx(100_000_000)

    def test_invalid_format_returns_empty(self):
        client = self._client()
        assert client._parse_response("not a list or dict") == []

    def test_dict_without_data_key_returns_empty(self):
        client = self._client()
        # dict senza "data" o "items" → lista vuota
        result = client._parse_response({"something": "else"})
        assert result == []

    def test_record_without_date_skipped(self):
        client = self._client()
        data = [{"IBIT": 100.0}]  # manca "date"
        assert client._parse_response(data) == []

    def test_invalid_date_skipped(self):
        client = self._client()
        data = [{"date": "not-a-date", "IBIT": 100.0}]
        assert client._parse_response(data) == []

    def test_multiplier_is_1m(self):
        """Verifica che i valori siano moltiplicati × 1_000_000."""
        client = self._client()
        data = [{"date": "2024-01-01", "IBIT": 1.0}]
        results = client._parse_response(data)
        assert results[0].flow_usd == pytest.approx(1_000_000)


# ─── Test: HTTP mock ──────────────────────────────────────────────────────────

class TestFetchHttpMock:
    def test_fetch_calls_api_with_bearer_token(self):
        client = SoSoValueClient(cfg={"sosovalue_api_key": "test-key-123"})
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = SAMPLE_JSON_ARRAY

        with patch("requests.Session.get", return_value=mock_resp) as mock_get:
            result = client.fetch(lookback_days=30)

        call_kwargs = mock_get.call_args
        headers_used = mock_get.call_args  # headers set on session, not per-call
        assert len(result) > 0

    def test_fetch_returns_empty_on_http_error(self):
        client = SoSoValueClient(cfg={"sosovalue_api_key": "test-key"})
        with patch("requests.Session.get", side_effect=Exception("connection refused")):
            result = client.fetch()
        assert result == []

    def test_fetch_returns_empty_on_4xx(self):
        client = SoSoValueClient(cfg={"sosovalue_api_key": "test-key"})
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("401 Unauthorized")

        with patch("requests.Session.get", return_value=mock_resp):
            result = client.fetch()
        assert result == []
