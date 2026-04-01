"""Test per CoinGlassClient — mock via unittest.mock."""
from __future__ import annotations

from datetime import timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.flows.coinglass_client import CoinGlassClient, CoinGlassError
from src.flows.models import EtfFlowData


# ─── Helpers ─────────────────────────────────────────────────────────────────

BASE = "https://open-api-v4.coinglass.com"


def _client(api_key: str = "test-key") -> CoinGlassClient:
    cfg = {
        "api_key":        api_key,
        "base_url":       BASE,
        "timeout_s":      5,
        "rate_limit_rps": 0,
    }
    return CoinGlassClient(cfg=cfg)


def _mock_get(client: CoinGlassClient, return_data) -> MagicMock:
    """Patcha client._get per restituire return_data senza HTTP."""
    m = MagicMock(return_value=return_data)
    client._get = m
    return m


def _mock_get_raise(client: CoinGlassClient, exc: Exception) -> MagicMock:
    m = MagicMock(side_effect=exc)
    client._get = m
    return m


# ─── ETF Flows ────────────────────────────────────────────────────────────────

class TestFetchEtfFlows:
    def test_returns_etf_flow_data_list(self) -> None:
        """Risposta valida → list[EtfFlowData]."""
        client = _client()
        _mock_get(client, [
            {"date": "2026-03-01", "flows_by_ticker": {"IBIT": 300e6, "FBTC": 50e6}},
            {"date": "2026-03-02", "flows_by_ticker": {"IBIT": -100e6}},
        ])
        result = client.fetch_etf_flows()
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(r, EtfFlowData) for r in result)

    def test_filters_ibit_correctly(self) -> None:
        """Contiene record IBIT con il valore corretto."""
        client = _client()
        _mock_get(client, [
            {"date": "2026-03-10", "flows_by_ticker": {"IBIT": 200e6, "FBTC": 30e6}},
        ])
        result = client.fetch_etf_flows()
        ibit = [r for r in result if r.ticker == "IBIT"]
        assert len(ibit) == 1
        assert abs(ibit[0].flow_usd - 200e6) < 1.0

    def test_source_label_is_coinglass(self) -> None:
        """source deve essere 'coinglass'."""
        client = _client()
        _mock_get(client, [{"date": "2026-03-10", "flows_by_ticker": {"IBIT": 100e6}}])
        result = client.fetch_etf_flows()
        assert all(r.source == "coinglass" for r in result)

    def test_handles_api_error_returns_empty(self) -> None:
        """CoinGlassError → lista vuota (non rilancia)."""
        client = _client()
        _mock_get_raise(client, CoinGlassError("test error"))
        assert client.fetch_etf_flows() == []

    def test_handles_network_error_returns_empty(self) -> None:
        """Eccezione generica di rete → lista vuota."""
        client = _client()
        _mock_get_raise(client, ConnectionError("timeout"))
        assert client.fetch_etf_flows() == []

    def test_handles_empty_data_array(self) -> None:
        """data: [] → lista vuota."""
        client = _client()
        _mock_get(client, [])
        assert client.fetch_etf_flows() == []

    def test_missing_api_key_raises_coinglass_error(self) -> None:
        """Chiave vuota → CoinGlassError al primo _get."""
        client = _client(api_key="")
        client._session = MagicMock()
        with pytest.raises(CoinGlassError, match="API key"):
            client._get("/api/etf/bitcoin/flow-history")

    def test_unix_ms_timestamp_parsed(self) -> None:
        """Timestamp ms UNIX viene convertito in date."""
        ts_ms = 1741564800000  # 2025-03-10 UTC
        client = _client()
        _mock_get(client, [{"date": ts_ms, "flows_by_ticker": {"IBIT": 50e6}}])
        result = client.fetch_etf_flows()
        assert len(result) >= 1
        assert result[0].date.year == 2025

    def test_sorted_by_date_ascending(self) -> None:
        """Risultati ordinati per data crescente."""
        client = _client()
        _mock_get(client, [
            {"date": "2026-03-02", "flows_by_ticker": {"IBIT": 100e6}},
            {"date": "2026-03-01", "flows_by_ticker": {"IBIT": 200e6}},
        ])
        result = client.fetch_etf_flows()
        ibit = [r for r in result if r.ticker == "IBIT"]
        assert ibit[0].date < ibit[1].date

    def test_negative_flow_preserved(self) -> None:
        """Outflow negativo mantenuto con segno."""
        client = _client()
        _mock_get(client, [{"date": "2026-03-15", "flows_by_ticker": {"IBIT": -500e6}}])
        result = client.fetch_etf_flows()
        ibit = [r for r in result if r.ticker == "IBIT"]
        assert ibit[0].flow_usd < 0


# ─── Funding Rate ─────────────────────────────────────────────────────────────

class TestFetchFundingRate:
    def test_returns_series(self) -> None:
        """Risposta OHLC valida → pd.Series non vuota."""
        ts1 = 1741478400000
        ts2 = 1741564800000
        client = _client()
        _mock_get(client, [
            [ts1, 0.01, 0.02, 0.00, 0.015],
            [ts2, 0.015, 0.03, 0.01, 0.02],
        ])
        result = client.fetch_funding_rate_history(days=7)
        assert isinstance(result, pd.Series)
        assert not result.empty

    def test_returns_empty_on_error(self) -> None:
        """Errore → Serie vuota."""
        client = _client()
        _mock_get_raise(client, CoinGlassError("err"))
        assert client.fetch_funding_rate_history().empty

    def test_series_has_datetimeindex(self) -> None:
        """La serie ha DatetimeIndex."""
        ts = 1741564800000
        client = _client()
        _mock_get(client, [[ts, 0.01, 0.02, 0.00, 0.015]])
        result = client.fetch_funding_rate_history()
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_empty_data_returns_empty_series(self) -> None:
        """data: [] → Serie vuota."""
        client = _client()
        _mock_get(client, [])
        assert client.fetch_funding_rate_history().empty
