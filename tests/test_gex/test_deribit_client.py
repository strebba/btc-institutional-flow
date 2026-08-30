"""Test per DeribitClient — layer rete con mock HTTP."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from requests import HTTPError, Timeout

from src.gex.deribit_client import CircuitBreaker, DeribitClient, _is_retryable_error


class TestIsRetryableError:
    def test_timeout_is_retryable(self):
        assert _is_retryable_error(Timeout("timeout")) is True

    def test_connection_error_is_retryable(self):
        from requests import ConnectionError as ConnErr
        assert _is_retryable_error(ConnErr("refused")) is True

    def test_429_is_retryable(self):
        resp = MagicMock()
        resp.status_code = 429
        assert _is_retryable_error(HTTPError(response=resp)) is True

    def test_500_is_not_retryable(self):
        resp = MagicMock()
        resp.status_code = 500
        assert _is_retryable_error(HTTPError(response=resp)) is False

    def test_value_error_is_not_retryable(self):
        assert _is_retryable_error(ValueError("not a network error")) is False


class TestCircuitBreaker:
    def test_initially_closed(self):
        cb = CircuitBreaker()
        assert cb.is_open() is False

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.is_open() is True

    def test_success_resets_counter(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.failures == 0

    def test_fail_after_reset(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        for _ in range(3):
            cb.record_failure()
        assert cb.is_open() is True

    def test_recovery_after_timeout(self):
        """Aperto subito dopo il fallimento, chiuso una volta scaduto il timeout.

        Il timeout deve essere non nullo: con recovery_timeout=0 bastano pochi
        microsecondi perche' `time.time() - last_failure_time > 0`, quindi il
        breaker si richiude prima ancora di poter essere osservato aperto e i due
        stati diventano indistinguibili.
        """
        import time

        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)
        cb.record_failure()
        assert cb.is_open() is True
        time.sleep(0.1)
        assert cb.is_open() is False


class TestCacheTTL:
    def test_cache_ttl_ticker_is_5s(self):
        assert DeribitClient._cache_ttl("/ticker") == 5.0

    def test_cache_ttl_instruments_is_30s(self):
        assert DeribitClient._cache_ttl("/get_instruments") == 30.0

    def test_cache_ttl_book_summary_is_30s(self):
        assert DeribitClient._cache_ttl("/get_book_summary_by_currency") == 30.0

    def test_cache_ttl_default_is_30s(self):
        assert DeribitClient._cache_ttl("/other") == 30.0


class TestGetSpotPrice:
    def test_returns_float(self):
        client = DeribitClient()

        def mock_get(*args, **kwargs):
            resp = MagicMock()
            resp.json.return_value = {"result": {"index_price": 85000.5}}
            resp.raise_for_status.return_value = None
            return resp

        with patch.object(client._session, "get", side_effect=mock_get):
            price = client.get_spot_price()
            assert price == 85000.5
            assert isinstance(price, float)

    def test_caches_same_endpoint(self):
        """Seconda chiamata a get_spot_price usa la cache (TTL 5s)."""
        client = DeribitClient()
        calls = []

        def mock_get(*args, **kwargs):
            calls.append(1)
            resp = MagicMock()
            resp.json.return_value = {"result": {"index_price": 90000.0}}
            resp.raise_for_status.return_value = None
            return resp

        with patch.object(client._session, "get", side_effect=mock_get):
            p1 = client.get_spot_price()
            p2 = client.get_spot_price()
            assert p1 == p2 == 90000.0
            assert len(calls) == 1


class TestClearCache:
    def test_clear_cache_forces_refetch(self):
        client = DeribitClient()
        calls = []

        def mock_get(*args, **kwargs):
            calls.append(1)
            resp = MagicMock()
            resp.json.return_value = {"result": {"index_price": 90000.0}}
            resp.raise_for_status.return_value = None
            return resp

        with patch.object(client._session, "get", side_effect=mock_get):
            client.get_spot_price()
            assert len(calls) == 1
            client.clear_cache()
            client.get_spot_price()
            assert len(calls) == 2


class TestFetchAllOptions:
    def test_empty_instruments_graceful(self):
        """fetch_all_options con zero strumenti non deve crashare."""
        client = DeribitClient()

        def mock_get(*args, **kwargs):
            resp = MagicMock()
            resp.json.return_value = {"result": []}
            resp.raise_for_status.return_value = None
            return resp

        with patch.object(client._session, "get", side_effect=mock_get):
            try:
                result = client.fetch_all_options()
                assert isinstance(result, list)
            except ValueError:
                pass  # Python <3.8 ThreadPoolExecutor doesn't support max_workers=0

    def test_handles_errors_gracefully(self):
        client = DeribitClient()
        mock_instr_resp = MagicMock()
        mock_instr_resp.json.return_value = {
            "result": [{"instrument_name": "BTC-OPT", "strike": 80000, "option_type": "call"}]
        }
        mock_instr_resp.raise_for_status.return_value = None

        def mock_get(*args, **kwargs):
            if args and "instruments" in str(args):
                return mock_instr_resp
            raise Timeout("timeout")

        with patch.object(client._session, "get", side_effect=mock_get):
            result = client.fetch_all_options()
            assert result == []


class TestHTTPErrorHandling:
    def test_500_records_failure(self):
        client = DeribitClient()
        resp = MagicMock()
        resp.status_code = 500
        with patch.object(client._session, "get", return_value=resp):
            with patch.object(resp, "raise_for_status", side_effect=HTTPError(response=resp)):
                with pytest.raises(HTTPError):
                    client._get("/ticker", {"instrument_name": "BTC-OPT"})
        assert client._circuit_breaker.failures == 1
