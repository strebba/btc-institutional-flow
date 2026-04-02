"""Client per l'API pubblica di Deribit (senza autenticazione).

Endpoints usati (tutti GET, no auth):
  GET /public/get_instruments?currency=BTC&kind=option&expired=false
  GET /public/ticker?instrument_name={name}
  GET /public/get_index_price?index_name=btc_usd
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from src.config import get_settings, setup_logging

_log = setup_logging("gex.deribit")

# Module-level throttle: shared across ALL DeribitClient instances so concurrent
# requests from different endpoints don't collectively exceed Deribit's rate limit.
_global_throttle_lock = threading.Lock()
_global_last_request: float = 0.0


def _is_retryable_error(exc: Exception) -> bool:
    """Tenacity predicate: retry on network errors and 429 rate-limit responses."""
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True
    if isinstance(exc, requests.HTTPError):
        return exc.response is not None and exc.response.status_code == 429
    return False


class CircuitBreaker:
    """Circuit breaker per evitare cascata errori durante fetch massivo."""

    def __init__(self, failure_threshold: int = 20, recovery_timeout: int = 60) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time: float = 0.0
        self._lock = threading.Lock()

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1
            self.last_failure_time = time.time()

    def record_success(self) -> None:
        with self._lock:
            self.failures = 0

    def is_open(self) -> bool:
        with self._lock:
            if self.failures >= self.failure_threshold:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.failures = 0
                    return False
                return True
            return False


class DeribitClient:
    """Client per le API pubbliche di Deribit.

    Args:
        cfg: blocco configurazione deribit (da settings.yaml).
    """

    BASE_URL = "https://www.deribit.com/api/v2/public"

    def __init__(self, cfg: dict | None = None) -> None:
        self._cfg = cfg or get_settings()["deribit"]
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": "ibit-gamma-tracker/1.0",
                "Accept": "application/json",
            }
        )
        self._min_interval = 1.0 / self._cfg["rate_limit_rps"]
        self._cache: dict[str, Any] = {}
        self._circuit_breaker = CircuitBreaker(failure_threshold=20, recovery_timeout=60)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _throttle(self) -> None:
        """Rispetta il rate limit configurato (thread-safe, globale tra istanze)."""
        global _global_last_request
        with _global_throttle_lock:
            elapsed = time.monotonic() - _global_last_request
            wait = self._min_interval - elapsed
            if wait > 0:
                time.sleep(wait)
            _global_last_request = time.monotonic()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=10),
        retry=retry_if_exception(_is_retryable_error),
        reraise=True,
    )
    def _get(self, endpoint: str, params: dict | None = None) -> Any:
        """GET con throttle, retry e caching.

        Args:
            endpoint: es. "/get_instruments".
            params: query parameters.

        Returns:
            Any: campo "result" della risposta JSON Deribit.

        Raises:
            RuntimeError: se l'API ritorna un errore applicativo.
            requests.HTTPError: per errori HTTP.
        """
        cache_key = f"{endpoint}:{params}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self._circuit_breaker.is_open():
            _log.warning("Circuit breaker OPEN - throttling requests")
            time.sleep(5)

        self._throttle()
        url = f"{self.BASE_URL}{endpoint}"
        try:
            resp = self._session.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            if "error" in data:
                raise RuntimeError(f"Deribit API error: {data['error']}")

            result = data.get("result")
            self._cache[cache_key] = result
            self._circuit_breaker.record_success()
            return result
        except (requests.Timeout, requests.ConnectionError) as e:
            self._circuit_breaker.record_failure()
            raise
        except requests.HTTPError as e:
            # NOTE: bool(requests.Response) == response.ok, which is False for 4xx/5xx.
            # Must use "is not None" to distinguish "no response" from "error response".
            status = e.response.status_code if e.response is not None else 0
            if status == 429:
                # 429 è un rate-limit applicativo, non un errore di rete:
                # non conta come failure del circuit breaker (altrimenti il circuit
                # si apre e dorme 5s su ogni request, rallentando il fetch).
                retry_after = int(e.response.headers.get("Retry-After", 2))
                _log.debug("429 on %s — sleeping %ds before retry", endpoint, retry_after)
                time.sleep(retry_after)
            else:
                self._circuit_breaker.record_failure()
                _log.warning("HTTP error fetching %s: %s (status=%d)", endpoint, e, status)
            raise

    def clear_cache(self) -> None:
        """Svuota la cache in memoria."""
        self._cache.clear()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API methods
    # ──────────────────────────────────────────────────────────────────────────

    def get_spot_price(self) -> float:
        """Restituisce il prezzo spot corrente di BTC-USD.

        Returns:
            float: prezzo in USD.
        """
        result = self._get("/get_index_price", {"index_name": "btc_usd"})
        return float(result["index_price"])

    def get_instruments(
        self,
        currency: str = "BTC",
        kind: str = "option",
        expired: bool = False,
    ) -> list[dict]:
        """Restituisce la lista degli strumenti attivi.

        Args:
            currency: "BTC" o "ETH".
            kind: "option" o "future".
            expired: se True include gli strumenti scaduti.

        Returns:
            list[dict]: lista di strumenti con nome, strike, tipo, scadenza.
        """
        result = self._get(
            "/get_instruments",
            {
                "currency": currency,
                "kind": kind,
                "expired": str(expired).lower(),
            },
        )
        return result or []

    def get_ticker(self, instrument_name: str) -> dict:
        """Restituisce il ticker completo per un singolo strumento.

        Contiene: mark_price, mark_iv, greeks (delta, gamma, theta, vega),
        open_interest, best_bid, best_ask.

        Args:
            instrument_name: es. "BTC-25JAN25-100000-C".

        Returns:
            dict: dati ticker.
        """
        result = self._get("/ticker", {"instrument_name": instrument_name})
        return result or {}

    def get_order_book(
        self,
        instrument_name: str,
        depth: int = 1,
    ) -> dict:
        """Restituisce il book ordini (con greche e OI).

        Args:
            instrument_name: nome strumento Deribit.
            depth: profondità del book (default 1).

        Returns:
            dict: book con gamma, delta, open_interest, etc.
        """
        result = self._get(
            "/get_order_book",
            {
                "instrument_name": instrument_name,
                "depth": depth,
            },
        )
        return result or {}

    # ──────────────────────────────────────────────────────────────────────────
    # Batch fetching
    # ──────────────────────────────────────────────────────────────────────────

    def _fetch_one_option(self, instr: dict) -> dict | None:
        """Scarica ticker + greche per un singolo strumento.

        Usato in parallelo da fetch_all_options. Il throttle thread-safe
        garantisce che le richieste vengano avviate al rate configurato
        mentre le HTTP calls si sovrappongono concorrentemente.

        Args:
            instr: dict strumento da get_instruments().

        Returns:
            dict con i dati dell'opzione, oppure None se errore.
        """
        name = instr.get("instrument_name", "")
        if not name:
            return None
        try:
            ticker = self.get_ticker(name)
            greeks = ticker.get("greeks", {})
            return {
                "instrument_name": name,
                "strike": float(instr.get("strike", 0)),
                "option_type": instr.get("option_type", ""),
                "expiration_timestamp": instr.get("expiration_timestamp", 0),
                "gamma": float(greeks.get("gamma", 0) or 0),
                "delta": float(greeks.get("delta", 0) or 0),
                "vega": float(greeks.get("vega", 0) or 0),
                "open_interest": float(ticker.get("open_interest", 0) or 0),
                "mark_price": float(ticker.get("mark_price", 0) or 0),
                "mark_iv": float(ticker.get("mark_iv", 0) or 0),
                "underlying_price": float(ticker.get("underlying_price", 0) or 0),
                "best_bid": float(ticker.get("best_bid_price", 0) or 0),
                "best_ask": float(ticker.get("best_ask_price", 0) or 0),
            }
        except Exception as e:
            _log.warning("Errore fetch opzione %s: %s", name, e)
            return None

    def fetch_all_options(self, currency: str = "BTC") -> list[dict]:
        """Scarica i dati di tutte le opzioni attive con greche e OI.

        Usa ThreadPoolExecutor per avviare le richieste in parallelo mantenendo
        il rate limit via _throttle thread-safe. Con 500 opzioni a 15 rps
        il tempo di fetch scende da ~100s (sequenziale) a ~35s.

        Args:
            currency: "BTC" o "ETH".

        Returns:
            list[dict]: lista di dati per ogni opzione con campi:
                instrument_name, strike, option_type, expiration_timestamp,
                gamma, open_interest, mark_price, mark_iv, delta, underlying_price.
        """
        instruments = self.get_instruments(currency=currency)
        _log.info("Strumenti %s attivi: %d — avvio fetch concorrente", currency, len(instruments))

        results: list[dict] = []
        errors = 0
        # max_workers > rate_limit_rps: i thread extra restano in attesa del throttle
        max_workers = min(len(instruments), 20)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._fetch_one_option, instr): instr for instr in instruments
            }
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result is not None:
                    results.append(result)
                else:
                    errors += 1
                if i % 50 == 0:
                    _log.info("Progresso: %d/%d opzioni processate", i, len(instruments))

        _log.info(
            "Fetch completato: %d opzioni valide, %d errori",
            len(results),
            errors,
        )
        return results
