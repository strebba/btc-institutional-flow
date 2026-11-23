"""Client per l'API pubblica di Deribit (senza autenticazione).

Endpoints usati (tutti GET, no auth):
  GET /public/get_instruments?currency=BTC&kind=option&expired=false
  GET /public/get_book_summary_by_currency?currency=BTC&kind=option
  GET /public/get_index_price?index_name=btc_usd
  GET /public/ticker?instrument_name={name}   (solo get_ticker, non nella chain)
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from src.config import get_settings, setup_logging
from src.gex.greeks import black_scholes_greeks, time_to_expiry_years

_log = setup_logging("gex.deribit")

# Module-level throttle: shared across ALL DeribitClient instances so concurrent
# requests from different endpoints don't collectively exceed Deribit's rate limit.
_global_throttle_lock = threading.Lock()
_global_last_request: float = 0.0


def parse_index_chart(result: Any) -> list[tuple[datetime, float]]:
    """Converte il result di get_index_chart_data in punti (timestamp, prezzo)."""
    points: list[tuple[datetime, float]] = []
    if not isinstance(result, list):
        return points
    for item in result:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            ts = datetime.fromtimestamp(float(item[0]) / 1000.0, tz=timezone.utc)
            points.append((ts, float(item[1])))
        except (TypeError, ValueError, OSError):
            continue
    return points


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
        self._cache_ts: dict[str, float] = {}
        self._circuit_breaker = CircuitBreaker(failure_threshold=20, recovery_timeout=60)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _cache_ttl(endpoint: str) -> float:
        """TTL in secondi per tipo di endpoint. Spot/ticker brevi, instrument list lunghi."""
        if "ticker" in endpoint:
            return 5.0
        if "get_instruments" in endpoint or "get_book_summary" in endpoint:
            return 30.0
        return 30.0

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
            age = time.monotonic() - self._cache_ts.get(cache_key, 0)
            if age < self._cache_ttl(endpoint):
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
            self._cache_ts[cache_key] = time.monotonic()
            self._circuit_breaker.record_success()
            return result
        except (requests.Timeout, requests.ConnectionError):
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

    # ──────────────────────────────────────────────────────────────────────────
    # Public API methods
    # ──────────────────────────────────────────────────────────────────────────

    def get_spot_price(self, asset: str = "BTC") -> float:
        """Restituisce il prezzo spot corrente dell'indice Deribit dell'asset.

        Args:
            asset: "BTC" (indice btc_usd) o "ETH" (indice eth_usd).

        Returns:
            float: prezzo in USD.
        """
        from src.assets import get_asset

        index_name = get_asset(asset).deribit_index
        result = self._get("/get_index_price", {"index_name": index_name})
        return float(result["index_price"])

    def get_index_chart(
        self, index_name: str = "btc_usd", chart_range: str = "1m",
    ) -> list[tuple[datetime, float]]:
        """Storico dell'indice (public/get_index_chart_data).

        Ogni punto è ``(timestamp UTC, prezzo medio)``. ``chart_range`` è uno
        di 1h, 1d, 2d, 1m, 1y, all: la finestra è ancorata a *adesso*, non a
        una data. Il chiamante deve scartare i punti successivi allo slot.
        """
        result = self._get(
            "/get_index_chart_data",
            {"index_name": index_name, "range": chart_range},
        )
        return parse_index_chart(result)

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

    def get_book_summary(self, currency: str = "BTC", kind: str = "option") -> list[dict]:
        """Riepilogo di mercato di TUTTI gli strumenti di una valuta, in una chiamata.

        Per ogni strumento: open_interest, mark_price, mark_iv, underlying_price,
        bid_price/ask_price. Niente greche (le ricostruisce fetch_all_options).
        """
        result = self._get(
            "/get_book_summary_by_currency", {"currency": currency, "kind": kind},
        )
        return result or []

    # ──────────────────────────────────────────────────────────────────────────
    # Batch fetching
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _merge_option(instr: dict, book: dict, now_ms: float) -> dict:
        """Unisce strumento (strike/tipo/scadenza) e book summary (OI/IV/prezzi).

        Delta, gamma e vega sono Black-Scholes sullo stesso input che usa Deribit
        (forward della scadenza = ``underlying_price``, ``mark_iv``, r=q=0):
        verificato il 23/11/2026 su un campione live, delta e vega coincidono
        con il ticker alla quarta cifra. La gamma esce anzi a piena precisione,
        mentre il ticker la arrotonda a 5 decimali. ``vega`` segue la convenzione
        Deribit: per punto di volatilità. Greche a 0 se non calcolabili
        (IV assente, scadenza entro un'ora).
        """
        strike = float(instr.get("strike", 0) or 0)
        option_type = instr.get("option_type", "") or ""
        expiration = instr.get("expiration_timestamp", 0) or 0
        mark_iv = float(book.get("mark_iv", 0) or 0)
        underlying = float(book.get("underlying_price", 0) or 0)

        greeks = black_scholes_greeks(
            spot=underlying,
            strike=strike,
            t_years=time_to_expiry_years(expiration, now_ms),
            sigma=mark_iv / 100.0,                 # Deribit quota la IV in %
            option_type=option_type,
        )
        return {
            "instrument_name": instr.get("instrument_name", ""),
            "strike": strike,
            "option_type": option_type,
            "expiration_timestamp": expiration,
            "gamma": greeks.gamma if greeks else 0.0,
            "delta": greeks.delta if greeks else 0.0,
            "vega": greeks.vega / 100.0 if greeks else 0.0,
            "open_interest": float(book.get("open_interest", 0) or 0),
            "mark_price": float(book.get("mark_price", 0) or 0),
            "mark_iv": mark_iv,
            "underlying_price": underlying,
            "best_bid": float(book.get("bid_price", 0) or 0),
            "best_ask": float(book.get("ask_price", 0) or 0),
        }

    def fetch_all_options(self, currency: str = "BTC") -> list[dict]:
        """Scarica tutte le opzioni attive con greche e OI, in DUE richieste.

        ``get_instruments`` (strike, tipo, scadenza) + ``get_book_summary_by_currency``
        (OI, IV, prezzi) uniti per nome. Prima si chiamava ``/ticker`` una volta
        per strumento: ~1000 richieste per BTC, ~800 per ETH. Bastava che due
        processi (API e cron) girassero insieme dallo stesso IP perché Deribit
        rispondesse 429 a metà della chain: il GEX usciva calcolato su metà
        dell'OI e ``/api/gex`` andava in timeout (novembre 2026).

        Args:
            currency: "BTC" o "ETH".

        Returns:
            list[dict]: un dict per opzione con instrument_name, strike,
                option_type, expiration_timestamp, gamma, delta, vega,
                open_interest, mark_price, mark_iv, underlying_price,
                best_bid, best_ask. Lista vuota se una delle due chiamate fallisce.
        """
        try:
            instruments = self.get_instruments(currency=currency)
            books = self.get_book_summary(currency=currency)
        except Exception as e:
            _log.warning("Fetch chain %s fallito: %s", currency, e)
            return []

        book_by_name = {b.get("instrument_name"): b for b in books}
        now_ms = time.time() * 1000.0

        results: list[dict] = []
        missing = 0
        for instr in instruments:
            book = book_by_name.get(instr.get("instrument_name"))
            if book is None:
                missing += 1   # listato da pochi secondi, non ancora nel book
                continue
            results.append(self._merge_option(instr, book, now_ms))

        _log.info(
            "Fetch completato %s: %d opzioni valide, %d senza book summary",
            currency, len(results), missing,
        )
        return results
