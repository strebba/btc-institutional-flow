"""Client per CoinGlass API v4 (Startup tier).

Fornisce:
  - fetch_etf_flows()          → list[EtfFlowData]  (IBIT + tutti i BTC ETF spot US)
  - fetch_funding_rate_history() → pd.Series         (funding rate OI-weighted, daily)
  - fetch_aggregated_oi_history() → pd.Series        (futures OI aggregato, daily)

Autenticazione: header CG-API-KEY.
API key: env var COINGLASS_API_KEY > settings.yaml coinglass.api_key.

Docs: https://docs.coinglass.com
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime, timezone
from typing import Any

import requests

try:
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

    _HAS_TENACITY = True
except ImportError:
    _HAS_TENACITY = False

    # No-op fallback: il decorator non fa nulla senza tenacity
    def retry(*args, **kwargs):  # type: ignore[misc]
        def decorator(fn):
            return fn

        return decorator

    def retry_if_exception_type(*args):  # type: ignore[misc]
        return None

    def stop_after_attempt(n):  # type: ignore[misc]
        return None

    def wait_exponential(**kwargs):  # type: ignore[misc]
        return None


import pandas as pd

from src.config import get_settings, setup_logging
from src.flows.models import EtfFlowData

_log = setup_logging("flows.coinglass")


class CoinGlassError(Exception):
    """Errore transitorio CoinGlass (rete, 429, 5xx) — viene ritentato da tenacity."""


class CoinGlassApiError(Exception):
    """Errore permanente dall'API CoinGlass (parametri invalidi, exchange non supportato).
    Non è una sottoclasse di CoinGlassError: tenacity non lo ritenta."""


class CoinGlassClient:
    """Client HTTP per CoinGlass API v4.

    Args:
        cfg: configurazione (default da settings.yaml sezione 'coinglass').
    """

    BASE_URL = "https://open-api-v4.coinglass.com"

    def __init__(self, cfg: dict | None = None) -> None:
        settings = get_settings()
        self._cfg = cfg or settings.get("coinglass", {})

        # API key: env var ha precedenza su settings.yaml
        self._api_key = (os.getenv("COINGLASS_API_KEY") or self._cfg.get("api_key", "")).strip()

        self._timeout = self._cfg.get("timeout_s", 15)
        self._rate_limit = self._cfg.get("rate_limit_rps", 1.0)
        self._last_call_ts = 0.0

        self._session = requests.Session()
        self._session.headers.update(
            {
                "CG-API-KEY": self._api_key,
                "Accept": "application/json",
            }
        )
        _log.info("CoinGlass client initialized (API key present: %s)", bool(self._api_key))

    def _throttle(self) -> None:
        """Rispetta il rate limit configurato (req/s)."""
        if self._rate_limit > 0:
            elapsed = time.time() - self._last_call_ts
            wait = (1.0 / self._rate_limit) - elapsed
            if wait > 0:
                time.sleep(wait)
        self._last_call_ts = time.time()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=1, max=10),
        retry=retry_if_exception_type(CoinGlassError),
    )
    def _get(self, path: str, params: dict | None = None) -> Any:
        """Esegue GET con retry su errori transitori.

        Args:
            path: percorso relativo es. '/api/etf/bitcoin/flow-history'.
            params: query string params.

        Returns:
            Il campo 'data' della risposta JSON CoinGlass.

        Raises:
            CoinGlassError: su errore API o chiave mancante.
        """
        if not self._api_key:
            raise CoinGlassApiError(
                "CoinGlass API key mancante. "
                "Imposta env var COINGLASS_API_KEY o config coinglass.api_key"
            )

        self._throttle()
        url = self._cfg.get("base_url", self.BASE_URL) + path
        try:
            resp = self._session.get(url, params=params or {}, timeout=self._timeout)
        except requests.Timeout:
            _log.warning("CoinGlass timeout: %s %s", path, params)
            raise CoinGlassError(f"Timeout requesting {path}")
        except requests.ConnectionError as e:
            _log.warning("CoinGlass connection error: %s", e)
            raise CoinGlassError(f"Connection error requesting {path}")

        if resp.status_code == 401:
            raise CoinGlassError("CoinGlass: API key non valida (401)")
        if resp.status_code == 403:
            _log.warning("CoinGlass: endpoint %s non disponibile per il tier corrente (403)", path)
            raise CoinGlassError(f"Endpoint {path} not available (403)")
        if resp.status_code == 429:
            raise CoinGlassError("CoinGlass: rate limit superato (429)")

        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            _log.error("CoinGlass HTTP error: %s (status=%d)", e, resp.status_code)
            raise CoinGlassError(f"HTTP error: {resp.status_code}")

        body = resp.json()
        if body.get("code") != "0":
            msg = body.get("msg", body)
            _log.warning("CoinGlass API error: %s", msg)
            raise CoinGlassApiError(f"API error: {msg}")

        return body.get("data", [])

    # ─── ETF Flows ────────────────────────────────────────────────────────────

    def fetch_etf_flows(self, days: int = 365) -> list[EtfFlowData]:
        """Scarica i flussi giornalieri di tutti i Bitcoin ETF spot USA.

        Args:
            days: numero di giorni storici da richiedere.

        Returns:
            list[EtfFlowData] con source='coinglass', ordinata per data crescente.
            Lista vuota se l'endpoint non risponde o non ci sono dati.
        """
        try:
            data = self._get("/api/etf/bitcoin/flow-history", {"limit": min(days, 500)})
        except (CoinGlassError, CoinGlassApiError) as e:
            _log.warning("CoinGlass ETF flows: %s", e)
            return []
        except Exception as e:
            _log.warning("CoinGlass ETF flows (rete): %s", e)
            return []

        if not isinstance(data, list):
            _log.warning("CoinGlass ETF flows: risposta inattesa %s", type(data))
            return []

        results: list[EtfFlowData] = []
        for entry in data:
            # Supporta due strutture di risposta:
            # v4: {"timestamp": ms, "etf_flows": [{"etf_ticker": "IBIT", "flow_usd": ...}]}
            # v3: {"date": "YYYY-MM-DD", "flows_by_ticker": {"IBIT": ...}}
            raw_date = entry.get("timestamp") or entry.get("date") or entry.get("t")
            if raw_date is None:
                continue

            try:
                if isinstance(raw_date, (int, float)):
                    flow_date = datetime.fromtimestamp(raw_date / 1000, tz=timezone.utc).date()
                else:
                    flow_date = date.fromisoformat(str(raw_date)[:10])
            except (ValueError, OSError):
                _log.debug("CoinGlass: data non parsabile: %s", raw_date)
                continue

            # v4 format: etf_flows è una lista di {etf_ticker, flow_usd}
            etf_flows_list: list[dict] = entry.get("etf_flows") or []
            if etf_flows_list:
                for etf_entry in etf_flows_list:
                    ticker = etf_entry.get("etf_ticker") or etf_entry.get("ticker")
                    flow_val = etf_entry.get("flow_usd")
                    if not ticker or flow_val is None:
                        continue
                    try:
                        results.append(
                            EtfFlowData(
                                date=flow_date,
                                ticker=str(ticker),
                                flow_usd=float(flow_val),
                                source="coinglass",
                            )
                        )
                    except (TypeError, ValueError):
                        continue
            else:
                # v3 format: flows_by_ticker è un dict {TICKER: value}
                flows_by_ticker: dict = entry.get("flows_by_ticker") or {}
                for ticker, flow_val in flows_by_ticker.items():
                    if flow_val is None:
                        continue
                    try:
                        results.append(
                            EtfFlowData(
                                date=flow_date,
                                ticker=str(ticker),
                                flow_usd=float(flow_val),
                                source="coinglass",
                            )
                        )
                    except (TypeError, ValueError):
                        continue

        _log.info("CoinGlass ETF flows: %d record scaricati", len(results))
        return sorted(results, key=lambda r: r.date)

    # ─── Funding Rate ─────────────────────────────────────────────────────────

    def fetch_funding_rate_history(self, days: int = 365) -> pd.Series:
        """Scarica il funding rate OI-weighted (daily, annualizzato).

        Args:
            days: numero di giorni storici.

        Returns:
            pd.Series con DatetimeIndex (UTC) e valori float (funding rate 8h in %).
            Serie vuota su errore.
        """
        try:
            data = self._get(
                "/api/futures/funding-rate/oi-weight-history",
                {"symbol": "BTC", "interval": "8h", "limit": min(days * 3, 1000)},
            )
        except (CoinGlassError, Exception) as e:
            _log.warning("CoinGlass funding rate: %s", e)
            return pd.Series(dtype=float, name="funding_rate")

        if not isinstance(data, list) or not data:
            return pd.Series(dtype=float, name="funding_rate")

        # Risposta OHLC: [t, o, h, l, c] oppure dict con t/c
        rows: list[tuple] = []
        for item in data:
            if isinstance(item, list) and len(item) >= 5:
                ts_ms, _, _, _, close = item[0], item[1], item[2], item[3], item[4]
            elif isinstance(item, dict):
                ts_ms = item.get("t") or item.get("time")
                close = item.get("c") or item.get("close")
            else:
                continue
            if ts_ms is None or close is None:
                continue
            try:
                dt = datetime.fromtimestamp(float(ts_ms) / 1000, tz=timezone.utc)
                rows.append((dt, float(close)))
            except (TypeError, ValueError, OSError):
                continue

        if not rows:
            return pd.Series(dtype=float, name="funding_rate")

        # Ricampiona a daily usando l'ultimo valore del giorno
        df = pd.DataFrame(rows, columns=["ts", "rate"]).set_index("ts")
        daily = df["rate"].resample("1D").last().dropna()
        daily.name = "funding_rate"
        return daily

    # ─── Aggregated Futures OI ────────────────────────────────────────────────

    def fetch_aggregated_oi_history(self, days: int = 365) -> pd.Series:
        """Scarica l'open interest futures aggregato cross-exchange (daily, USD).

        Args:
            days: numero di giorni storici.

        Returns:
            pd.Series con DatetimeIndex (UTC) e valori float (OI in USD).
            Serie vuota su errore.
        """
        try:
            data = self._get(
                "/api/futures/open-interest/aggregated-history",
                {"symbol": "BTC", "interval": "1d", "limit": min(days, 500)},
            )
        except (CoinGlassError, Exception) as e:
            _log.warning("CoinGlass aggregated OI: %s", e)
            return pd.Series(dtype=float, name="futures_oi_usd")

        if not isinstance(data, list) or not data:
            return pd.Series(dtype=float, name="futures_oi_usd")

        rows: list[tuple] = []
        for item in data:
            if isinstance(item, list) and len(item) >= 2:
                ts_ms, oi = item[0], item[1]
            elif isinstance(item, dict):
                ts_ms = item.get("t") or item.get("time")
                # "close" è il campo reale nella risposta v4 OHLC aggregated OI
                oi = (
                    item.get("close") or item.get("c") or item.get("openInterest") or item.get("oi")
                )
            else:
                continue
            if ts_ms is None or oi is None:
                continue
            try:
                dt = datetime.fromtimestamp(float(ts_ms) / 1000, tz=timezone.utc)
                rows.append((dt, float(oi)))
            except (TypeError, ValueError, OSError):
                continue

        if not rows:
            return pd.Series(dtype=float, name="futures_oi_usd")

        df = pd.DataFrame(rows, columns=["ts", "oi"]).set_index("ts")
        daily = df["oi"].resample("1D").last().dropna()
        daily.name = "futures_oi_usd"
        return daily

    # ─── Long/Short Ratio ────────────────────────────────────────────────────

    def fetch_long_short_ratio(self, days: int = 90) -> pd.Series:
        """Scarica il rapporto long/short globale degli account futures BTC.

        Un valore > 1.0 indica più account long che short.
        Valore > 2.0 = folla retail crowded long = segnale contrarian bearish.

        Args:
            days: numero di giorni storici.

        Returns:
            pd.Series con DatetimeIndex (tz-naive) e valori float (ratio).
            Serie vuota su errore.
        """
        data = None

        # Try different exchange/symbol combinations based on CoinGlass API docs
        param_sets = [
            {"exchange": "Binance", "symbol": "BTCUSDT", "interval": "1d", "limit": min(days, 500)},
            {"exchange": "OKX", "symbol": "BTCUSDT", "interval": "1d", "limit": min(days, 500)},
            {"exchange": "Bybit", "symbol": "BTCUSDT", "interval": "1d", "limit": min(days, 500)},
            {
                "exchange": "Binance,OKX,Bybit",
                "symbol": "BTCUSDT",
                "interval": "1d",
                "limit": min(days, 500),
            },
        ]

        for params in param_sets:
            try:
                data = self._get(
                    "/api/futures/global-long-short-account-ratio/history",
                    params,
                )
                if data:
                    break
            except (CoinGlassError, CoinGlassApiError):
                continue

        if data is None:
            _log.warning("CoinGlass long/short ratio: all parameter combinations failed")
            return pd.Series(dtype=float, name="long_short_ratio")

        if not isinstance(data, list) or not data:
            return pd.Series(dtype=float, name="long_short_ratio")

        rows: list[tuple] = []
        for item in data:
            if isinstance(item, list) and len(item) >= 2:
                ts_ms, ratio = item[0], item[1]
            elif isinstance(item, dict):
                ts_ms = item.get("t") or item.get("time") or item.get("createTime")
                # Updated field names from CoinGlass API docs
                ratio = (
                    item.get("global_account_long_short_ratio")
                    or item.get("longShortRatio")
                    or item.get("ratio")
                    or item.get("c")
                )
            else:
                continue
            if ts_ms is None or ratio is None:
                continue
            try:
                dt = datetime.fromtimestamp(float(ts_ms) / 1000, tz=timezone.utc)
                rows.append((dt, float(ratio)))
            except (TypeError, ValueError, OSError):
                continue

        if not rows:
            return pd.Series(dtype=float, name="long_short_ratio")

        df = pd.DataFrame(rows, columns=["ts", "ratio"]).set_index("ts")
        daily = df["ratio"].resample("1D").last().dropna()
        daily.index = daily.index.tz_localize(None)  # tz-naive per join
        daily.name = "long_short_ratio"
        return daily

    # ─── Liquidations ────────────────────────────────────────────────────────

    def fetch_liquidations(self, days: int = 90) -> pd.DataFrame:
        """Scarica la storia delle liquidazioni futures BTC (long + short).

        Utile per identificare cascade risk: liquidazioni >500M USD in 24h
        indicano eventi di stress con potenziale momentum direzionale.

        Args:
            days: numero di giorni storici.

        Returns:
            pd.DataFrame con DatetimeIndex (tz-naive) e colonne:
            'long_usd' (liquidazioni long), 'short_usd' (liquidazioni short),
            'total_usd' (somma).
            DataFrame vuoto su errore.
        """
        _EMPTY = pd.DataFrame(columns=["long_usd", "short_usd", "total_usd"])

        data = None
        # Try different exchange_list based on CoinGlass API docs - requires exchange_list param
        param_sets = [
            {
                "exchange_list": "Binance,OKX,Bybit",
                "symbol": "BTC",
                "interval": "1d",
                "limit": min(days, 500),
            },
            {
                "exchange_list": "Binance",
                "symbol": "BTC",
                "interval": "1d",
                "limit": min(days, 500),
            },
            {"exchange_list": "OKX", "symbol": "BTC", "interval": "1d", "limit": min(days, 500)},
            {"exchange_list": "Bybit", "symbol": "BTC", "interval": "1d", "limit": min(days, 500)},
        ]

        for params in param_sets:
            try:
                data = self._get(
                    "/api/futures/liquidation/aggregated-history",
                    params,
                )
                if data:
                    break
            except (CoinGlassError, CoinGlassApiError):
                continue

        if data is None:
            _log.warning("CoinGlass liquidations: all parameter combinations failed")
            return _EMPTY

        if not isinstance(data, list) or not data:
            return _EMPTY

        rows: list[dict] = []
        for item in data:
            if isinstance(item, list) and len(item) >= 3:
                ts_ms, long_v, short_v = item[0], item[1], item[2]
            elif isinstance(item, dict):
                ts_ms = item.get("t") or item.get("time")
                # Updated field names from CoinGlass API docs
                long_v = (
                    item.get("aggregated_long_liquidation_usd")
                    or item.get("long_liquidation_usd_24h")
                    or item.get("longLiquidationUsd")
                    or item.get("buyUsd")
                    or item.get("long")
                    or 0.0
                )
                short_v = (
                    item.get("aggregated_short_liquidation_usd")
                    or item.get("short_liquidation_usd_24h")
                    or item.get("shortLiquidationUsd")
                    or item.get("sellUsd")
                    or item.get("short")
                    or 0.0
                )
            else:
                continue
            if ts_ms is None:
                continue
            try:
                dt = datetime.fromtimestamp(float(ts_ms) / 1000, tz=timezone.utc)
                rows.append({"ts": dt, "long_usd": float(long_v), "short_usd": float(short_v)})
            except (TypeError, ValueError, OSError):
                continue

        if not rows:
            return _EMPTY

        df = pd.DataFrame(rows).set_index("ts")
        daily = df.resample("1D").sum()
        daily.index = daily.index.tz_localize(None)
        daily["total_usd"] = daily["long_usd"] + daily["short_usd"]
        return daily.dropna(how="all")

    # ─── Taker Buy/Sell Volume ────────────────────────────────────────────────

    def fetch_taker_volume(self, days: int = 90) -> pd.Series:
        """Scarica il rapporto taker buy/(buy+sell) per futures BTC.

        Indica la pressione direzionale degli ordini market (aggressori):
        > 0.55 = pressione acquisto dominante; < 0.45 = pressione vendita.

        Args:
            days: numero di giorni storici.

        Returns:
            pd.Series con DatetimeIndex (tz-naive) e valori float (ratio 0-1).
            Serie vuota su errore.
        """
        data = None
        # Use correct endpoint from CoinGlass API docs: /api/futures/taker-buy-sell-volume/exchange-list
        param_sets = [
            {"symbol": "BTC", "range": "1d"},
            {"symbol": "BTC", "range": "4h"},
            {"symbol": "BTC", "range": "24h"},
        ]

        for params in param_sets:
            try:
                data = self._get(
                    "/api/futures/taker-buy-sell-volume/exchange-list",
                    params,
                )
                if data:
                    break
            except (CoinGlassError, CoinGlassApiError):
                continue

        if data is None:
            _log.warning("CoinGlass taker volume: all parameter combinations failed")
            return pd.Series(dtype=float, name="taker_buy_ratio")

        # Formato dict: snapshot puntuale {"buy_ratio": 51.01, "sell_ratio": 48.99, ...}
        if isinstance(data, dict):
            buy_ratio = data.get("buy_ratio")
            if buy_ratio is not None:
                now = datetime.now(timezone.utc).replace(tzinfo=None)
                return pd.Series([float(buy_ratio) / 100], index=[now], name="taker_buy_ratio")
            return pd.Series(dtype=float, name="taker_buy_ratio")

        # Formato lista: storico con t/buyVolume/sellVolume
        if not isinstance(data, list) or not data:
            return pd.Series(dtype=float, name="taker_buy_ratio")

        rows: list[tuple] = []
        for item in data:
            ts_ms = item.get("t") or item.get("time")
            buy_v = item.get("buyVolume", 0.0)
            sell_v = item.get("sellVolume", 0.0)
            total = (buy_v or 0.0) + (sell_v or 0.0)
            if ts_ms is None or total == 0:
                continue
            try:
                dt = datetime.fromtimestamp(float(ts_ms) / 1000, tz=timezone.utc)
                rows.append((dt, float(buy_v or 0.0) / total))
            except (TypeError, ValueError, OSError):
                continue

        if not rows:
            return pd.Series(dtype=float, name="taker_buy_ratio")

        df = pd.DataFrame(rows, columns=["ts", "ratio"]).set_index("ts")
        daily = df["ratio"].resample("1D").last().dropna()
        daily.index = daily.index.tz_localize(None)
        daily.name = "taker_buy_ratio"
        return daily

    # ─── Options Info ─────────────────────────────────────────────────────────

    def fetch_options_info(self, symbol: str = "BTC") -> list[dict]:
        """OI totale, volume e market share per exchange (opzioni BTC/ETH).

        Usato per calcolare il coverage score del GEX: confronta l'OI che
        abbiamo fetchato da Deribit con l'OI totale dichiarato da CoinGlass.

        Args:
            symbol: "BTC" o "ETH".

        Returns:
            list[dict] con un elemento per exchange. Campi principali:
            exchange_name, open_interest (contratti), open_interest_usd,
            oi_market_share (%), volume_usd_24h.
            Lista vuota su errore o endpoint non disponibile nel tier.
        """
        try:
            data = self._get("/api/option/info", {"symbol": symbol})
        except (CoinGlassError, CoinGlassApiError) as e:
            _log.warning("CoinGlass options info [%s]: %s", symbol, e)
            return []
        except Exception as e:
            _log.warning("CoinGlass options info (rete) [%s]: %s", symbol, e)
            return []

        if not isinstance(data, list):
            _log.warning("CoinGlass options info: risposta inattesa %s", type(data))
            return []

        return data

    # ─── Options Max Pain ─────────────────────────────────────────────────────

    def fetch_options_max_pain(
        self, symbol: str = "BTC", exchange: str = "Deribit"
    ) -> list[dict]:
        """Max pain e call/put OI per expiry per un singolo exchange.

        Usato per costruire put/call ratio e max pain multi-exchange
        (Deribit + CME + Binance + OKX) più rappresentativi del posizionamento
        istituzionale complessivo.

        Args:
            symbol: "BTC" o "ETH".
            exchange: "Deribit", "CME", "Binance", "OKX", "Bybit".

        Returns:
            list[dict] con un elemento per data di scadenza. Campi principali:
            date (str "YYMMDD"), max_pain_price (str),
            call_open_interest (contratti), call_open_interest_notional (USD),
            put_open_interest (contratti), put_open_interest_notional (USD).
            Lista vuota se endpoint non disponibile o dati assenti.
        """
        try:
            data = self._get(
                "/api/option/max-pain",
                {"symbol": symbol, "exchange": exchange},
            )
        except (CoinGlassError, CoinGlassApiError) as e:
            _log.warning("CoinGlass max pain [%s/%s]: %s", symbol, exchange, e)
            return []
        except Exception as e:
            _log.warning("CoinGlass max pain (rete) [%s/%s]: %s", symbol, exchange, e)
            return []

        if not isinstance(data, list):
            _log.warning(
                "CoinGlass max pain [%s/%s]: risposta inattesa %s",
                symbol, exchange, type(data),
            )
            return []

        return data
