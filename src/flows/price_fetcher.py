"""Scarica prezzi OHLCV di BTC-USD e IBIT via yfinance.

Calcola il rapporto IBIT/BTC usato per convertire i barrier levels,
e gestisce il download incrementale (solo date mancanti).
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import get_settings, setup_logging
from src.flows.models import PriceData

_log = setup_logging("flows.prices")

# ─── Schema SQLite per i prezzi ──────────────────────────────────────────────

_DDL_PRICES = """
CREATE TABLE IF NOT EXISTS prices (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT    NOT NULL,
    date        TEXT    NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    volume      REAL,
    daily_return REAL,
    created_at  TEXT,
    UNIQUE(ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON prices(ticker, date);
"""


class PriceFetcher:
    """Scarica e mantiene lo storico prezzi di BTC-USD e IBIT.

    Usa SQLite per il caching locale (download incrementale).

    Args:
        db_path: percorso al file SQLite (default da settings.yaml).
        cfg: configurazione flows.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        cfg: dict | None = None,
    ) -> None:
        settings  = get_settings()
        self._cfg = cfg or settings["flows"]
        self._path = Path(db_path or settings["database"]["path"])
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_DDL_PRICES)

    # ──────────────────────────────────────────────────────────────────────────
    # Download
    # ──────────────────────────────────────────────────────────────────────────

    def _last_date_in_db(self, ticker: str) -> Optional[date]:
        """Ritorna l'ultima data presente nel DB per il ticker.

        Args:
            ticker: es. "BTC-USD".

        Returns:
            date | None.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT MAX(date) as d FROM prices WHERE ticker=?", (ticker,)
            ).fetchone()
            return date.fromisoformat(row["d"]) if row["d"] else None

    def _store_df(self, ticker: str, df: pd.DataFrame) -> int:
        """Salva un DataFrame OHLCV nel DB.

        Args:
            ticker: es. "BTC-USD".
            df: DataFrame con colonne Open/High/Low/Close/Volume e DatetimeIndex.

        Returns:
            int: numero di righe inserite.
        """
        if df.empty:
            return 0

        now = datetime.utcnow().isoformat()
        df  = df.copy()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        # Calcola rendimento log giornaliero
        df["daily_return"] = np.log(df["Close"] / df["Close"].shift(1))

        inserted = 0
        with self._conn() as conn:
            for idx, row in df.iterrows():
                d = idx.date() if hasattr(idx, "date") else idx
                try:
                    cur = conn.execute(
                        """
                        INSERT OR IGNORE INTO prices
                            (ticker, date, open, high, low, close, volume, daily_return, created_at)
                        VALUES (?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            ticker, d.isoformat(),
                            float(row.get("Open", 0)),
                            float(row.get("High", 0)),
                            float(row.get("Low",  0)),
                            float(row.get("Close", 0)),
                            float(row.get("Volume", 0)),
                            float(row["daily_return"]) if pd.notna(row["daily_return"]) else None,
                            now,
                        ),
                    )
                    inserted += cur.rowcount  # 1 se inserita, 0 se già esistente
                except Exception as e:
                    _log.debug("Skip %s %s: %s", ticker, d, e)
        return inserted

    def fetch(
        self,
        ticker: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Scarica i prezzi per un ticker, usando la cache locale.

        Se `force_refresh=False`, scarica solo i giorni mancanti dal DB.

        Args:
            ticker: es. "BTC-USD" o "IBIT".
            start_date: data di inizio (default: 365gg fa).
            end_date: data di fine (default: oggi).
            force_refresh: ignora cache e riscarica tutto.

        Returns:
            pd.DataFrame: prezzi con DatetimeIndex.
        """
        end   = end_date   or date.today()
        start = start_date or end - timedelta(days=self._cfg.get("lookback_days", 365))

        if not force_refresh:
            last = self._last_date_in_db(ticker)
            if last and last >= end - timedelta(days=1):
                _log.debug("%s: cache aggiornata fino al %s", ticker, last)
                return self._load_from_db(ticker, start, end)
            # Scarica solo dalla data successiva all'ultima in cache
            if last:
                start = last + timedelta(days=1)
                _log.info("%s: download incrementale da %s", ticker, start)

        _log.info("%s: download %s → %s", ticker, start, end)
        try:
            df = yf.download(
                ticker,
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                progress=False,
                auto_adjust=True,
            )
        except Exception as e:
            _log.error("%s: download fallito — %s", ticker, e)
            return self._load_from_db(ticker, start, end)

        if df.empty:
            _log.warning("%s: nessun dato scaricato", ticker)
            return self._load_from_db(ticker, start, end)

        # Appiattisci colonne multi-index (es. ("Close", "IBIT") → "Close")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        inserted = self._store_df(ticker, df)
        _log.info("%s: %d nuove righe salvate", ticker, inserted)

        return self._load_from_db(ticker, start, end)

    def _load_from_db(
        self, ticker: str, start: date, end: date
    ) -> pd.DataFrame:
        """Carica i prezzi dal DB come DataFrame.

        Args:
            ticker: ticker da caricare.
            start: data inizio.
            end: data fine.

        Returns:
            pd.DataFrame: con DatetimeIndex e colonne close, open, high, low, volume, daily_return.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT date, open, high, low, close, volume, daily_return
                FROM prices
                WHERE ticker=? AND date>=? AND date<=?
                ORDER BY date
                """,
                (ticker, start.isoformat(), end.isoformat()),
            ).fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame([dict(r) for r in rows])
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df

    # ──────────────────────────────────────────────────────────────────────────
    # Dati preconfezionati per i moduli successivi
    # ──────────────────────────────────────────────────────────────────────────

    def fetch_btc(self, **kwargs) -> pd.DataFrame:
        """Scarica BTC-USD.

        Returns:
            pd.DataFrame.
        """
        return self.fetch(self._cfg.get("btc_ticker", "BTC-USD"), **kwargs)

    def fetch_ibit(self, **kwargs) -> pd.DataFrame:
        """Scarica IBIT.

        Returns:
            pd.DataFrame.
        """
        return self.fetch(self._cfg.get("ibit_ticker", "IBIT"), **kwargs)

    def get_ibit_btc_ratio(
        self,
        target_date: Optional[date] = None,
    ) -> Optional[float]:
        """Calcola il rapporto IBIT/BTC per una data specifica.

        Utile per convertire i barrier levels IBIT in prezzi BTC.

        Args:
            target_date: data di riferimento (default: ultima disponibile).

        Returns:
            float | None: rapporto IBIT price / BTC price.
        """
        d = target_date or date.today()
        with self._conn() as conn:
            # Cerca la data più vicina disponibile (fino a 5 giorni prima)
            for delta in range(6):
                check = (d - timedelta(days=delta)).isoformat()
                ibit_row = conn.execute(
                    "SELECT close FROM prices WHERE ticker='IBIT' AND date=?", (check,)
                ).fetchone()
                btc_row  = conn.execute(
                    "SELECT close FROM prices WHERE ticker='BTC-USD' AND date=?", (check,)
                ).fetchone()
                if ibit_row and btc_row and btc_row["close"]:
                    ratio = ibit_row["close"] / btc_row["close"]
                    _log.debug("IBIT/BTC ratio @ %s: %.6f", check, ratio)
                    return ratio
        _log.warning("Impossibile calcolare IBIT/BTC ratio per %s", d)
        return None

    def get_all_prices(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Scarica entrambi (BTC + IBIT) e li unisce in un DataFrame.

        Colonne: btc_close, btc_return, ibit_close, ibit_btc_ratio.

        Args:
            start_date: inizio periodo.
            end_date: fine periodo.

        Returns:
            pd.DataFrame.
        """
        btc  = self.fetch_btc(start_date=start_date,  end_date=end_date)
        ibit = self.fetch_ibit(start_date=start_date, end_date=end_date)

        if btc.empty and ibit.empty:
            return pd.DataFrame()

        merged = pd.DataFrame(index=btc.index if not btc.empty else ibit.index)

        if not btc.empty:
            merged["btc_close"]  = btc["close"]
            merged["btc_return"] = btc["daily_return"]

        if not ibit.empty:
            merged["ibit_close"] = ibit["close"]

        if "btc_close" in merged and "ibit_close" in merged:
            merged["ibit_btc_ratio"] = merged["ibit_close"] / merged["btc_close"]

        # Volatilità realizzata BTC 7 giorni
        if "btc_return" in merged:
            merged["btc_vol_7d"] = (
                merged["btc_return"]
                .rolling(7, min_periods=4)
                .std() * (252 ** 0.5)  # annualizzata
            )

        return merged.dropna(how="all")

    def to_price_data_list(self, ticker: str, df: pd.DataFrame) -> list[PriceData]:
        """Converte DataFrame in lista di PriceData dataclass.

        Args:
            ticker: ticker del dataframe.
            df: DataFrame con colonne close, open, high, low, volume, daily_return.

        Returns:
            list[PriceData].
        """
        result: list[PriceData] = []
        for idx, row in df.iterrows():
            d = idx.date() if hasattr(idx, "date") else idx
            result.append(PriceData(
                date=d,
                ticker=ticker,
                open=float(row.get("open", 0)),
                high=float(row.get("high", 0)),
                low=float(row.get("low",  0)),
                close=float(row.get("close", 0)),
                volume=float(row.get("volume", 0)),
                daily_return=float(row["daily_return"]) if pd.notna(row.get("daily_return")) else None,
            ))
        return result
