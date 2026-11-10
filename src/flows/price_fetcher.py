"""Scarica prezzi OHLCV di BTC-USD e IBIT via yfinance.

Calcola il rapporto IBIT/BTC usato per convertire i barrier levels.
Il download è incrementale, ma le ultime ``TAIL_REFRESH_DAYS`` sessioni
già salvate vengono sempre riscaricate e sovrascritte: una barra scritta
a metà della sessione UTC non resta per sempre il "close" di quel giorno.

La sessione del giorno UTC corrente non è chiusa. Viene salvata con
``partial=1`` (lo spot al momento del fetch, utile al display) e va
sostituita al giro successivo. La verifica degli esiti usa solo le barre
con ``partial=0`` (vedi ``complete_bars`` e ``src.forecast.verifier``).

Queste barre NON sono feature del segnale dealer-flow. ``gather_dealer_flow_context``
legge lo spot e i livelli da Deribit e i flussi da Farside/CoinGlass; il
``ref_price`` salvato sulla predizione è lo spot Deribit, non questo close.
Il frame dei flussi viene unito a questi prezzi, ma nel segnale entrano solo
le colonne di flusso. Il close/high/low di questa tabella alimentano la
verifica (e dashboard/correlazioni), non la predizione.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Generator, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from src.assets import get_asset
from src.config import get_settings, setup_logging
from src.db import connect, ensure_postgres_schema, is_postgres


_log = setup_logging("flows.prices")

# Sessioni già in cache che ogni fetch riscarica e sovrascrive. Copre il giorno
# salvato mentre era ancora aperto e un paio di giri mancati; uno storico più
# vecchio si sistema con ``backfill_history`` (il bug ha lasciato barre parziali
# anche a mesi di distanza, perché il fetch partiva dal giorno successivo).
TAIL_REFRESH_DAYS = 3

_OHLC_FIELDS = ("open", "high", "low", "close")
_PRICE_TOL = 0.01

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
    partial     INTEGER NOT NULL DEFAULT 0,  -- 1 = sessione UTC non ancora chiusa
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
    def _conn(self) -> Generator[Any, None, None]:
        """Connessione al backend attivo (SQLite su ``self._path`` o Postgres)."""
        with connect(self._path) as conn:
            yield conn

    def _init_schema(self) -> None:
        if is_postgres():
            ensure_postgres_schema()
            return
        with self._conn() as conn:
            conn.executescript(_DDL_PRICES)
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(prices)")}
            if "partial" not in cols:
                conn.execute(
                    "ALTER TABLE prices ADD COLUMN partial INTEGER NOT NULL DEFAULT 0"
                )

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
        """Salva un DataFrame OHLCV nel DB (UPSERT: una barra già presente si aggiorna).

        ``partial`` (colonna del frame, default 0) marca la sessione UTC ancora aperta.
        Il rendimento della prima riga usa il close completo precedente, se c'è,
        così un refresh della coda non azzera ``daily_return``. Un rendimento
        mancante non sovrascrive quello già salvato.

        Args:
            ticker: es. "BTC-USD".
            df: DataFrame con colonne Open/High/Low/Close/Volume e DatetimeIndex.

        Returns:
            int: numero di righe scritte (inserite o aggiornate).
        """
        if df.empty:
            return 0

        now = datetime.now(timezone.utc).isoformat()
        df  = df.copy()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        if "partial" not in df.columns:
            df["partial"] = 0

        closes = pd.to_numeric(df["Close"], errors="coerce")
        first_day = _as_date(df.index[0])
        prev = self._previous_close(ticker, first_day)
        shifted = closes.shift(1)
        if prev is not None and prev > 0:
            shifted.iloc[0] = prev
        df["daily_return"] = np.log(closes / shifted)

        inserted = 0
        with self._conn() as conn:
            for idx, row in df.iterrows():
                d = _as_date(idx)
                close_v = _finite(row.get("Close"))
                if close_v is None:
                    continue
                ret = row["daily_return"]
                ret_v = float(ret) if pd.notna(ret) else None
                partial = 1 if int(row.get("partial") or 0) else 0
                try:
                    cur = conn.execute(
                        """
                        INSERT INTO prices
                            (ticker, date, open, high, low, close, volume,
                             daily_return, created_at, partial)
                        VALUES (?,?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(ticker, date) DO UPDATE SET
                            open=excluded.open, high=excluded.high, low=excluded.low,
                            close=excluded.close, volume=excluded.volume,
                            daily_return=COALESCE(excluded.daily_return, prices.daily_return),
                            created_at=excluded.created_at,
                            partial=excluded.partial
                        """,
                        (
                            ticker, d.isoformat(),
                            _finite(row.get("Open"), default=0.0),
                            _finite(row.get("High"), default=0.0),
                            _finite(row.get("Low"), default=0.0),
                            close_v,
                            _finite(row.get("Volume"), default=0.0),
                            ret_v,
                            now,
                            partial,
                        ),
                    )
                    inserted += cur.rowcount  # 1 se inserita o aggiornata
                except Exception as e:
                    _log.debug("Skip %s %s: %s", ticker, d, e)
        return inserted

    def _previous_close(self, ticker: str, day: date) -> Optional[float]:
        """Ultimo close completo strettamente precedente a ``day``."""
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT close FROM prices
                WHERE ticker=? AND date<? AND COALESCE(partial, 0)=0
                ORDER BY date DESC LIMIT 1
                """,
                (ticker, day.isoformat()),
            ).fetchone()
        if not row or row["close"] is None:
            return None
        return float(row["close"])

    def fetch(
        self,
        ticker: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        force_refresh: bool = False,
        asof: Optional[date] = None,
    ) -> pd.DataFrame:
        """Scarica i prezzi per un ticker, usando la cache locale.

        Se `force_refresh=False` non riscarica tutto lo storico, ma non si
        ferma nemmeno al giorno successivo all'ultima data salvata: rifà
        sempre le ultime ``TAIL_REFRESH_DAYS`` sessioni e le sovrascrive.
        Una cache "aggiornata a ieri" non è una scusa per tenere il close
        di ieri scritto a metà sessione.

        La barra del giorno UTC ``asof`` (default: oggi UTC) è ancora aperta:
        se il download la include, viene salvata con ``partial=1`` e non è
        un close. Un fetch vuoto — weekend, festivo, mercato non ancora
        aperto, errore di rete — non scarta i close già salvati e non ne inventa.

        Args:
            ticker: es. "BTC-USD" o "IBIT".
            start_date: data di inizio (default: 365gg fa).
            end_date: data di fine (default: ``asof``).
            force_refresh: ignora la coda e riscarica l'intera finestra richiesta.
            asof: giorno UTC "corrente" per la regola partial (i test lo fissano).

        Returns:
            pd.DataFrame: prezzi con DatetimeIndex sulla finestra richiesta,
            colonna ``partial`` inclusa.
        """
        asof = asof or datetime.now(timezone.utc).date()
        end = end_date or asof
        requested_start = start_date or end - timedelta(days=self._cfg.get("lookback_days", 365))
        download_start = requested_start

        if not force_refresh:
            last = self._last_date_in_db(ticker)
            if last is not None:
                # Inclusa l'ultima data salvata: è quella che il vecchio
                # incrementale (last+1) non rinfrescava mai.
                tail_start = last - timedelta(days=TAIL_REFRESH_DAYS - 1)
                download_start = max(requested_start, tail_start)
                _log.info(
                    "%s: refresh dal %s (ultime %d sessioni salvate + giorni nuovi)",
                    ticker, download_start, TAIL_REFRESH_DAYS,
                )

        if download_start > end:
            return self._load_from_db(ticker, requested_start, end)

        _log.info("%s: download %s → %s", ticker, download_start, end)
        try:
            raw = self._download_ohlcv(ticker, download_start, end)
        except Exception as e:
            _log.error(
                "%s: download fallito — %s. Nessun prezzo inventato; uso la cache %s → %s.",
                ticker, e, requested_start, end,
            )
            return self._load_from_db(ticker, requested_start, end)

        prepared = _prepare_downloaded(raw, asof)
        if prepared.empty:
            cached = self._load_from_db(ticker, requested_start, end)
            _log.warning(
                "%s: nessun dato scaricato per %s → %s "
                "(weekend, festivo, pre-apertura o fetch vuoto). "
                "Nessun prezzo inventato; cache %s → %s: %d righe.",
                ticker, download_start, end, requested_start, end, len(cached),
            )
            return cached

        inserted = self._store_df(ticker, prepared)
        _log.info("%s: %d righe salvate (upsert, partial escluso dal close)", ticker, inserted)

        return self._load_from_db(ticker, requested_start, end)

    def _download_ohlcv(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Barre giornaliere yfinance su [start, end] inclusi. ``end`` di yfinance è esclusivo."""
        df = yf.download(
            ticker,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            progress=False,
            auto_adjust=True,
        )
        if df is None:
            return pd.DataFrame()
        return df

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
                SELECT date, open, high, low, close, volume, daily_return, partial
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

    def _stored_rows(self, ticker: str) -> dict[date, dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT date, open, high, low, close, partial
                FROM prices WHERE ticker=? ORDER BY date
                """,
                (ticker,),
            ).fetchall()
        out: dict[date, dict[str, Any]] = {}
        for row in rows:
            out[date.fromisoformat(row["date"])] = {
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "partial": int(row["partial"] or 0),
            }
        return out

    def _tickers(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT ticker FROM prices ORDER BY ticker"
            ).fetchall()
        return [r["ticker"] for r in rows]

    def backfill_history(
        self,
        *,
        tickers: Optional[list[str]] = None,
        dry_run: bool = True,
        asof: Optional[date] = None,
    ) -> "BackfillReport":
        """Riscarica l'OHLC giornaliero completo di tutto lo storico salvato.

        Sostituisce le barre parziali (e qualunque OHLC diverso dal daily bar
        chiuso). Non scrive la sessione UTC ``asof`` come close: se il download
        la include, resta ``partial=1``. Un download vuoto non cancella nulla
        e fallisce, così un buco di rete non viene scambiato per "nessuna diff".

        ``dry_run=True`` (default) calcola e logga le diff senza scrivere.
        """
        asof = asof or datetime.now(timezone.utc).date()
        names = tickers or self._tickers()
        downloaded: dict[str, pd.DataFrame] = {}
        for ticker in names:
            stored = self._stored_rows(ticker)
            if not stored:
                continue
            start = min(stored)
            try:
                raw = self._download_ohlcv(ticker, start, asof)
            except Exception as exc:
                raise RuntimeError(f"{ticker}: download fallito, backfill annullato ({exc})") from exc
            prepared = _prepare_downloaded(raw, asof)
            if prepared.empty:
                raise RuntimeError(
                    f"{ticker}: download vuoto dal {start}, nessuna riga sostituita"
                )
            downloaded[ticker] = prepared

        revisions: list[PriceRevision] = []
        compared = 0
        for ticker, fresh in downloaded.items():
            stored = self._stored_rows(ticker)
            records = _ohlc_records(fresh)
            compared += len(records)
            for day, new in sorted(records.items()):
                old = stored.get(day)
                changes = _ohlc_changes(old, new)
                if changes:
                    revisions.append(PriceRevision(ticker, day.isoformat(), changes))
            if not dry_run:
                changed_days = {date.fromisoformat(r.day) for r in revisions if r.ticker == ticker}
                if changed_days:
                    positions = [
                        i for i, ts in enumerate(fresh.index) if _as_date(ts) in changed_days
                    ]
                    self._store_df(ticker, fresh.iloc[positions])

        for rev in revisions:
            bits = ", ".join(f"{k} {_fmt(a)} -> {_fmt(b)}" for k, (a, b) in rev.changes.items())
            _log.info("%s %s %s", rev.ticker, rev.day, bits)
        report = BackfillReport(
            revisions=tuple(revisions), compared=compared, dry_run=dry_run,
        )
        _log.info(
            "backfill %s: confrontate=%d revisioni=%d",
            "dry-run" if dry_run else "write", compared, len(revisions),
        )
        return report

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
                    "SELECT close FROM prices WHERE ticker='IBIT' AND date=? "
                    "AND COALESCE(partial, 0)=0",
                    (check,),
                ).fetchone()
                btc_row  = conn.execute(
                    "SELECT close FROM prices WHERE ticker='BTC-USD' AND date=? "
                    "AND COALESCE(partial, 0)=0",
                    (check,),
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
        asset: str = "BTC",
    ) -> pd.DataFrame:
        """Scarica spot e lead ETF dell'asset e li unisce in un DataFrame.

        Colonne (nomi dallo spec dell'asset): per BTC btc_close, btc_return,
        ibit_close, ibit_btc_ratio, btc_vol_7d; per ETH eth_close, eth_return,
        etha_close, etha_eth_ratio, eth_vol_7d.

        Args:
            start_date: inizio periodo.
            end_date: fine periodo.
            asset: "BTC" o "ETH".

        Returns:
            pd.DataFrame.
        """
        spec = get_asset(asset)
        if spec.key == "BTC":
            spot = self.fetch_btc(start_date=start_date,  end_date=end_date)
            lead = self.fetch_ibit(start_date=start_date, end_date=end_date)
        else:
            spot = self.fetch(spec.spot_ticker, start_date=start_date, end_date=end_date)
            lead = self.fetch(spec.lead_etf, start_date=start_date, end_date=end_date)

        if spot.empty and lead.empty:
            return pd.DataFrame()

        merged = pd.DataFrame(index=spot.index if not spot.empty else lead.index)
        lead_close = f"{spec.lead_prefix}_close"

        if not spot.empty:
            merged[spec.price_col]  = spot["close"]
            merged[spec.return_col] = spot["daily_return"]

        if not lead.empty:
            merged[lead_close] = lead["close"]

        if spec.price_col in merged and lead_close in merged:
            merged[f"{spec.lead_prefix}_{spec.prefix}_ratio"] = merged[lead_close] / merged[spec.price_col]

        # Volatilità realizzata 7 giorni (annualizzata — le crypto tradano 365 giorni/anno)
        if spec.return_col in merged:
            merged[spec.vol_col] = (
                merged[spec.return_col]
                .rolling(7, min_periods=4)
                .std() * (365 ** 0.5)
            )

        return merged.dropna(how="all")


# ─── Barre complete e backfill ───────────────────────────────────────────────


def complete_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Toglie le sessioni UTC non chiuse (``partial=1``).

    Senza la colonna il frame è trattato come già definitivo: i test e i
    chiamanti storici non la passano. La verifica deve usare questo frame
    (close di fine orizzonte e high/low della finestra), non la barra del
    giorno ancora aperto.
    """
    if df is None or df.empty or "partial" not in df.columns:
        return df
    flag = pd.to_numeric(df["partial"], errors="coerce").fillna(0)
    return df.loc[flag.eq(0)]


@dataclass(frozen=True)
class PriceRevision:
    """Una data il cui OHLC (o il flag partial) non coincide con la barra scaricata."""

    ticker: str
    day: str
    changes: dict[str, tuple[Any, Any]]


@dataclass(frozen=True)
class BackfillReport:
    revisions: tuple[PriceRevision, ...]
    compared: int
    dry_run: bool

    def format_lines(self) -> list[str]:
        lines = []
        for rev in self.revisions:
            bits = ", ".join(
                f"{name} {_fmt(old)} -> {_fmt(new)}"
                for name, (old, new) in rev.changes.items()
            )
            lines.append(f"{rev.ticker} {rev.day}: {bits}")
        lines.append(
            f"summary: compared={self.compared} revisions={len(self.revisions)} "
            f"written={not self.dry_run}"
        )
        return lines


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.date()


def _finite(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return number


def _prepare_downloaded(df: pd.DataFrame, asof: date) -> pd.DataFrame:
    """Normalizza il frame yfinance e marca ``partial`` il giorno UTC ``asof``.

    Le date successive ad ``asof`` non vengono salvate. ``asof`` stesso non è
    un close: la sessione chiude a 00:00 UTC del giorno dopo.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] for c in out.columns]
    else:
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out.loc[out.index.normalize() <= pd.Timestamp(asof)]
    if out.empty:
        return out
    out["partial"] = [1 if _as_date(ts) >= asof else 0 for ts in out.index]
    return out


def _ohlc_records(df: pd.DataFrame) -> dict[date, dict[str, Any]]:
    records: dict[date, dict[str, Any]] = {}
    for idx, row in df.iterrows():
        records[_as_date(idx)] = {
            "open": _finite(row.get("Open")),
            "high": _finite(row.get("High")),
            "low": _finite(row.get("Low")),
            "close": _finite(row.get("Close")),
            "partial": 1 if int(row.get("partial") or 0) else 0,
        }
    return records


def _ohlc_changes(
    old: Optional[dict[str, Any]], new: dict[str, Any],
) -> dict[str, tuple[Any, Any]]:
    changes: dict[str, tuple[Any, Any]] = {}
    if old is None:
        for name in (*_OHLC_FIELDS, "partial"):
            changes[name] = (None, new.get(name))
        return changes
    for name in _OHLC_FIELDS:
        if _price_differs(old.get(name), new.get(name)):
            changes[name] = (old.get(name), new.get(name))
    if int(old.get("partial") or 0) != int(new.get("partial") or 0):
        changes["partial"] = (int(old.get("partial") or 0), int(new.get("partial") or 0))
    return changes


def _price_differs(old: Any, new: Any) -> bool:
    if old is None and new is None:
        return False
    if old is None or new is None:
        return True
    return abs(float(old) - float(new)) > _PRICE_TOL


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)

