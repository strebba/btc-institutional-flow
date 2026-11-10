"""Test unitari per PriceFetcher."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
from src.flows.price_fetcher import PriceFetcher, complete_bars


@pytest.fixture
def fetcher(tmp_path: Path) -> PriceFetcher:
    return PriceFetcher(db_path=tmp_path / "test.db")


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """DataFrame OHLCV fittizio per 5 giorni."""
    idx = pd.date_range("2024-10-01", periods=5, freq="D")
    df  = pd.DataFrame({
        "Open":   [60_000, 61_000, 59_500, 62_000, 63_000],
        "High":   [61_500, 62_000, 60_500, 63_500, 64_000],
        "Low":    [59_000, 60_500, 58_000, 61_000, 62_000],
        "Close":  [61_000, 59_500, 62_000, 63_000, 62_500],
        "Volume": [1e9, 9e8, 1.1e9, 8e8, 1.2e9],
    }, index=idx)
    return df


class TestStoreDf:
    def test_stores_rows(self, fetcher, sample_df):
        n = fetcher._store_df("BTC-USD", sample_df)
        assert n == 5

    def test_idempotent(self, fetcher, sample_df):
        fetcher._store_df("BTC-USD", sample_df)
        n2 = fetcher._store_df("BTC-USD", sample_df)
        assert n2 == len(sample_df)  # UPSERT: ogni riga esistente viene aggiornata

    def test_daily_return_computed(self, fetcher, sample_df):
        fetcher._store_df("BTC-USD", sample_df)
        df = fetcher._load_from_db("BTC-USD", date(2024, 10, 1), date(2024, 10, 5))
        # Prima riga ha return NaN, le altre no
        assert df["daily_return"].iloc[1:].notna().all()


class TestLoadFromDb:
    def test_empty_db(self, fetcher):
        df = fetcher._load_from_db("BTC-USD", date(2024, 1, 1), date(2024, 1, 31))
        assert df.empty

    def test_date_filter(self, fetcher, sample_df):
        fetcher._store_df("BTC-USD", sample_df)
        df = fetcher._load_from_db("BTC-USD", date(2024, 10, 2), date(2024, 10, 3))
        assert len(df) == 2

    def test_columns_present(self, fetcher, sample_df):
        fetcher._store_df("BTC-USD", sample_df)
        df = fetcher._load_from_db("BTC-USD", date(2024, 10, 1), date(2024, 10, 5))
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in df.columns


class TestIbitBtcRatio:
    @pytest.mark.sqlite_only
    def test_ratio_calculation(self, fetcher):
        # Simula BTC=60000 e IBIT=60 → ratio=0.001
        import sqlite3
        from datetime import datetime, timezone
        with sqlite3.connect(fetcher._path) as conn:
            now = datetime.now(timezone.utc).isoformat()
            d   = date.today().isoformat()
            conn.execute(
                "INSERT INTO prices (ticker,date,close,open,high,low,volume,created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                ("BTC-USD", d, 60_000, 0, 0, 0, 0, now),
            )
            conn.execute(
                "INSERT INTO prices (ticker,date,close,open,high,low,volume,created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                ("IBIT", d, 60, 0, 0, 0, 0, now),
            )
            conn.commit()
        ratio = fetcher.get_ibit_btc_ratio(target_date=date.today())
        assert ratio == pytest.approx(0.001, rel=1e-3)

    def test_ratio_missing(self, fetcher):
        ratio = fetcher.get_ibit_btc_ratio(target_date=date(2020, 1, 1))
        assert ratio is None


class TestGetAllPricesAsset:
    @pytest.fixture
    def offline(self, fetcher, sample_df, monkeypatch):
        """Prezzi già in DB e nessun download: fetch legge solo la cache locale."""
        for ticker, scala in (("BTC-USD", 1.0), ("IBIT", 0.0006), ("ETH-USD", 0.04), ("ETHA", 0.008)):
            fetcher._store_df(ticker, sample_df * scala)
        monkeypatch.setattr(
            fetcher, "fetch",
            lambda t, start_date=None, end_date=None, **k: fetcher._load_from_db(
                t, date(2024, 10, 1), date(2024, 10, 5)
            ),
        )
        return fetcher

    def test_btc_conserva_le_colonne_storiche(self, offline):
        df = offline.get_all_prices()
        assert {"btc_close", "btc_return", "ibit_close", "ibit_btc_ratio", "btc_vol_7d"} <= set(df.columns)
        assert not any(c.startswith("eth") for c in df.columns)

    def test_eth_usa_eth_usd_ed_etha(self, offline):
        df = offline.get_all_prices(asset="ETH")
        assert {"eth_close", "eth_return", "etha_close", "etha_eth_ratio", "eth_vol_7d"} <= set(df.columns)
        assert df["eth_close"].iloc[0] == pytest.approx(61_000 * 0.04)
        assert not any(c.startswith("btc") for c in df.columns)


def _ohlcv(dates: list[str], closes: list[float]) -> pd.DataFrame:
    idx = pd.to_datetime(dates)
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": [1_000_000.0] * len(closes),
        },
        index=idx,
    )


class TestMissingIbitClose:
    """Lunedì 2026-11-09, job del mattino: IBIT non ha ancora una barra nuova.

    BTC tratta nel weekend, quindi la cache spot è aggiornata fino a domenica.
    IBIT si ferma al close di venerdì 6 novembre; il download incrementale
    sabato→lunedì (mercato USA chiuso) torna vuoto. Quel vuoto non deve
    cancellare ``ibit_close`` né scrivere un close inventato.
    """

    def test_empty_incremental_fetch_keeps_cached_ibit_close(self, fetcher, monkeypatch):
        fetcher._store_df("IBIT", _ohlcv(["2026-11-05", "2026-11-06"], [53.5, 54.25]))
        fetcher._store_df(
            "BTC-USD",
            _ohlcv(
                ["2026-11-05", "2026-11-06", "2026-11-07", "2026-11-08"],
                [100_000, 101_000, 102_000, 103_000],
            ),
        )
        calls: list[tuple] = []

        def empty_download(ticker, **kwargs):
            calls.append((ticker, kwargs.get("start"), kwargs.get("end")))
            return pd.DataFrame()

        monkeypatch.setattr("src.flows.price_fetcher.yf.download", empty_download)

        monday = date(2026, 11, 9)
        prices = fetcher.get_all_prices(start_date=date(2026, 11, 1), end_date=monday)

        ibit_calls = [c for c in calls if c[0] == "IBIT"]
        # La coda include l'ultima sessione salvata (6 nov), non parte dal giorno dopo:
        # un close scritto a metà sessione va riscaricato. 3 giorni → start 4 nov.
        assert ibit_calls == [("IBIT", "2026-11-04", "2026-11-10")]
        assert "ibit_close" in prices.columns
        observed = prices["ibit_close"].dropna()
        assert observed.iloc[-1] == pytest.approx(54.25)
        assert observed.index.max().date() == date(2026, 11, 6)
        # Il close IBIT del weekend non viene forward-fillato sul calendario BTC.
        assert pd.isna(prices.loc[pd.Timestamp("2026-11-08"), "ibit_close"])
        assert prices.loc[pd.Timestamp("2026-11-08"), "btc_close"] == pytest.approx(103_000)
        stored = fetcher._load_from_db("IBIT", date(2026, 11, 1), monday)
        assert [ts.date() for ts in stored.index] == [date(2026, 11, 5), date(2026, 11, 6)]

    def test_download_error_still_returns_cached_close(self, fetcher, monkeypatch):
        fetcher._store_df("IBIT", _ohlcv(["2026-11-06"], [54.25]))

        def boom(*_args, **_kwargs):
            raise RuntimeError("yfinance down")

        monkeypatch.setattr("src.flows.price_fetcher.yf.download", boom)
        loaded = fetcher.fetch(
            "IBIT", start_date=date(2026, 11, 1), end_date=date(2026, 11, 9)
        )
        assert list(loaded.index.date) == [date(2026, 11, 6)]
        assert loaded["close"].iloc[-1] == pytest.approx(54.25)

    def test_no_history_and_empty_fetch_does_not_fabricate_close(self, fetcher, monkeypatch):
        fetcher._store_df("BTC-USD", _ohlcv(["2026-11-08"], [103_000]))
        monkeypatch.setattr(
            "src.flows.price_fetcher.yf.download",
            lambda *_a, **_k: pd.DataFrame(),
        )
        prices = fetcher.get_all_prices(
            start_date=date(2026, 11, 1), end_date=date(2026, 11, 9)
        )
        assert "btc_close" in prices.columns
        assert "ibit_close" not in prices.columns
        assert "ibit_btc_ratio" not in prices.columns
        assert fetcher._load_from_db("IBIT", date(2026, 11, 1), date(2026, 11, 9)).empty


def _ohlcv_full(
    dates: list[str],
    closes: list[float],
    *,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> pd.DataFrame:
    idx = pd.to_datetime(dates)
    return pd.DataFrame(
        {
            "Open": closes,
            "High": highs or closes,
            "Low": lows or closes,
            "Close": closes,
            "Volume": [1_000_000.0] * len(closes),
        },
        index=idx,
    )


class TestPartialDailyBar:
    """Regressione: il close di una sessione ancora aperta restava per sempre.

    Il fetch partiva dal giorno successivo all'ultima data salvata, e se la
    cache arrivava a ieri non scaricava nulla. 121 barre BTC dal 6 giugno sono
    rimaste lo spot del momento della verifica (es. 21/9 a 81_695 invece del
    close 86_620).
    """

    def test_fresh_cache_overwrites_tail_and_does_not_keep_today_as_close(
        self, fetcher, monkeypatch,
    ):
        asof = date(2026, 11, 10)
        fetcher._store_df(
            "BTC-USD",
            _ohlcv_full(
                ["2026-11-07", "2026-11-08", "2026-11-09"],
                [80_000, 81_000, 81_695],
                highs=[80_100, 81_100, 81_800],
                lows=[79_900, 80_900, 81_500],
            ),
        )
        calls: list[str] = []

        def download(ticker, **kwargs):
            calls.append(kwargs["start"])
            return _ohlcv_full(
                ["2026-11-07", "2026-11-08", "2026-11-09", "2026-11-10"],
                [80_000, 81_000, 86_620, 87_000],
                highs=[80_500, 82_000, 87_100, 87_200],
                lows=[79_500, 80_500, 81_000, 86_500],
            )

        monkeypatch.setattr("src.flows.price_fetcher.yf.download", download)
        loaded = fetcher.fetch(
            "BTC-USD", start_date=date(2026, 11, 1), end_date=asof, asof=asof,
        )

        # last=9 nov e end=10 nov: prima il fetch usciva senza chiamare yfinance.
        assert calls == ["2026-11-07"]
        stored = fetcher._load_from_db("BTC-USD", date(2026, 11, 7), asof)
        nov9 = stored.loc[pd.Timestamp("2026-11-09")]
        assert nov9["close"] == pytest.approx(86_620)
        assert nov9["high"] == pytest.approx(87_100)
        assert nov9["low"] == pytest.approx(81_000)
        assert int(nov9["partial"]) == 0
        assert int(stored.loc[pd.Timestamp("2026-11-10"), "partial"]) == 1
        complete = complete_bars(loaded)
        assert date(2026, 11, 10) not in [ts.date() for ts in complete.index]
        assert complete.loc[pd.Timestamp("2026-11-09"), "close"] == pytest.approx(86_620)

    def test_backfill_replaces_old_partial_rows_and_dry_run_does_not_write(
        self, fetcher, monkeypatch,
    ):
        asof = date(2026, 11, 10)
        fetcher._store_df(
            "BTC-USD",
            _ohlcv_full(
                ["2026-09-21", "2026-11-09"],
                [81_695, 90_000],
                highs=[82_000, 90_100],
                lows=[81_000, 89_000],
            ),
        )
        full = _ohlcv_full(
            ["2026-09-21", "2026-11-09", "2026-11-10"],
            [86_620, 91_000, 91_500],
            highs=[87_200, 92_000, 91_800],
            lows=[81_200, 89_500, 91_000],
        )

        def download(ticker, **kwargs):
            start = pd.Timestamp(kwargs["start"])
            return full.loc[full.index >= start]

        monkeypatch.setattr("src.flows.price_fetcher.yf.download", download)

        preview = fetcher.backfill_history(dry_run=True, asof=asof)
        assert any(r.day == "2026-09-21" and r.changes["close"][0] == pytest.approx(81_695)
                   for r in preview.revisions)
        assert preview.dry_run is True
        still = fetcher._load_from_db("BTC-USD", date(2026, 9, 21), date(2026, 9, 21))
        assert still["close"].iloc[0] == pytest.approx(81_695)

        # Il fetch quotidiano non torna indietro fino al 21 settembre: l'ultima
        # data salvata è il 9 novembre, la coda è di 3 giorni.
        fetcher.fetch("BTC-USD", start_date=date(2026, 9, 1), end_date=asof, asof=asof)
        after_fetch = fetcher._load_from_db("BTC-USD", date(2026, 9, 21), date(2026, 9, 21))
        assert after_fetch["close"].iloc[0] == pytest.approx(81_695)

        written = fetcher.backfill_history(dry_run=False, asof=asof)
        assert any(r.day == "2026-09-21" for r in written.revisions)
        fixed = fetcher._load_from_db("BTC-USD", date(2026, 9, 1), asof)
        assert fixed.loc[pd.Timestamp("2026-09-21"), "close"] == pytest.approx(86_620)
        assert fixed.loc[pd.Timestamp("2026-09-21"), "high"] == pytest.approx(87_200)
        assert int(fixed.loc[pd.Timestamp("2026-11-10"), "partial"]) == 1
        assert date(2026, 11, 10) not in [
            ts.date() for ts in complete_bars(fixed).index
        ]

        again = fetcher.backfill_history(dry_run=False, asof=asof)
        assert again.revisions == ()

    @pytest.mark.sqlite_only
    def test_ratio_skips_unclosed_session(self, fetcher):
        import sqlite3
        from datetime import datetime, timezone

        today = date.today()
        yesterday = today - timedelta(days=1)
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(fetcher._path) as conn:
            conn.execute(
                "INSERT INTO prices (ticker, date, close, partial, created_at) "
                "VALUES (?,?,?,?,?)",
                ("BTC-USD", yesterday.isoformat(), 60_000, 0, now),
            )
            conn.execute(
                "INSERT INTO prices (ticker, date, close, partial, created_at) "
                "VALUES (?,?,?,?,?)",
                ("IBIT", yesterday.isoformat(), 60, 0, now),
            )
            # Sessione aperta: BTC e IBIT dello stesso istante, ma non un close.
            conn.execute(
                "INSERT INTO prices (ticker, date, close, partial, created_at) "
                "VALUES (?,?,?,?,?)",
                ("BTC-USD", today.isoformat(), 10_000, 1, now),
            )
            conn.execute(
                "INSERT INTO prices (ticker, date, close, partial, created_at) "
                "VALUES (?,?,?,?,?)",
                ("IBIT", today.isoformat(), 50, 1, now),
            )
        ratio = fetcher.get_ibit_btc_ratio(target_date=today)
        assert ratio == pytest.approx(0.001, rel=1e-3)
