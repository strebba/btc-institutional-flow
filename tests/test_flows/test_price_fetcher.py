"""Test unitari per PriceFetcher."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from src.flows.price_fetcher import PriceFetcher


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
        assert ibit_calls == [("IBIT", "2026-11-07", "2026-11-10")]
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
