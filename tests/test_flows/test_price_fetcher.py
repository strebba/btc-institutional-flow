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
