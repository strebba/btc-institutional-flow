"""Fixture globali della suite.

Isola i test dal DB versionato del repo: alcuni moduli (API, GEX, IFI,
PriceFetcher) aprono il DB al path di default di settings.yaml
(data/structured_notes.db, versionato in git) e senza override ogni run di
pytest crea tabelle extra nel file, sporcando il working tree.

DB_PATH va impostato a import-time, PRIMA che i moduli di test importino il
codice applicativo: get_settings() e' lru_cache-ata e legge la env alla prima
chiamata. setdefault preserva un eventuale DB_PATH esplicito del chiamante.
"""
from __future__ import annotations

import os
import tempfile

import pytest

_TEST_DB_DIR = tempfile.mkdtemp(prefix="ibit-tests-")
os.environ.setdefault("DB_PATH", os.path.join(_TEST_DB_DIR, "structured_notes.db"))

# Backend DB dei test. Default: SQLite, sempre. DATABASE_URL viene forzata a
# stringa vuota (non rimossa) cosi' load_dotenv(override=False) non puo'
# ripescarla da un .env locale: un pytest non deve MAI scrivere sul Postgres
# di produzione. Per far girare l'intera suite su un Postgres usa-e-getta:
#   BIF_TEST_BACKEND=postgres TEST_DATABASE_URL=postgresql://...@localhost/... pytest
# (accettati solo host locali; le tabelle vengono svuotate prima di ogni test).
_PG_TEST_URL = (os.getenv("TEST_DATABASE_URL") or "").strip()
_PG_SUITE = os.getenv("BIF_TEST_BACKEND", "").lower() == "postgres"


def _is_local_pg_url(url: str) -> bool:
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    return parts.scheme in ("postgres", "postgresql") and parts.hostname in (
        "localhost", "127.0.0.1", "::1", "postgres",  # "postgres" = service container CI
    )


if _PG_SUITE:
    if not _is_local_pg_url(_PG_TEST_URL):
        raise RuntimeError("BIF_TEST_BACKEND=postgres richiede TEST_DATABASE_URL su host locale")
    os.environ["DATABASE_URL"] = _PG_TEST_URL
else:
    os.environ["DATABASE_URL"] = ""


@pytest.fixture
def pg_url():
    """URL del Postgres usa-e-getta per i test del backend Postgres (skip se assente)."""
    if not _is_local_pg_url(_PG_TEST_URL):
        pytest.skip("TEST_DATABASE_URL (Postgres locale) non impostata")
    return _PG_TEST_URL


def truncate_pg_tables(url: str) -> None:
    """Svuota tutte le tabelle applicative (mai usata fuori da host locali)."""
    assert _is_local_pg_url(url)
    from src.db import ensure_postgres_schema, pg_connect

    ensure_postgres_schema(url)
    conn = pg_connect(url)
    try:
        conn.executescript(
            "TRUNCATE predictions, outcomes, weight_versions, alert_state, prices, "
            "gex_snapshots, notes, barrier_levels, barrier_snapshots, refresh_runs, "
            "macro_snapshots RESTART IDENTITY CASCADE"
        )
        conn.commit()
    finally:
        conn.close()


def pytest_collection_modifyitems(config, items):
    """Con la suite su Postgres salta i test che verificano dettagli SQLite."""
    if not _PG_SUITE:
        return
    skip = pytest.mark.skip(reason="sqlite_only: suite in esecuzione su Postgres")
    for item in items:
        if "sqlite_only" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _pg_suite_isolation():
    """Con BIF_TEST_BACKEND=postgres ogni test parte da tabelle vuote."""
    if _PG_SUITE:
        truncate_pg_tables(_PG_TEST_URL)
    yield


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Pulisce la cache lru_cache di get_settings() prima di ogni test.

    Senza questa fixture, get_settings() restituisce il valore cached della
    prima chiamata e ignorerebbe modifiche a variabili d'ambiente (es. DB_PATH).
    """
    from src.config import get_settings
    get_settings.cache_clear()
