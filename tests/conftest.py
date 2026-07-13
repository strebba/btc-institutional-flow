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


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Pulisce la cache lru_cache di get_settings() prima di ogni test.

    Senza questa fixture, get_settings() restituisce il valore cached della
    prima chiamata e ignorerebbe modifiche a variabili d'ambiente (es. DB_PATH).
    """
    from src.config import get_settings
    get_settings.cache_clear()
