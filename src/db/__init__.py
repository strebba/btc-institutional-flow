"""Accesso al database unificato: SQLite (default) o Postgres (``DATABASE_URL``).

Scelta del backend
------------------
* ``DATABASE_URL`` vuota o assente → SQLite sul path passato dal chiamante
  (comportamento storico: sviluppo locale, test, fallback).
* ``DATABASE_URL=postgres://...`` o ``postgresql://...`` → Postgres
  (produzione: Supabase). Tutte le classi DB scrivono nello stesso database.
* Qualsiasi altro schema → ``ValueError``: in produzione meglio fallire
  rumorosamente che scrivere in silenzio su un SQLite effimero.

Compatibilita'
--------------
Le classi DB del progetto sono scritte in dialetto SQLite (``?``,
``INSERT OR IGNORE``, ``PRAGMA``, ``sqlite3.Row``). Invece di riscriverle,
:class:`PgConnection` espone la stessa API minima di ``sqlite3.Connection``
e traduce al volo le poche differenze:

* placeholder ``?`` → ``%s`` (e ``%`` letterale → ``%%``);
* ``INSERT OR IGNORE INTO ...`` → ``INSERT INTO ... ON CONFLICT DO NOTHING``;
* ``PRAGMA table_info(t)`` → ``information_schema.columns``; altri PRAGMA → no-op;
* righe accessibili sia per indice sia per nome (come ``sqlite3.Row``);
* ``numeric`` letto come ``float``; ``bool``/numpy/date adattati come farebbe SQLite.

Le funzioni SQL specifiche di SQLite (``date('now', ...)``) e
``cursor.lastrowid`` NON sono tradotte: i chiamanti usano date calcolate in
Python e ``RETURNING id``, che funzionano su entrambi i backend.
"""
from __future__ import annotations

import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence
from urllib.parse import urlsplit

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

_PG_SCHEMES = ("postgres", "postgresql")


# ─── Selezione backend ───────────────────────────────────────────────────────


def get_database_url() -> str | None:
    """URL Postgres da ``DATABASE_URL`` (carica ``.env`` se presente), o None.

    Raises:
        ValueError: se ``DATABASE_URL`` e' valorizzata con uno schema non Postgres.
    """
    from src.config import get_settings

    get_settings()  # garantisce load_dotenv(.env) come per le altre env var
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        return None
    scheme = urlsplit(url).scheme.lower()
    if scheme not in _PG_SCHEMES:
        raise ValueError(
            f"DATABASE_URL con schema non supportato '{scheme}': "
            "usa postgres:// o postgresql://, oppure lasciala vuota per SQLite"
        )
    return url


def is_postgres() -> bool:
    """True se il backend attivo e' Postgres."""
    return get_database_url() is not None


def backend_name() -> str:
    """'postgres' o 'sqlite' (per health check e log; non espone l'URL)."""
    return "postgres" if is_postgres() else "sqlite"


def redact_url(url: str) -> str:
    """URL senza password, per i log."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    user = f"{parts.username}@" if parts.username else ""
    return f"{parts.scheme}://{user}{host}{port}{parts.path}"


def require_postgres(context: str) -> str:
    """Restituisce l'URL Postgres o termina: per i job schedulati di produzione.

    Un job di produzione che ricade su SQLite scriverebbe su un file effimero
    del runner e i dati andrebbero persi senza errori visibili.
    """
    url = get_database_url()
    if url is None:
        raise SystemExit(
            f"[{context}] DATABASE_URL non impostata: questo job scrive solo su "
            "Postgres (produzione). Per un run locale su SQLite usa gli script singoli."
        )
    return url


def days_ago_iso(days: int) -> str:
    """Data UTC di ``days`` giorni fa in formato ``YYYY-MM-DD``.

    Sostituisce ``date('now', '-N days')`` di SQLite, che non esiste in Postgres.
    """
    return (datetime.now(timezone.utc).date() - timedelta(days=int(days))).isoformat()


# ─── Connessione ─────────────────────────────────────────────────────────────


@contextmanager
def connect(
    sqlite_path: str | Path,
    *,
    sqlite_pragmas: Sequence[str] = ("journal_mode=WAL",),
    sqlite_timeout: float = 5.0,
) -> Iterator[Any]:
    """Connessione al backend attivo, con commit a fine blocco e rollback su errore.

    Args:
        sqlite_path: file SQLite da usare quando ``DATABASE_URL`` non e' impostata.
        sqlite_pragmas: PRAGMA applicati alla connessione SQLite.
        sqlite_timeout: timeout di lock SQLite in secondi.

    Yields:
        ``sqlite3.Connection`` (row_factory ``sqlite3.Row``) oppure
        :class:`PgConnection` con la stessa API minima.
    """
    url = get_database_url()
    if url is not None:
        conn: Any = pg_connect(url)
    else:
        conn = sqlite3.connect(str(sqlite_path), timeout=sqlite_timeout)
        conn.row_factory = sqlite3.Row
        for pragma in sqlite_pragmas:
            conn.execute(f"PRAGMA {pragma}")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def pg_connect(url: str) -> "PgConnection":
    """Apre una connessione psycopg con gli adattatori di compatibilita'."""
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - dipende dall'ambiente
        raise RuntimeError(
            "DATABASE_URL punta a Postgres ma psycopg non e' installato: "
            "pip install 'psycopg[binary]'"
        ) from exc

    # prepare_threshold=None: niente prepared statement lato server, necessario
    # con il transaction pooler di Supabase (pgbouncer, porta 6543).
    raw = psycopg.connect(url, prepare_threshold=None, connect_timeout=15)
    _register_adapters(raw)
    return PgConnection(raw)


# ─── Adattatori psycopg ──────────────────────────────────────────────────────


def _register_adapters(raw: Any) -> None:
    from psycopg.adapt import Dumper
    from psycopg.types.numeric import FloatLoader

    class _BoolAsIntDumper(Dumper):
        """SQLite salva i bool come 0/1 nelle colonne INTEGER: idem qui."""

        oid = 20  # int8

        def dump(self, obj: Any) -> bytes:
            return b"1" if obj else b"0"

    class _IsoTextDumper(Dumper):
        """date/datetime come testo ISO, come l'adapter di default di sqlite3.

        OID 0 (unknown): Postgres lo confronta liberamente con colonne TEXT.
        """

        oid = 0

        def dump(self, obj: Any) -> bytes:
            if isinstance(obj, datetime):
                return obj.isoformat(" ").encode()
            return obj.isoformat().encode()

    class _NumpyIntDumper(Dumper):
        oid = 20  # int8

        def dump(self, obj: Any) -> bytes:
            return str(int(obj)).encode()

    class _NumpyFloatDumper(Dumper):
        oid = 701  # float8

        def dump(self, obj: Any) -> bytes:
            return repr(float(obj)).encode()

    adapters = raw.adapters
    adapters.register_dumper(bool, _BoolAsIntDumper)
    adapters.register_dumper(date, _IsoTextDumper)
    adapters.register_dumper(datetime, _IsoTextDumper)
    # SUM/AVG su INTEGER restituiscono numeric → Decimal: SQLite darebbe float.
    adapters.register_loader("numeric", FloatLoader)
    try:
        import numpy as np

        adapters.register_dumper(np.bool_, _BoolAsIntDumper)
        adapters.register_dumper(np.integer, _NumpyIntDumper)
        adapters.register_dumper(np.floating, _NumpyFloatDumper)
    except ImportError:  # pragma: no cover
        pass


# ─── Righe e cursori compatibili sqlite3 ─────────────────────────────────────


class PgRow(tuple):
    """Tupla accessibile anche per nome colonna, come ``sqlite3.Row``."""

    _index: dict[str, int]

    def __new__(cls, values: Sequence[Any], index: dict[str, int]) -> "PgRow":
        obj = super().__new__(cls, values)
        obj._index = index
        return obj

    def __getitem__(self, key: Any) -> Any:  # type: ignore[override]
        if isinstance(key, str):
            try:
                return tuple.__getitem__(self, self._index[key])
            except KeyError:
                raise IndexError(f"No item with that key: {key!r}") from None
        return tuple.__getitem__(self, key)

    def keys(self) -> list[str]:
        return list(self._index)


def _row_factory(cursor: Any) -> Any:
    desc = cursor.description
    if desc is None:
        return tuple
    index = {col.name: i for i, col in enumerate(desc)}
    return lambda values: PgRow(values, index)


class PgCursor:
    """Cursore con l'API usata nel progetto (fetchone/fetchall/rowcount/iter)."""

    def __init__(self, raw: Any | None) -> None:
        self._raw = raw

    @property
    def rowcount(self) -> int:
        return -1 if self._raw is None else self._raw.rowcount

    @property
    def description(self) -> Any:
        return None if self._raw is None else self._raw.description

    @property
    def lastrowid(self) -> Any:
        raise AttributeError(
            "lastrowid non e' disponibile su Postgres: usa 'RETURNING id' e fetchone()"
        )

    def fetchone(self) -> Any:
        if self._raw is None or self._raw.description is None:
            return None
        return self._raw.fetchone()

    def fetchall(self) -> list[Any]:
        if self._raw is None or self._raw.description is None:
            return []
        return self._raw.fetchall()

    def __iter__(self) -> Iterator[Any]:
        return iter(self.fetchall())


class PgConnection:
    """Wrapper psycopg con l'API minima di ``sqlite3.Connection`` usata nel progetto."""

    def __init__(self, raw: Any) -> None:
        self._raw = raw
        raw.row_factory = _row_factory

    # sqlite3 compat: i chiamanti a volte assegnano row_factory; qui e' fissa.
    @property
    def row_factory(self) -> Any:
        return _row_factory

    @row_factory.setter
    def row_factory(self, _value: Any) -> None:
        pass

    @property
    def raw(self) -> Any:
        """Connessione psycopg sottostante (per usi specifici Postgres)."""
        return self._raw

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> PgCursor:
        pragma = _PRAGMA_RE.match(sql)
        if pragma:
            return self._pragma(pragma.group(1).lower(), pragma.group(2))
        cur = self._raw.cursor()
        cur.execute(translate_sql(sql), tuple(params or ()))
        return PgCursor(cur)

    def executemany(self, sql: str, seq: Any) -> PgCursor:
        cur = self._raw.cursor()
        cur.executemany(translate_sql(sql), [tuple(p) for p in seq])
        return PgCursor(cur)

    def executescript(self, script: str) -> PgCursor:
        """Esegue SQL nativo Postgres (piu' statement, nessun parametro)."""
        cur = self._raw.cursor()
        cur.execute(script)
        return PgCursor(cur)

    def commit(self) -> None:
        self._raw.commit()

    def rollback(self) -> None:
        self._raw.rollback()

    def close(self) -> None:
        self._raw.close()

    def __enter__(self) -> "PgConnection":
        return self

    def __exit__(self, exc_type: Any, *_: Any) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.rollback()

    def _pragma(self, name: str, arg: str | None) -> PgCursor:
        if name == "table_info" and arg:
            cur = self._raw.cursor()
            cur.execute(
                "SELECT ordinal_position - 1 AS cid, column_name AS name, "
                "data_type AS type FROM information_schema.columns "
                "WHERE table_schema = current_schema() AND table_name = %s "
                "ORDER BY ordinal_position",
                (arg.strip().strip("'\""),),
            )
            return PgCursor(cur)
        # journal_mode, foreign_keys, busy_timeout, wal_checkpoint, user_version...:
        # concetti SQLite senza equivalente da applicare su Postgres.
        return PgCursor(None)


_PRAGMA_RE = re.compile(r"^\s*PRAGMA\s+(\w+)\s*(?:\(\s*([^)]*)\)|=.*)?\s*;?\s*$", re.I | re.S)
_INSERT_OR_IGNORE_RE = re.compile(r"\bINSERT\s+OR\s+IGNORE\s+INTO\b", re.I)
_INSERT_OR_REPLACE_RE = re.compile(r"\bINSERT\s+OR\s+REPLACE\b", re.I)
_RETURNING_RE = re.compile(r"\bRETURNING\b", re.I)


def translate_sql(sql: str) -> str:
    """Traduce una query in dialetto SQLite nel dialetto psycopg/Postgres.

    Gestisce placeholder, ``%`` letterali e ``INSERT OR IGNORE``; le stringhe
    tra apici non vengono toccate.
    """
    if _INSERT_OR_REPLACE_RE.search(sql):
        raise ValueError("INSERT OR REPLACE non supportato: usa ON CONFLICT ... DO UPDATE")

    out: list[str] = []
    quote: str | None = None
    in_comment = False
    for i, ch in enumerate(sql):
        if in_comment:
            # i commenti "-- ..." possono contenere apostrofi (italiano: "e'")
            out.append("%%" if ch == "%" else ch)
            if ch == "\n":
                in_comment = False
            continue
        if quote:
            out.append("%%" if ch == "%" else ch)
            if ch == quote:
                quote = None
            continue
        if ch == "-" and sql[i + 1:i + 2] == "-":
            in_comment = True
            out.append(ch)
        elif ch in ("'", '"'):
            quote = ch
            out.append(ch)
        elif ch == "?":
            out.append("%s")
        elif ch == "%":
            out.append("%%")
        else:
            out.append(ch)
    text = "".join(out)

    if _INSERT_OR_IGNORE_RE.search(text):
        text = _INSERT_OR_IGNORE_RE.sub("INSERT INTO", text, count=1).rstrip().rstrip(";")
        m = _RETURNING_RE.search(text)
        if m:
            text = f"{text[:m.start()].rstrip()} ON CONFLICT DO NOTHING {text[m.start():]}"
        else:
            text = f"{text} ON CONFLICT DO NOTHING"
    return text


# ─── Schema Postgres ─────────────────────────────────────────────────────────

_schema_lock = threading.Lock()
_schema_ready: set[str] = set()
_ADVISORY_LOCK_KEY = 0x6266_6970  # 'bfip': serializza le migrazioni tra processi


def migration_files() -> list[Path]:
    """File SQL di migrazione Postgres, in ordine di versione."""
    return sorted(MIGRATIONS_DIR.glob("*.sql"))


def ensure_postgres_schema(url: str | None = None) -> list[str]:
    """Applica le migrazioni Postgres mancanti (una volta per processo e URL).

    Idempotente e sicuro con piu' processi concorrenti (advisory lock).

    Returns:
        list[str]: versioni applicate in questa chiamata (vuota se gia' allineato).
    """
    url = url or get_database_url()
    if url is None:
        raise RuntimeError("ensure_postgres_schema chiamata senza DATABASE_URL")
    with _schema_lock:
        if url in _schema_ready:
            return []
        applied_now: list[str] = []
        conn = pg_connect(url)
        try:
            raw = conn.raw
            with raw.transaction():
                raw.execute("SELECT pg_advisory_xact_lock(%s)", (_ADVISORY_LOCK_KEY,))
                raw.execute(
                    "CREATE TABLE IF NOT EXISTS schema_migrations ("
                    " version TEXT PRIMARY KEY,"
                    " applied_at TEXT NOT NULL)"
                )
                done = {r[0] for r in raw.execute("SELECT version FROM schema_migrations")}
                for path in migration_files():
                    version = path.stem
                    if version in done:
                        continue
                    raw.execute(path.read_text(encoding="utf-8"))
                    raw.execute(
                        "INSERT INTO schema_migrations (version, applied_at) VALUES (%s, %s)",
                        (version, datetime.now(timezone.utc).isoformat(timespec="seconds")),
                    )
                    applied_now.append(version)
        finally:
            conn.close()
        _schema_ready.add(url)
        return applied_now


def reset_schema_cache() -> None:
    """Dimentica gli schema gia' verificati (solo test)."""
    with _schema_lock:
        _schema_ready.clear()
