"""Migrazione una tantum dei dati SQLite (Mac mini / repo) verso Postgres.

Copia nel Postgres di ``--database-url`` (default: env ``DATABASE_URL``):

  --runtime-db  (es. data/runtime.db del Mac mini)
      predictions, outcomes, weight_versions, alert_state, prices
  --notes-db    (data/structured_notes.db, il DB versionato)
      notes, barrier_levels, barrier_snapshots, refresh_runs, macro_snapshots
  --gex-db      (default = --notes-db)
      gex_snapshots

Garanzie:
  * **Idempotente**: ``INSERT ... ON CONFLICT DO NOTHING`` su chiave primaria o
    chiave naturale; rilanciarla non duplica nulla e non sovrascrive righe
    gia' presenti in Postgres.
  * **Id preservati** per le tabelle referenziate (predictions, outcomes,
    weight_versions, notes, barrier_levels, refresh_runs): le FK e i riferimenti
    (``outcomes.prediction_id``, ``barrier_levels.note_id``,
    ``predictions.weights_version``) restano validi. Le sequenze IDENTITY vengono
    riallineate al max(id) a fine run.
  * **Tutto o niente**: un'unica transazione; un errore annulla l'intera copia.
  * **Righe figlie protette**: se in Postgres esiste gia' una riga padre con lo
    stesso id ma contenuto diverso (es. il job ha scritto prima della migrazione),
    le righe figlie che la referenziano vengono saltate e segnalate, invece di
    agganciarsi al padre sbagliato.
  * Le sorgenti SQLite sono aperte in **sola lettura**.

Uso:
    # anteprima: conta le righe, nessuna scrittura (Postgres solo letto, se dato)
    python3 scripts/migrate_sqlite_to_postgres.py --runtime-db data/runtime.db --dry-run

    # esecuzione reale (richiede --yes)
    DATABASE_URL=postgresql://... python3 scripts/migrate_sqlite_to_postgres.py \\
        --runtime-db data/runtime.db --notes-db data/structured_notes.db --yes

Exit code: 0 ok (anche dry-run), 1 errore/conflitti bloccanti, 2 argomenti non validi.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import ensure_postgres_schema, get_database_url, pg_connect, redact_url


@dataclass(frozen=True)
class TableSpec:
    name: str
    source: str  # "runtime" | "notes" | "gex"
    keep_id: bool = False
    #: colonne che identificano la riga: usate per verificare che lo stesso id
    #: in Postgres indichi davvero la stessa riga (solo tabelle con keep_id)
    natural: tuple[str, ...] = ()
    #: (colonna FK, tabella padre): righe con padre non coerente vengono saltate
    parent: tuple[str, str] | None = None


#: Ordine = ordine di inserimento (padri prima dei figli).
TABLES: tuple[TableSpec, ...] = (
    TableSpec("weight_versions", "runtime", keep_id=True, natural=("created_at", "source")),
    TableSpec("predictions", "runtime", keep_id=True,
              natural=("date", "source", "target_type", "horizon_days")),
    TableSpec("outcomes", "runtime", keep_id=True, natural=("prediction_id",),
              parent=("prediction_id", "predictions")),
    TableSpec("alert_state", "runtime"),
    TableSpec("prices", "runtime"),
    TableSpec("gex_snapshots", "gex"),
    TableSpec("notes", "notes", keep_id=True, natural=("filing_url",)),
    TableSpec("barrier_levels", "notes", keep_id=True,
              natural=("note_id", "barrier_type", "level_pct", "observation_date"),
              parent=("note_id", "notes")),
    TableSpec("barrier_snapshots", "notes", parent=("barrier_id", "barrier_levels")),
    TableSpec("refresh_runs", "notes", keep_id=True, natural=("run_at",)),
    TableSpec("macro_snapshots", "notes"),
)

BATCH = 1000


@dataclass
class TableResult:
    table: str
    source_rows: int = 0
    target_before: int | None = None
    inserted: int = 0
    skipped_orphans: int = 0
    id_conflicts: int = 0
    note: str = ""


# ─── SQLite (sola lettura) ───────────────────────────────────────────────────


def open_sqlite_ro(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"DB SQLite non trovato: {path}")
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def sqlite_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r["name"] for r in conn.execute(f"PRAGMA table_info({table})")]


# ─── Postgres ────────────────────────────────────────────────────────────────


def pg_columns(raw: Any, table: str) -> list[str]:
    rows = raw.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = current_schema() AND table_name = %s "
        "ORDER BY ordinal_position",
        (table,),
    ).fetchall()
    return [r[0] for r in rows]


def pg_count(raw: Any, table: str) -> int | None:
    if not pg_columns(raw, table):
        return None
    return int(raw.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def realign_identity(raw: Any, table: str) -> None:
    """Porta la sequenza IDENTITY di ``table.id`` oltre il max(id) corrente."""
    raw.execute(
        f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), "
        f'GREATEST((SELECT COALESCE(MAX(id), 0) FROM "{table}"), 1), '
        f'(SELECT COUNT(*) > 0 FROM "{table}"))'
    )


def consistent_ids(raw: Any, spec: TableSpec, src_rows: dict[int, tuple]) -> set[int]:
    """Id sorgente che in Postgres indicano la stessa riga (chiave naturale uguale)."""
    if not src_rows:
        return set()
    cols = ", ".join(f'"{c}"' for c in spec.natural)
    ids = list(src_rows)
    got = raw.execute(
        f'SELECT id, {cols} FROM "{spec.name}" WHERE id = ANY(%s)', (ids,)
    ).fetchall()
    return {
        int(r[0]) for r in got
        if _norm(tuple(r[1:])) == _norm(src_rows[int(r[0])])
    }


def _norm(values: tuple) -> tuple:
    """Normalizza per il confronto SQLite↔Postgres (interi vs float, numeric)."""
    out = []
    for v in values:
        if isinstance(v, bool):
            v = int(v)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            v = float(v)
        out.append(v)
    return tuple(out)


# ─── Migrazione ──────────────────────────────────────────────────────────────


def migrate(
    sources: dict[str, Path],
    database_url: str | None,
    *,
    dry_run: bool,
) -> list[TableResult]:
    """Copia le tabelle; in dry-run conta soltanto (nessuna scrittura)."""
    sq: dict[str, sqlite3.Connection] = {}
    for key, path in sources.items():
        sq[key] = open_sqlite_ro(path)

    pg = None
    raw = None
    if database_url:
        if not dry_run:
            ensure_postgres_schema(database_url)
        pg = pg_connect(database_url)
        raw = pg.raw
        if dry_run:
            raw.read_only = True

    results: list[TableResult] = []
    # id sorgente "buoni" per tabella padre, per filtrare i figli
    valid_parent_ids: dict[str, set[int]] = {}
    try:
        for spec in TABLES:
            res = TableResult(spec.name)
            results.append(res)
            src = sq.get(spec.source)
            if src is None:
                res.note = f"sorgente '{spec.source}' non indicata"
                continue
            src_cols = sqlite_columns(src, spec.name)
            if not src_cols:
                res.note = "tabella assente nella sorgente"
                continue
            res.source_rows = int(src.execute(f'SELECT COUNT(*) FROM "{spec.name}"').fetchone()[0])
            if raw is None:
                continue
            res.target_before = pg_count(raw, spec.name)
            if dry_run:
                if res.target_before is None:
                    res.note = "schema Postgres non ancora creato"
                continue

            tgt_cols = set(pg_columns(raw, spec.name))
            cols = [c for c in src_cols if c in tgt_cols and (spec.keep_id or c != "id")]
            dropped = sorted(set(src_cols) - set(cols) - ({"id"} if not spec.keep_id else set()))
            if dropped:
                res.note = f"colonne ignorate (assenti in Postgres): {dropped}"

            rows = src.execute(
                f'SELECT {", ".join(chr(34) + c + chr(34) for c in cols)} FROM "{spec.name}"'
                + (' ORDER BY id' if "id" in src_cols else "")
            ).fetchall()

            if spec.parent:
                fk_col, parent_table = spec.parent
                good = valid_parent_ids.get(parent_table)
                if good is not None:
                    idx = cols.index(fk_col)
                    kept = [r for r in rows if r[idx] in good]
                    res.skipped_orphans = len(rows) - len(kept)
                    rows = kept

            col_sql = ", ".join(f'"{c}"' for c in cols)
            ph = ", ".join(["%s"] * len(cols))
            sql = f'INSERT INTO "{spec.name}" ({col_sql}) VALUES ({ph}) ON CONFLICT DO NOTHING'
            values = [tuple(r) for r in rows]
            before = res.target_before or 0
            with raw.cursor() as cur:
                for i in range(0, len(values), BATCH):
                    cur.executemany(sql, values[i:i + BATCH])
            res.inserted = pg_count(raw, spec.name) - before  # type: ignore[operator]

            if spec.keep_id:
                realign_identity(raw, spec.name)
                if spec.natural:
                    id_idx = cols.index("id")
                    nat_idx = [cols.index(c) for c in spec.natural if c in cols]
                    src_map = {int(r[id_idx]): tuple(r[i] for i in nat_idx) for r in rows}
                    ok_ids = consistent_ids(raw, spec, src_map) if len(nat_idx) == len(spec.natural) \
                        else set(src_map)
                    res.id_conflicts = len(src_map) - len(ok_ids)
                    valid_parent_ids[spec.name] = ok_ids
                else:
                    valid_parent_ids[spec.name] = {int(r[cols.index("id")]) for r in rows}

        if raw is not None and not dry_run:
            raw.commit()
    except Exception:
        if raw is not None:
            raw.rollback()
        raise
    finally:
        if pg is not None:
            pg.close()
        for c in sq.values():
            c.close()
    return results


def print_report(results: list[TableResult], *, dry_run: bool) -> None:
    title = "DRY-RUN (nessuna scrittura)" if dry_run else "MIGRAZIONE"
    print(f"\n== {title} ==")
    print(f"{'tabella':<18} {'sorgente':>9} {'pg prima':>9} {'inserite':>9} "
          f"{'orfane':>7} {'conflitti id':>12}  note")
    for r in results:
        before = "-" if r.target_before is None else str(r.target_before)
        ins = "-" if dry_run else str(r.inserted)
        print(f"{r.table:<18} {r.source_rows:>9} {before:>9} {ins:>9} "
              f"{r.skipped_orphans:>7} {r.id_conflicts:>12}  {r.note}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Migra i DB SQLite verso Postgres (idempotente)")
    p.add_argument("--runtime-db", type=Path, help="SQLite con predizioni/esiti/pesi/prezzi/alert")
    p.add_argument("--notes-db", type=Path, help="SQLite con note EDGAR, barriere e macro")
    p.add_argument("--gex-db", type=Path, help="SQLite con gex_snapshots (default: --notes-db)")
    p.add_argument("--database-url", help="Postgres di destinazione (default: env DATABASE_URL)")
    p.add_argument("--dry-run", action="store_true", help="conta soltanto, nessuna scrittura")
    p.add_argument("--yes", action="store_true", help="conferma la scrittura su Postgres")
    args = p.parse_args(argv)

    sources: dict[str, Path] = {}
    if args.runtime_db:
        sources["runtime"] = args.runtime_db
    if args.notes_db:
        sources["notes"] = args.notes_db
    gex = args.gex_db or args.notes_db
    if gex:
        sources["gex"] = gex
    if not sources:
        p.error("indica almeno una sorgente: --runtime-db, --notes-db o --gex-db")

    url = args.database_url or get_database_url()
    if not args.dry_run:
        if not url:
            p.error("serve --database-url o DATABASE_URL (oppure usa --dry-run)")
        if not args.yes:
            p.error("scrittura su Postgres: aggiungi --yes (prova prima con --dry-run)")
    if url:
        print(f"Postgres: {redact_url(url)}")
    for k, v in sources.items():
        print(f"SQLite {k:<8}: {v}")

    results = migrate(sources, url, dry_run=args.dry_run)
    print_report(results, dry_run=args.dry_run)

    conflicts = sum(r.id_conflicts + r.skipped_orphans for r in results)
    if conflicts and not args.dry_run:
        print(f"\nATTENZIONE: {conflicts} righe in conflitto/orfane non copiate (vedi tabella).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
