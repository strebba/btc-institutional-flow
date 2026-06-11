#!/usr/bin/env python3
"""Export structured notes DB to JSON dump + compute BTC barrier prices."""

import json
import sqlite3
from datetime import date
from pathlib import Path

# Legge DB path da settings.yaml (rispetta .env override DB_PATH=runtime.db)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import get_settings
DB_PATH = Path(get_settings()["database"]["path"]).resolve()
RATIO = 0.0006
VAULT = Path.home() / "Documents/Obsidian Vault/Strebba_Wagmi"
OUTPUT = VAULT / "raw" / "data" / f"structured-notes-dump-{date.today().isoformat()}.json"

conn = sqlite3.connect(str(DB_PATH))
conn.row_factory = sqlite3.Row

# 1. Compute BTC prices for barriers missing them
total = conn.execute("SELECT COUNT(*) FROM barrier_levels").fetchone()[0]
missing = conn.execute(
    "SELECT COUNT(*) FROM barrier_levels WHERE level_price_btc IS NULL"
).fetchone()[0]

if missing:
    rows = conn.execute(
        "SELECT id, level_price_ibit FROM barrier_levels WHERE level_price_btc IS NULL AND level_price_ibit IS NOT NULL"
    ).fetchall()
    updated = 0
    for row in rows:
        btc_price = row["level_price_ibit"] / RATIO
        conn.execute(
            "UPDATE barrier_levels SET level_price_btc=? WHERE id=?",
            (btc_price, row["id"]),
        )
        updated += 1
    conn.commit()
    print(f"Updated BTC prices: {updated}/{missing} barriers (total {total})")
else:
    print(f"All {total} barriers already have BTC prices")

# 2. Export full dump: notes + barriers
notes = conn.execute("""
    SELECT n.*, b.id as barrier_id, b.barrier_type, b.level_pct,
           b.level_price_ibit, b.level_price_btc, b.observation_date,
           b.status as barrier_status
    FROM notes n
    LEFT JOIN barrier_levels b ON n.id = b.note_id
    WHERE n.issuer IS NOT NULL
    ORDER BY n.issue_date DESC
""").fetchall()

# Group by note
dump = {}
for row in notes:
    r = dict(row)
    nid = r["id"]
    if nid not in dump:
        dump[nid] = {
            "id": nid,
            "filing_url": r["filing_url"],
            "issuer": r["issuer"],
            "issue_date": r["issue_date"],
            "maturity_date": r["maturity_date"],
            "notional_usd": r["notional_usd"],
            "product_type": r["product_type"],
            "underlying": r["underlying"],
            "initial_level": r["initial_level"],
            "autocall_trigger_pct": r["autocall_trigger_pct"],
            "knockin_barrier_pct": r["knockin_barrier_pct"],
            "buffer_pct": r["buffer_pct"],
            "coupon_rate": r["coupon_rate"],
            "is_preliminary": bool(r["is_preliminary"]),
            "created_at": r["created_at"],
            "barriers": [],
        }
    if r["barrier_id"]:
        dump[nid]["barriers"].append({
            "id": r["barrier_id"],
            "barrier_type": r["barrier_type"],
            "level_pct": r["level_pct"],
            "level_price_ibit": r["level_price_ibit"],
            "level_price_btc": r["level_price_btc"],
            "observation_date": r["observation_date"],
            "status": r["barrier_status"],
        })

# 3. Summary stats
active_barriers = conn.execute(
    "SELECT COUNT(*) FROM barrier_levels WHERE status='active'"
).fetchone()[0]
knock_in = conn.execute(
    "SELECT COUNT(*) FROM barrier_levels WHERE barrier_type='knock_in' AND status='active'"
).fetchone()[0]
autocall = conn.execute(
    "SELECT COUNT(*) FROM barrier_levels WHERE barrier_type='autocall' AND status='active'"
).fetchone()[0]
buffer_ = conn.execute(
    "SELECT COUNT(*) FROM barrier_levels WHERE barrier_type='buffer' AND status='active'"
).fetchone()[0]
by_issuer = dict(conn.execute(
    "SELECT issuer, COUNT(*) FROM notes WHERE issuer IS NOT NULL GROUP BY issuer ORDER BY COUNT(*) DESC"
).fetchall())
total_notional = conn.execute(
    "SELECT SUM(notional_usd) FROM notes WHERE notional_usd IS NOT NULL"
).fetchone()[0] or 0

dump_summary = {
    "export_date": date.today().isoformat(),
    "ibt_btc_ratio": RATIO,
    "total_notes": len(dump),
    "total_barriers": active_barriers,
    "active_barriers": active_barriers,
    "by_type": {"knock_in": knock_in, "autocall": autocall, "buffer": buffer_},
    "by_issuer": by_issuer,
    "total_notional_usd": total_notional,
}

output = {"summary": dump_summary, "notes": list(dump.values())}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"Dump written: {OUTPUT} ({len(dump)} notes, {active_barriers} barriers)")

conn.close()
