#!/usr/bin/env python3
"""Re-parse Goldman Sachs (and other issuer) notes with fixed regex patterns.

Steps:
  1. Collect filing metadata for notes with NULL initial_level
  2. Re-parse each with the updated ProspectusParser
  3. Upsert into DB with new extracted fields
  4. Print before/after comparison

Usage:
  .venv/bin/python scripts/reparse_goldman.py [--issuer "Goldman Sachs"] [--dry-run]
"""

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

from src.config import get_settings, setup_logging  # noqa: E402
from src.edgar.parser import ProspectusParser  # noqa: E402
from src.edgar.structured_notes_db import StructuredNotesDB  # noqa: E402

_log = setup_logging("reparse")


def main():
    parser = argparse.ArgumentParser(description="Re-parse structured notes with fixed regex")
    parser.add_argument("--issuer", default="Goldman Sachs",
                        help="Issuer to re-parse (default: Goldman Sachs)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write to DB, only print what would change")
    args = parser.parse_args()

    cfg = get_settings()
    db = StructuredNotesDB()
    all_notes = db.get_all_notes()

    # Find notes to re-parse
    target_notes = [n for n in all_notes
                    if n.issuer == args.issuer
                    and not n.is_preliminary
                    and n.initial_level is None]

    if not target_notes:
        print(f"No {args.issuer} notes to re-parse (all have initial_level set or are preliminary)")
        return

    print(f"Found {len(target_notes)} {args.issuer} notes to re-parse")

    # Re-fetch and re-parse each
    reparser = ProspectusParser(cfg["edgar"])

    updated = 0
    unchanged = 0

    for note in target_notes:
        print(f"\n  [{note.id}] {note.filing_url.split('/')[-1][:60]}...")
        print(f"    Before: initial_level={note.initial_level}, product={note.product_type}, "
              f"knockin_pct={note.knockin_barrier_pct}, barriers={len(note.barriers)}")

        # Build filing_meta from existing note data
        filing_meta = {
            "url": note.filing_url,
            "entity_name": note.issuer,
            "filing_date": note.issue_date.isoformat() if note.issue_date else None,
            "entity_id": None,
            "accession_no": None,
            "form_type": "424B2",
        }

        new_note = reparser.parse(filing_meta)

        changed = False
        if new_note.initial_level != note.initial_level:
            print(f"    After:  initial_level={new_note.initial_level}")
            changed = True
        if len(new_note.barriers) != len(note.barriers):
            print(f"    After:  barriers={len(new_note.barriers)} (was {len(note.barriers)})")
            changed = True

        if changed:
            updated += 1
            if not args.dry_run:
                # Merge: preserve filing_url and id from original
                new_note.id = note.id
                new_note.filing_url = note.filing_url
                db.upsert_note(new_note)
                print("    → UPSERTED")
        else:
            unchanged += 1
            print("    → No change")

    print(f"\n{'='*60}")
    print(f"Done: {updated} updated, {unchanged} unchanged, {len(target_notes)} total")
    if args.dry_run:
        print("(dry-run — no writes)")


if __name__ == "__main__":
    main()
