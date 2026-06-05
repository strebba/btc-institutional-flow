"""Cron script: calibrazione settimanale — calcola metriche e propone nuovi pesi.

Non attiva nulla: la proposta resta 'proposed' nel DB. L'attivazione è human-gated
(workflow Tuning nel vault). Eseguire dopo che si sono accumulati esiti maturi.

Uso:
    python3 scripts/cron_calibrate.py [--source dealer_flow] [--days 180]

Scheduling suggerito (settimanale):
    0 8 * * 1  /path/venv/bin/python3 /path/scripts/cron_calibrate.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import setup_logging

_log = setup_logging("cron_calibrate")


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrazione human-gated dei pesi")
    parser.add_argument("--source", default="dealer_flow")
    parser.add_argument("--days", type=int, default=180)
    args = parser.parse_args()

    from src.forecast.jobs import run_weekly_calibrate

    report = run_weekly_calibrate(source=args.source, days=args.days)

    print(f"=== Calibrazione {args.source} ===")
    print(f"esiti totali: {report.metrics['total_scored']}  gate={'OK' if report.gate_ok else 'NO'}")
    for tt, m in report.metrics["by_target_type"].items():
        print(f"  {tt:9} scored={m['scored']:3} hit_rate={m['hit_rate']} "
              f"binom_p={m['binomial_p']} brier={m['mean_brier']}")
    if report.proposed_weights:
        print("  proposta pesi:", {k: round(v, 3) for k, v in report.proposed_weights.items()})
        print("  →", report.rationale)
    for n in report.notes:
        print("  •", n)
    print("\nNB: nessun peso attivato. Attivazione manuale via workflow Tuning.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
