"""Cron script: verifica le previsioni mature e ne salva gli esiti.

Wrapper sottile su src.forecast.jobs.run_daily_verify. Idempotente (UNIQUE su outcomes).

Uso:
    python3 scripts/cron_verify.py

Scheduling suggerito (una volta al giorno, dopo cron_predict):
    30 7 * * * /path/venv/bin/python3 /path/scripts/cron_verify.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    from src.forecast.jobs import run_daily_verify

    res = run_daily_verify()
    print(f"[{res['status']}] mature={res['due']} verificate={res['verified']} "
          f"hit={res['hit']} miss={res['miss']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
