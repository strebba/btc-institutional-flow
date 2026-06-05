"""Cron script: genera le previsioni dealer-flow del giorno e le salva (status open).

Wrapper sottile su src.forecast.jobs.run_daily_predict (stessa logica usata dallo scheduler API).

Uso:
    python3 scripts/cron_predict.py [--horizon 5]

Scheduling suggerito (una volta al giorno, mattina UTC):
    0 7 * * 1-5  /path/venv/bin/python3 /path/scripts/cron_predict.py

Exit code: 0 ok, 1 fetch critico fallito.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera le previsioni dealer-flow del giorno")
    parser.add_argument("--horizon", type=int, default=5, help="Orizzonte in giorni (default 5)")
    args = parser.parse_args()

    from src.forecast.jobs import run_daily_predict

    res = run_daily_predict(horizon=args.horizon)
    print(f"[{res['status']}] " + (
        f"{res.get('signal')} score={res.get('score')} → "
        f"{res.get('inserted')}/{res.get('total')} previsioni (weights_v={res.get('weights_version')})"
        if res["status"] == "ok" else res.get("error", "")
    ))
    return 0 if res["status"] in ("ok", "skipped_kill_switch") else 1


if __name__ == "__main__":
    sys.exit(main())
