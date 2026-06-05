"""Cron script: genera le previsioni dealer-flow del giorno e le salva (status open).

Riusa il contesto di gather_dealer_flow_context (stessi data source di cron_signal) e
applica i pesi attivi versionati (self-learning). Le previsioni restano `open` finché
cron_verify non le valuta alla maturazione dell'orizzonte.

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

from src.config import setup_logging

_log = setup_logging("cron_predict")


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera le previsioni dealer-flow del giorno")
    parser.add_argument("--horizon", type=int, default=5, help="Orizzonte in giorni (default 5)")
    args = parser.parse_args()

    from src.forecast.context import DataUnavailable, gather_dealer_flow_context
    from src.forecast.prediction_db import PredictionDB
    from src.forecast.sources.dealer_flow import SOURCE, build_dealer_flow_predictions

    db = PredictionDB()

    # Pesi attivi versionati (self-learning); None → default del SignalModel.
    active = db.get_active_weights(SOURCE)
    weights = active[1] if active else None
    weights_version = active[0] if active else None

    try:
        ctx = gather_dealer_flow_context(weights=weights)
    except DataUnavailable as exc:
        _log.error("%s", exc)
        return 1

    snap = ctx.snapshot
    preds = build_dealer_flow_predictions(
        ctx.result,
        spot_price=ctx.spot,
        gamma_flip=snap.gamma_flip_price,
        max_pain=snap.max_pain,
        put_wall=snap.put_wall,
        call_wall=snap.call_wall,
        total_net_gex=snap.total_net_gex,
        horizon_days=args.horizon,
        weights_version=weights_version,
    )

    inserted = 0
    for p in preds:
        if db.insert_prediction(p) is not None:
            inserted += 1

    print(
        f"[OK] {ctx.result.signal} score={ctx.result.score:.1f} spot={ctx.spot:,.0f} "
        f"gex={snap.total_net_gex/1e6:+.1f}M → {inserted}/{len(preds)} previsioni salvate "
        f"(horizon {args.horizon}g, weights_v={weights_version})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
