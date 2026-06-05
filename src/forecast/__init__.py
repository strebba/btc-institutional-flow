"""Forecast & self-learning spine.

Pipeline source-agnostic: Prediction → Outcome → Calibration.

- `models`        — dataclass Prediction / Outcome + costanti tipo target.
- `prediction_db` — persistenza SQLite (predictions, outcomes, weight_versions).
- `sources`       — adapter che producono Prediction da un dominio (dealer_flow, ema, portfolio).
- `verifier`      — assegna gli esiti alle predizioni mature usando prezzi reali.
- `calibration`   — (fase 2) propone nuovi pesi dai risultati storici.
"""
