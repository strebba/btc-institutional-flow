"""Rilevamento del regime di gamma e generazione di alert.

Classifica il regime corrente in:
  - positive_gamma: GEX totale > threshold → mercato stabilizzante
  - negative_gamma: GEX totale < -threshold → mercato amplificante
  - neutral: |GEX| < threshold

Genera alert per:
  - Gamma flip (cambio di segno del GEX totale)
  - Spot entro X% di put_wall o call_wall
  - GEX nel 10° percentile storico (regime estremo)
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np

from src.config import get_settings, setup_logging
from src.gex.models import GexSnapshot, RegimeState

_log = setup_logging("gex.regime")


class RegimeDetector:
    """Classifica il regime di gamma e genera alert.

    Args:
        cfg: configurazione deribit (da settings.yaml).
        alert_cfg: configurazione analytics (da settings.yaml).
    """

    def __init__(
        self,
        cfg: dict | None = None,
        alert_cfg: dict | None = None,
    ) -> None:
        settings       = get_settings()
        self._cfg      = cfg or settings["deribit"]
        self._alert_cfg = alert_cfg or settings["analytics"]
        self._history: list[GexSnapshot] = []  # storico in memoria

    def load_history_from_db(self, snapshots: list[GexSnapshot]) -> None:
        """Pre-popola lo storico in memoria con snapshot letti dal DB.

        Da chiamare all'avvio (API, dashboard, cron) per garantire che
        gex_percentile e gamma-flip alert siano calcolati su dati reali
        e non solo sulla sessione corrente.

        Args:
            snapshots: lista ordinata per data crescente (da GexDB.get_latest_n).
        """
        self._history = list(snapshots)
        _log.info("RegimeDetector: storico pre-popolato con %d snapshot", len(self._history))

    def add_snapshot(self, snapshot: GexSnapshot) -> None:
        """Aggiunge uno snapshot allo storico in memoria.

        Args:
            snapshot: GexSnapshot da aggiungere.
        """
        self._history.append(snapshot)

    def detect(self, snapshot: GexSnapshot) -> RegimeState:
        """Classifica il regime e genera alert per lo snapshot corrente.

        Args:
            snapshot: snapshot GEX corrente.

        Returns:
            RegimeState: regime classificato con alert.
        """
        threshold = self._cfg.get("gex_threshold_usd", 1_000_000)
        proximity = self._alert_cfg.get("barrier_proximity_pct", 3.0)
        gex       = snapshot.total_net_gex

        # Classificazione regime
        if gex > threshold:
            regime = "positive_gamma"
        elif gex < -threshold:
            regime = "negative_gamma"
        else:
            regime = "neutral"

        # Calcolo percentile effettivo rispetto allo storico
        gex_percentile = None
        if len(self._history) >= 5:
            hist_vals    = [s.total_net_gex for s in self._history]
            gex_percentile = float(
                np.sum(np.array(hist_vals) <= gex) / len(hist_vals) * 100
            )

        # Generazione alert
        alerts: list[str] = []

        # 1. Gamma flip appena avvenuto
        if len(self._history) >= 1:
            prev_gex = self._history[-1].total_net_gex
            if (prev_gex >= 0) != (gex >= 0):
                direction = "positivo → negativo" if gex < 0 else "negativo → positivo"
                alerts.append(f"GAMMA FLIP: {direction} (GEX={gex/1e6:.1f}M)")

        # 2. Spot near put_wall
        if snapshot.put_wall and snapshot.spot_price:
            dist_put = abs(snapshot.spot_price - snapshot.put_wall) / snapshot.spot_price * 100
            if dist_put <= proximity:
                alerts.append(
                    f"NEAR PUT_WALL: spot ${snapshot.spot_price:,.0f} "
                    f"a {dist_put:.1f}% da ${snapshot.put_wall:,.0f}"
                )

        # 3. Spot near call_wall
        if snapshot.call_wall and snapshot.spot_price:
            dist_call = abs(snapshot.spot_price - snapshot.call_wall) / snapshot.spot_price * 100
            if dist_call <= proximity:
                alerts.append(
                    f"NEAR CALL_WALL: spot ${snapshot.spot_price:,.0f} "
                    f"a {dist_call:.1f}% da ${snapshot.call_wall:,.0f}"
                )

        # 4. GEX nel 10° percentile storico
        if gex_percentile is not None and gex_percentile <= 10:
            alerts.append(
                f"GEX ESTREMO NEGATIVO: percentile={gex_percentile:.0f}% "
                f"(GEX={gex/1e6:.1f}M)"
            )

        if alerts:
            for a in alerts:
                _log.warning("ALERT: %s", a)

        state = RegimeState(
            timestamp=snapshot.timestamp,
            regime=regime,
            total_net_gex=gex,
            spot_price=snapshot.spot_price,
            put_wall=snapshot.put_wall,
            call_wall=snapshot.call_wall,
            gamma_flip=snapshot.gamma_flip_price,
            alerts=alerts,
            gex_percentile=gex_percentile,
        )

        # Aggiungi allo storico dopo la classificazione (per il prossimo ciclo)
        self.add_snapshot(snapshot)

        return state

    def summary(self, state: RegimeState) -> str:
        """Genera una stringa di riepilogo leggibile del regime corrente.

        Args:
            state: RegimeState corrente.

        Returns:
            str: summary formattato.
        """
        regime_emoji = {
            "positive_gamma": "🟢",
            "negative_gamma": "🔴",
            "neutral":        "🟡",
        }.get(state.regime, "⚪")

        lines = [
            f"{regime_emoji} Regime: {state.regime.upper()}",
            f"  Spot:           ${state.spot_price:>10,.0f}",
            f"  Total Net GEX:  ${state.total_net_gex/1e6:>10.1f}M",
            f"  Gamma Flip:     ${state.gamma_flip or 0:>10,.0f}",
            f"  Put Wall:       ${state.put_wall or 0:>10,.0f}",
            f"  Call Wall:      ${state.call_wall or 0:>10,.0f}",
        ]
        if state.gex_percentile is not None:
            lines.append(f"  GEX percentile: {state.gex_percentile:>9.0f}%")
        if state.alerts:
            lines.append("\n  ⚠️  ALERT:")
            for a in state.alerts:
                lines.append(f"    • {a}")
        return "\n".join(lines)
