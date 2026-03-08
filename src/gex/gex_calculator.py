"""Calcolo del Gamma Exposure (GEX) per strike dalle opzioni BTC su Deribit.

Formula GEX per ogni opzione:
  gex = gamma × open_interest × contract_size × (spot_price²) × 0.01
  segno: +1 per call, -1 per put (dealer tipicamente short gamma)

Metriche calcolate:
  total_net_gex   → somma di tutti i GEX
  gamma_flip_price → dove il GEX cumulativo cambia segno
  put_wall        → strike con massimo |GEX| negativo
  call_wall       → strike con massimo GEX positivo
  max_pain        → strike che minimizza il payoff totale
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Optional

import numpy as np

from src.config import get_settings, setup_logging
from src.gex.models import GexByStrike, GexSnapshot

_log = setup_logging("gex.calculator")


class GexCalculator:
    """Calcola il GEX dalla chain di opzioni scaricata da Deribit.

    Args:
        cfg: configurazione deribit (da settings.yaml).
    """

    def __init__(self, cfg: dict | None = None) -> None:
        self._cfg = cfg or get_settings()["deribit"]

    # ──────────────────────────────────────────────────────────────────────────
    # Core calculation
    # ──────────────────────────────────────────────────────────────────────────

    def _option_gex(
        self,
        gamma: float,
        open_interest: float,
        option_type: str,
        spot_price: float,
    ) -> float:
        """Calcola il GEX per una singola opzione.

        Formula: GEX = gamma × OI × contract_size × spot² × 0.01 × sign
        Il fattore 0.01 normalizza per il movimento di 1% dello spot.

        Il segno riflette la posizione tipica del dealer:
          - Call: dealer tende ad essere short gamma → +GEX stabilizza il mercato
          - Put:  dealer tende ad essere short gamma → -GEX destabilizza

        Args:
            gamma: gamma dell'opzione (in BTC per % di movimento spot).
            open_interest: open interest in numero di contratti.
            option_type: "call" o "put".
            spot_price: prezzo spot BTC corrente.

        Returns:
            float: GEX in USD.
        """
        contract_size = self._cfg.get("contract_size", 1.0)
        sign = 1.0 if option_type == "call" else -1.0
        return sign * gamma * open_interest * contract_size * (spot_price ** 2) * 0.01

    def calculate_gex(
        self,
        options_data: list[dict],
        spot_price: float,
    ) -> GexSnapshot:
        """Calcola il GEX completo dalla chain di opzioni.

        Args:
            options_data: lista di dict da DeribitClient.fetch_all_options().
            spot_price: prezzo spot BTC corrente.

        Returns:
            GexSnapshot: snapshot completo con tutte le metriche.
        """
        if not options_data:
            _log.warning("Nessun dato opzioni disponibile")
            return GexSnapshot(
                timestamp=datetime.utcnow(),
                spot_price=spot_price,
                total_net_gex=0.0,
                gamma_flip_price=None,
                put_wall=None,
                call_wall=None,
                max_pain=None,
            )

        # Aggrega per strike
        gex_map: dict[float, GexByStrike] = {}
        total_call_oi = 0.0
        total_put_oi  = 0.0

        for opt in options_data:
            strike       = opt.get("strike", 0.0)
            gamma        = opt.get("gamma", 0.0)
            oi           = opt.get("open_interest", 0.0)
            option_type  = opt.get("option_type", "").lower()

            if strike <= 0 or gamma <= 0 or oi <= 0:
                continue
            if option_type not in ("call", "put"):
                continue

            gex_val = self._option_gex(gamma, oi, option_type, spot_price)

            if strike not in gex_map:
                gex_map[strike] = GexByStrike(strike=strike)

            gs = gex_map[strike]
            if option_type == "call":
                gs.call_gex += gex_val
                gs.call_oi  += oi
                total_call_oi += oi
            else:
                gs.put_gex += gex_val
                gs.put_oi  += oi
                total_put_oi += oi

        # Calcola net GEX per strike
        for gs in gex_map.values():
            gs.net_gex = gs.call_gex + gs.put_gex

        gex_by_strike = sorted(gex_map.values(), key=lambda g: g.strike)
        total_net_gex = sum(g.net_gex for g in gex_by_strike)

        # ── Gamma flip price ──────────────────────────────────────────────────
        gamma_flip = self._find_gamma_flip(gex_by_strike)

        # ── Put wall (strike con minimo GEX netto) ────────────────────────────
        put_wall = None
        negative_strikes = [g for g in gex_by_strike if g.net_gex < 0]
        if negative_strikes:
            put_wall = min(negative_strikes, key=lambda g: g.net_gex).strike

        # ── Call wall (strike con massimo GEX netto) ──────────────────────────
        call_wall = None
        positive_strikes = [g for g in gex_by_strike if g.net_gex > 0]
        if positive_strikes:
            call_wall = max(positive_strikes, key=lambda g: g.net_gex).strike

        # ── Max pain ──────────────────────────────────────────────────────────
        max_pain = self._calculate_max_pain(options_data, spot_price)

        _log.info(
            "GEX calcolato: total_net=%.2fM, flip=%.0f, put_wall=%.0f, call_wall=%.0f, max_pain=%.0f",
            total_net_gex / 1e6,
            gamma_flip or 0,
            put_wall or 0,
            call_wall or 0,
            max_pain or 0,
        )

        return GexSnapshot(
            timestamp=datetime.utcnow(),
            spot_price=spot_price,
            total_net_gex=total_net_gex,
            gamma_flip_price=gamma_flip,
            put_wall=put_wall,
            call_wall=call_wall,
            max_pain=max_pain,
            gex_by_strike=gex_by_strike,
            total_call_oi=total_call_oi,
            total_put_oi=total_put_oi,
        )

    def _find_gamma_flip(self, gex_by_strike: list[GexByStrike]) -> Optional[float]:
        """Trova il prezzo di gamma flip (dove il GEX cumulativo cambia segno).

        Il gamma flip è calcolato cumulando il GEX dagli strike più bassi
        verso quelli più alti e cercando il cambio di segno.

        Args:
            gex_by_strike: lista di GexByStrike ordinata per strike crescente.

        Returns:
            float | None: strike del gamma flip.
        """
        if not gex_by_strike:
            return None

        cumulative = 0.0
        prev_sign  = None

        for gs in gex_by_strike:
            cumulative += gs.net_gex
            curr_sign   = 1 if cumulative >= 0 else -1
            if prev_sign is not None and curr_sign != prev_sign:
                return gs.strike
            prev_sign = curr_sign

        return None

    def _calculate_max_pain(
        self,
        options_data: list[dict],
        spot_price: float,
    ) -> Optional[float]:
        """Calcola il max pain: strike che minimizza il payoff totale delle opzioni.

        Per ogni potenziale strike di scadenza, calcola il valore totale che
        tutti i holder di opzioni riceverebbero. Il max pain è il minimo di questa
        funzione (cioè, lo strike che fa perdere di più agli holder e
        beneficia i market maker).

        Args:
            options_data: lista di opzioni da Deribit.
            spot_price: prezzo spot BTC.

        Returns:
            float | None: strike max pain.
        """
        strikes = sorted(set(opt["strike"] for opt in options_data if opt.get("strike", 0) > 0))
        if not strikes:
            return None

        min_pain   = float("inf")
        pain_strike = None

        for test_strike in strikes:
            total_pain = 0.0
            for opt in options_data:
                oi          = opt.get("open_interest", 0.0)
                strike      = opt.get("strike", 0.0)
                option_type = opt.get("option_type", "").lower()
                if not oi or not strike:
                    continue
                if option_type == "call":
                    total_pain += max(0, test_strike - strike) * oi
                elif option_type == "put":
                    total_pain += max(0, strike - test_strike) * oi

            if total_pain < min_pain:
                min_pain    = total_pain
                pain_strike = test_strike

        return pain_strike

    # ──────────────────────────────────────────────────────────────────────────
    # Visualization helper
    # ──────────────────────────────────────────────────────────────────────────

    def gex_to_dict(self, snapshot: GexSnapshot) -> dict:
        """Serializza uno GexSnapshot in un dict Python-nativo.

        Args:
            snapshot: istanza GexSnapshot.

        Returns:
            dict: rappresentazione serializzabile.
        """
        return {
            "timestamp":               snapshot.timestamp.isoformat(),
            "spot_price":              snapshot.spot_price,
            "total_net_gex":           snapshot.total_net_gex,           # raw USD
            "total_net_gex_m":         round(snapshot.total_net_gex / 1e6, 2),
            "gamma_flip_price":        snapshot.gamma_flip_price,
            "put_wall":                snapshot.put_wall,
            "call_wall":               snapshot.call_wall,
            "max_pain":                snapshot.max_pain,
            "total_call_oi":           snapshot.total_call_oi,
            "total_put_oi":            snapshot.total_put_oi,
            "put_call_ratio":          round(snapshot.put_call_ratio or 0, 3),
            "distance_to_put_wall_pct":  snapshot.distance_to_put_wall_pct,
            "distance_to_call_wall_pct": snapshot.distance_to_call_wall_pct,
            "num_strikes":             len(snapshot.gex_by_strike),
            "n_instruments":           len(snapshot.gex_by_strike),      # alias
        }
