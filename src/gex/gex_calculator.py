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

from datetime import datetime, timezone
from typing import Optional


from src.config import get_settings, setup_logging
from src.gex.greeks import black_scholes_greeks, time_to_expiry_years
from src.gex.models import GexByStrike, GexSnapshot

_log = setup_logging("gex.calculator")

#: Griglia di spot ipotetici per il profilo gamma: ±25% dallo spot, passo 0,25%.
_FLIP_RANGE_PCT = 0.25
_FLIP_STEP_PCT = 0.0025


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

    @staticmethod
    def _bs_gamma(opt: dict, spot: float, now_ms: float) -> Optional[float]:
        """Gamma Black-Scholes ricostruita da mark_iv / scadenza / underlying.

        Deribit pubblica ``greeks.gamma`` arrotondata a 5 decimali: su BTC (gamma
        tipica 1e-5..5e-5) resta una sola cifra significativa e le opzioni OTM
        con gamma < 5e-6 diventano 0 e vengono scartate dal GEX. La ricostruzione
        usa gli stessi input di ``src.gex.greeks`` (già validato su delta Deribit).

        Returns:
            gamma per unità di sottostante, oppure None se non calcolabile.
        """
        exp = opt.get("expiration_timestamp") or 0
        iv = opt.get("mark_iv") or 0.0
        if not exp or iv <= 0:
            return None
        g = black_scholes_greeks(
            spot=opt.get("underlying_price") or spot,   # forward della scadenza
            strike=opt.get("strike", 0.0),
            t_years=time_to_expiry_years(exp, now_ms),
            sigma=iv / 100.0,              # Deribit quota la IV in %
            option_type=opt.get("option_type", ""),
        )
        return g.gamma if g is not None else None

    def calculate_gex(
        self,
        options_data: list[dict],
        spot_price: float,
        now_ms: Optional[float] = None,
    ) -> GexSnapshot:
        """Calcola il GEX completo dalla chain di opzioni.

        Args:
            options_data: lista di dict da DeribitClient.fetch_all_options().
            spot_price: prezzo spot BTC corrente.

            now_ms: istante di riferimento in ms (default: adesso). Serve alla
                ricostruzione delle greche; parametrico per i test.

        Returns:
            GexSnapshot: snapshot completo con tutte le metriche.
        """
        if now_ms is None:
            now_ms = datetime.now(timezone.utc).timestamp() * 1000.0
        if not options_data:
            _log.warning("Nessun dato opzioni disponibile")
            return GexSnapshot(
                timestamp=datetime.now(timezone.utc),
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
            oi           = opt.get("open_interest", 0.0)
            option_type  = opt.get("option_type", "").lower()

            if strike <= 0 or oi <= 0 or option_type not in ("call", "put"):
                continue

            # Gamma BS a piena precisione; fallback sulla gamma Deribit (arrotondata).
            gamma = self._bs_gamma(opt, spot_price, now_ms)
            if gamma is None:
                gamma = opt.get("gamma", 0.0) or 0.0
            if gamma <= 0:
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
        # Metodo corretto: zero del profilo GEX(S) ricalcolato su spot ipotetici.
        # Il cumulativo per strike resta solo come fallback per chain senza
        # IV/scadenza (es. fixture di test o dati parziali).
        gamma_flip, n_legs = self._gamma_profile_flip(options_data, spot_price, now_ms)
        if n_legs == 0:
            _log.warning("Gamma flip: chain senza IV/scadenza, fallback cumulativo per strike")
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
        max_pain = self._calculate_max_pain(options_data)

        _log.info(
            "GEX calcolato: total_net=%.2fM, flip=%.0f, put_wall=%.0f, call_wall=%.0f, max_pain=%.0f",
            total_net_gex / 1e6,
            gamma_flip or 0,
            put_wall or 0,
            call_wall or 0,
            max_pain or 0,
        )

        return GexSnapshot(
            timestamp=datetime.now(timezone.utc),
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

    def _gamma_profile_flip(
        self,
        options_data: list[dict],
        spot_price: float,
        now_ms: float,
    ) -> tuple[Optional[float], int]:
        """Gamma flip come zero del profilo GEX totale in funzione dello spot.

        Per ogni spot ipotetico S nella griglia ±``_FLIP_RANGE_PCT`` ricalcola la
        gamma BS di ogni opzione a S e somma il GEX dealer (stessa convenzione di
        segno di ``_option_gex``). Il flip è lo zero (interpolato) più vicino allo
        spot corrente: sopra il flip GEX > 0, sotto GEX < 0, coerente con
        ``total_net_gex`` e con il regime.

        Returns:
            (flip, n_legs): flip None se il profilo non cambia segno nel range;
            n_legs = opzioni ri-prezzabili (0 → il chiamante usa il fallback).
        """
        if spot_price <= 0:
            return None, 0

        # (strike, t, iv, tipo, oi, basis): basis = forward/index della scadenza,
        # così lo spot ipotetico S muove il forward in proporzione.
        legs: list[tuple[float, float, float, str, float, float]] = []
        for opt in options_data:
            strike = opt.get("strike", 0.0) or 0.0
            oi = opt.get("open_interest", 0.0) or 0.0
            tipo = (opt.get("option_type") or "").lower()
            iv = (opt.get("mark_iv") or 0.0) / 100.0
            exp = opt.get("expiration_timestamp") or 0
            if strike <= 0 or oi <= 0 or tipo not in ("call", "put") or iv <= 0 or not exp:
                continue
            t = time_to_expiry_years(exp, now_ms)
            if t <= 0:
                continue
            und = opt.get("underlying_price") or spot_price
            legs.append((strike, t, iv, tipo, oi, und / spot_price))

        if not legs:
            return None, 0

        def total_gex_at(s: float) -> float:
            tot = 0.0
            for strike, t, iv, tipo, oi, basis in legs:
                g = black_scholes_greeks(
                    spot=s * basis, strike=strike, t_years=t, sigma=iv, option_type=tipo
                )
                if g is not None:
                    tot += self._option_gex(g.gamma, oi, tipo, s)
            return tot

        n = int(round(_FLIP_RANGE_PCT / _FLIP_STEP_PCT))
        grid = [spot_price * (1.0 + i * _FLIP_STEP_PCT) for i in range(-n, n + 1)]
        values = [total_gex_at(s) for s in grid]

        crossings: list[float] = []
        for (s0, v0), (s1, v1) in zip(zip(grid, values), zip(grid[1:], values[1:])):
            if (v0 < 0) != (v1 < 0):
                den = abs(v0) + abs(v1)
                crossings.append(s0 + (s1 - s0) * (abs(v0) / den if den > 0 else 0.0))

        if not crossings:
            return None, len(legs)
        return min(crossings, key=lambda x: abs(x - spot_price)), len(legs)

    def _find_gamma_flip(self, gex_by_strike: list[GexByStrike]) -> Optional[float]:
        """Fallback legacy: zero del GEX cumulativo per strike (dal basso).

        NB: non è il gamma flip in senso stretto (usa le gamma allo spot
        corrente e cade strutturalmente vicino all'ATM, dove le put lasciano
        spazio alle call). Usato solo se la chain non ha IV/scadenza.

        Il gamma flip è calcolato cumulando il GEX dagli strike più bassi
        verso quelli più alti e cercando il cambio di segno. Quando il segno
        cambia tra due strike, viene interpolato linearmente per maggiore
        precisione.

        Args:
            gex_by_strike: lista di GexByStrike ordinata per strike crescente.

        Returns:
            float | None: prezzo interpolato del gamma flip.
        """
        if not gex_by_strike:
            return None

        cumulative = 0.0
        prev_sign = None
        last_strike = gex_by_strike[0].strike if gex_by_strike else None

        for gs in gex_by_strike:
            prev_cum = cumulative
            cumulative += gs.net_gex
            curr_sign = 1 if cumulative >= 0 else -1
            if prev_sign is not None and curr_sign != prev_sign:
                if abs(prev_cum) + abs(cumulative) > 0:
                    ratio = abs(prev_cum) / (abs(prev_cum) + abs(cumulative))
                    return last_strike + (gs.strike - last_strike) * ratio
                return gs.strike
            prev_sign = curr_sign
            last_strike = gs.strike

        return None

    def _calculate_max_pain(
        self,
        options_data: list[dict],
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
