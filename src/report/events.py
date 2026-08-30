"""Rilevamento degli eventi che fanno uscire un'edizione del Desk Note.

Pubblicare ogni giorno obbliga a scrivere sei card anche quando non si muove
niente, ed è così che si finisce a riempire. Qui invece l'edizione esce quando
qualcosa di verificabile è cambiato: un regime che si ribalta, una barriera
attraversata, un muro superato, un flusso fuori scala.

:func:`detect_events` è una funzione pura — confronta due fotografie e dice
cosa è successo in mezzo. La persistenza dello stato sta in
:class:`ReportStateDB`, separata, così la logica resta testabile senza database.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import get_settings, setup_logging
from src.report import formatting as fmt

_log = setup_logging("report.events")

#: Sopra questa severità l'edizione esce da sola; sotto, resta in coda.
PUBLISH_THRESHOLD = 0.55

#: Soglie operative del segnale composito (allineate a pillars.score_to_signal).
_SIGNAL_THRESHOLDS = (40.0, 65.0)

#: Un flusso ETF a 3 giorni oltre questo valore assoluto è notizia da solo.
_FLOW_SPIKE_USD_M = 500.0


@dataclass
class DeskEvent:
    """Un cambiamento verificabile fra due fotografie del mercato."""

    key: str
    severity: float          # 0-1, quanto merita di far uscire un'edizione
    title: str
    detail: str
    meta: dict[str, Any] = field(default_factory=dict)


# ─── Fotografia ───────────────────────────────────────────────────────────────


def snapshot_state(
    *,
    gex: dict | None = None,
    barriers: dict | None = None,
    signals: dict | None = None,
) -> dict:
    """Riduce i payload alle poche grandezze che servono a rilevare un evento.

    Tenere la fotografia piccola è deliberato: è ciò che viene serializzato e
    confrontato a ogni giro, e un diff su trenta campi produce rumore, non eventi.
    """
    snap = (gex or {}).get("snapshot") or {}
    regime = (gex or {}).get("regime") or {}
    rows = (barriers or {}).get("barriers") or []
    spot = snap.get("spot_price") or (barriers or {}).get("spot_price")

    livelli = sorted(
        b["level_price_btc"]
        for b in rows
        if isinstance(b.get("level_price_btc"), (int, float)) and b["level_price_btc"] > 0
    )

    return {
        "spot": spot,
        "regime": regime.get("label"),
        "gamma_flip": snap.get("gamma_flip_price"),
        "put_wall": snap.get("put_wall"),
        "call_wall": snap.get("call_wall"),
        "signal_score": (signals or {}).get("score"),
        "flow_3d_usd_m": ((signals or {}).get("inputs") or {}).get("ibit_flow_3d_usd_m"),
        "barrier_levels": livelli,
    }


# ─── Rilevamento ──────────────────────────────────────────────────────────────


def _crossed(prev: float | None, cur: float | None, level: float | None) -> int:
    """Direzione dell'attraversamento di ``level``: +1 sopra, -1 sotto, 0 nessuno."""
    if prev is None or cur is None or level is None:
        return 0
    if prev < level <= cur:
        return 1
    if prev > level >= cur:
        return -1
    return 0


def detect_events(current: dict, previous: dict | None) -> list[DeskEvent]:
    """Confronta due fotografie e restituisce gli eventi, dal più grave al meno.

    Senza fotografia precedente non ci sono eventi: la prima esecuzione stabilisce
    la linea di base invece di dichiarare che è cambiato tutto.
    """
    if not previous:
        return []

    eventi: list[DeskEvent] = []
    spot_prev, spot_cur = previous.get("spot"), current.get("spot")

    # ── regime gamma ribaltato ────────────────────────────────────────────────
    r_prev, r_cur = previous.get("regime"), current.get("regime")
    if r_prev and r_cur and r_prev != r_cur:
        verso = (
            "I dealer sono passati ad amplificare i movimenti"
            if r_cur == "negative_gamma"
            else "I dealer sono tornati ad assorbire i movimenti"
        )
        eventi.append(
            DeskEvent(
                key="gamma_regime_flip",
                severity=0.95,
                title="Il regime gamma si è ribaltato",
                detail=f"{r_prev} → {r_cur}. {verso}.",
                meta={"from": r_prev, "to": r_cur},
            )
        )

    # ── spot che attraversa il gamma flip ─────────────────────────────────────
    verso_flip = _crossed(spot_prev, spot_cur, current.get("gamma_flip"))
    if verso_flip:
        sopra = verso_flip > 0
        eventi.append(
            DeskEvent(
                key="gamma_flip_crossed",
                severity=0.85,
                title=(
                    "Lo spot è passato sopra il gamma flip"
                    if sopra
                    else "Lo spot è passato sotto il gamma flip"
                ),
                detail=(
                    f"Flip a {fmt.price(current.get('gamma_flip'))}, "
                    f"spot da {fmt.price(spot_prev)} a {fmt.price(spot_cur)}."
                ),
                meta={"direction": "up" if sopra else "down"},
            )
        )

    # ── muri superati ─────────────────────────────────────────────────────────
    for nome, etichetta in (("call_wall", "call wall"), ("put_wall", "put wall")):
        verso = _crossed(spot_prev, spot_cur, current.get(nome))
        if verso:
            eventi.append(
                DeskEvent(
                    key=f"{nome}_crossed",
                    severity=0.75,
                    title=f"Superato il {etichetta} a {fmt.price(current.get(nome))}",
                    detail=(
                        f"Lo spot è passato da {fmt.price(spot_prev)} a "
                        f"{fmt.price(spot_cur)}, {'sopra' if verso > 0 else 'sotto'} il livello."
                    ),
                    meta={"level": current.get(nome), "direction": "up" if verso > 0 else "down"},
                )
            )

    # ── barriere di note strutturate attraversate ─────────────────────────────
    rotte = [
        lvl for lvl in current.get("barrier_levels", []) if _crossed(spot_prev, spot_cur, lvl)
    ]
    if rotte:
        piu_vicina = min(rotte, key=lambda lvl: abs(lvl - (spot_cur or 0)))
        eventi.append(
            DeskEvent(
                key="barrier_breached",
                # più barriere attraversate insieme = evento più grave, fino a 1.0
                severity=min(1.0, 0.80 + 0.05 * len(rotte)),
                title=(
                    f"{fmt.count(len(rotte))} barriere bancarie attraversate"
                    if len(rotte) > 1
                    else f"Barriera attraversata a {fmt.price(piu_vicina)}"
                ),
                detail=(
                    f"Lo spot è passato da {fmt.price(spot_prev)} a {fmt.price(spot_cur)}, "
                    f"attraversando {fmt.count(len(rotte))} livelli depositati alla SEC."
                ),
                meta={"levels": rotte},
            )
        )

    # ── segnale composito che cambia lato ─────────────────────────────────────
    s_prev, s_cur = previous.get("signal_score"), current.get("signal_score")
    for soglia in _SIGNAL_THRESHOLDS:
        verso = _crossed(s_prev, s_cur, soglia)
        if verso:
            eventi.append(
                DeskEvent(
                    key=f"signal_crossed_{int(soglia)}",
                    severity=0.70,
                    title=f"Il segnale ha attraversato quota {fmt.count(soglia)}",
                    detail=(
                        f"Punteggio composito da {fmt.count(s_prev)} a {fmt.count(s_cur)}."
                    ),
                    meta={"threshold": soglia, "direction": "up" if verso > 0 else "down"},
                )
            )

    # ── flusso ETF fuori scala ────────────────────────────────────────────────
    f_cur = current.get("flow_3d_usd_m")
    f_prev = previous.get("flow_3d_usd_m")
    # solo se ha appena superato la soglia: un flusso che resta alto e' uno stato,
    # non una notizia nuova a ogni giro
    if (
        f_cur is not None
        and abs(f_cur) >= _FLOW_SPIKE_USD_M
        and (f_prev is None or abs(f_prev) < _FLOW_SPIKE_USD_M)
    ):
        eventi.append(
            DeskEvent(
                key="flow_spike",
                severity=0.65,
                title=f"Flusso ETF fuori scala: {fmt.usd_millions(f_cur * 1e6)}",
                detail=(
                    f"Il netto IBIT a tre giorni ha superato "
                    f"{fmt.usd_millions(_FLOW_SPIKE_USD_M * 1e6, force_sign=False)}."
                ),
                meta={"flow_3d_usd_m": f_cur},
            )
            )

    return sorted(eventi, key=lambda e: e.severity, reverse=True)


def should_publish(events: list[DeskEvent], *, threshold: float = PUBLISH_THRESHOLD) -> bool:
    """Vero se almeno un evento è abbastanza grave da far uscire un'edizione."""
    return any(e.severity >= threshold for e in events)


# ─── Persistenza ──────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS report_state (
    key        TEXT PRIMARY KEY,
    updated_at TEXT NOT NULL,          -- ISO UTC datetime
    payload    TEXT NOT NULL           -- JSON della fotografia
);
"""


class ReportStateDB:
    """Ultima fotografia pubblicata, per confrontarla col giro successivo.

    Segue la convenzione di AlertDB: rispetta ``DB_PATH`` via settings, così in
    sviluppo scrive su ``data/runtime.db`` e non sporca il seed versionato.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        cfg = get_settings()
        self._path = Path(db_path or cfg["database"]["path"])
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_table()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_table(self) -> None:
        with self._conn() as conn:
            conn.executescript(_DDL)

    def load(self, key: str = "desk_note") -> dict | None:
        """Ultima fotografia salvata, None se non c'è o è illeggibile."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT payload FROM report_state WHERE key = ?", (key,)
            ).fetchone()
        if not row:
            return None
        try:
            return json.loads(row["payload"])
        except (ValueError, TypeError) as exc:
            # una riga corrotta non deve impedire l'edizione: riparte da zero
            _log.warning("Fotografia %s illeggibile, la ignoro: %s", key, exc)
            return None

    def save(self, state: dict, key: str = "desk_note") -> None:
        """Sovrascrive la fotografia corrente."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO report_state (key, updated_at, payload) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET updated_at = excluded.updated_at, "
                "payload = excluded.payload",
                (key, datetime.now(timezone.utc).isoformat(), json.dumps(state)),
            )
