"""Ri-parsa le note EDGAR e confronta il risultato col DB attuale.

La correzione all'estrazione del testo (`_INLINE_TAGS` sciolti prima di
`get_text`) cambia l'input di **tutti** i parser, non solo di quello del
nozionale. Non basta guardare quanti valori si guadagnano: bisogna verificare
che non se ne perdano, e che i livelli di barriera già estratti non cambino —
un livello sbagliato è peggio di un livello mancante, perché finisce in una card
pubblicata.

Lo script non tocca mai il DB versionato: scrive su una copia e produce un
report. La sostituzione è una decisione umana, presa leggendo il report.

Uso:
    python3 scripts/reparse_notional.py --limit 40      # campione, per iniziare
    python3 scripts/reparse_notional.py                 # tutte le note
    python3 scripts/reparse_notional.py --out /tmp/nuovo.db

Exit code:
    0 — nessuna regressione: si può procedere alla sostituzione
    1 — errore di esecuzione
    2 — regressioni rilevate: NON sostituire il DB senza averle ispezionate
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import setup_logging
from src.edgar.parser import ProspectusParser

_log = setup_logging("reparse_notional")

_DB = Path(__file__).resolve().parent.parent / "data" / "structured_notes.db"

#: Emittente di controllo: oggi funziona all'87%, quindi qualunque perdita qui
#: significa che la correzione ha rotto qualcosa che andava bene.
_CONTROLLO = "Morgan Stanley"

#: Fuori da questo intervallo un livello percentuale non è credibile per una nota
#: strutturata su ETF: sono i valori che il testo spezzato produceva leggendo
#: "50% per annum deduction" come barriera, o troncando "103.75%" in "3.75".
_LIVELLO_MIN_PCT = 10.0
_LIVELLO_MAX_PCT = 300.0


def _plausibile(livello_pct: float | None) -> bool:
    """Vero se il livello sta nell'intervallo credibile per una nota su ETF."""
    return livello_pct is not None and _LIVELLO_MIN_PCT <= livello_pct <= _LIVELLO_MAX_PCT


def _leggi_note(conn: sqlite3.Connection, limit: int | None) -> list[dict]:
    conn.row_factory = sqlite3.Row
    sql = (
        "SELECT id, filing_url, issuer, notional_usd, initial_level, is_preliminary "
        "FROM notes ORDER BY id"
    )
    if limit:
        sql += f" LIMIT {int(limit)}"
    return [dict(r) for r in conn.execute(sql)]


def _barriere_di(conn: sqlite3.Connection, note_id: int) -> list[tuple[str, float]]:
    conn.row_factory = sqlite3.Row
    return [
        (r["barrier_type"], r["level_pct"])
        for r in conn.execute(
            "SELECT barrier_type, level_pct FROM barrier_levels "
            "WHERE note_id = ? ORDER BY barrier_type, level_pct",
            (note_id,),
        )
    ]


class Report:
    """Accumula il confronto campo per campo, spaccato per emittente."""

    def __init__(self) -> None:
        self.per_issuer: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        #: Barriere plausibili sparite: sono le uniche vere candidate a regressione.
        self.rimozioni_sospette: list[str] = []
        #: Barriere nuove fuori intervallo: il parser starebbe inventando.
        self.aggiunte_implausibili: list[str] = []
        #: Note che hanno perso initial_level o notional: da ispezionare come le barriere.
        self.valori_persi: list[str] = []
        self.errori: list[str] = []

    def conta(self, issuer: str, chiave: str) -> None:
        self.per_issuer[issuer][chiave] += 1

    # ── regressioni ───────────────────────────────────────────────────────────

    def perdite_controllo(self) -> int:
        c = self.per_issuer.get(_CONTROLLO, {})
        return c.get("notional_perso", 0) + c.get("initial_perso", 0)

    def ha_regressioni(self) -> bool:
        """Cosa conta come regressione.

        Non il numero di barriere in meno: togliere livelli fantasma e' lo scopo
        della correzione. Conta perdere valori sul gruppo di controllo, far
        sparire barriere plausibili, o introdurne di implausibili.
        """
        return bool(
            self.perdite_controllo()
            or self.rimozioni_sospette
            or self.aggiunte_implausibili
        )

    # ── stampa ────────────────────────────────────────────────────────────────

    def stampa(self) -> None:
        colonne = [
            ("note", "note"),
            ("notional_guadagnato", "+notional"),
            ("notional_perso", "-notional"),
            ("notional_cambiato", "~notional"),
            ("initial_guadagnato", "+initial"),
            ("initial_perso", "-initial"),
            ("barriere_aggiunte", "+barriere"),
            ("barriere_rimosse", "-barriere"),
            ("fantasmi_rimossi", "-fantasmi"),
        ]
        print()
        print("REPORT DI REGRESSIONE — ri-parse contro il DB attuale")
        print("=" * 100)
        intest = f"{'emittente':<22}" + "".join(f"{e:>11}" for _, e in colonne)
        print(intest)
        print("-" * 100)

        totali: dict[str, int] = defaultdict(int)
        for issuer in sorted(self.per_issuer, key=lambda i: -self.per_issuer[i]["note"]):
            v = self.per_issuer[issuer]
            marchio = " *" if issuer == _CONTROLLO else "  "
            riga = f"{issuer[:20]:<20}{marchio}"
            for chiave, _ in colonne:
                riga += f"{v.get(chiave, 0):>11}"
                totali[chiave] += v.get(chiave, 0)
            print(riga)
        print("-" * 100)
        riga = f"{'TOTALE':<22}"
        for chiave, _ in colonne:
            riga += f"{totali[chiave]:>11}"
        print(riga)
        print(f"\n  * {_CONTROLLO} e' il gruppo di controllo: qualunque perdita qui")
        print("    significa che la correzione ha rotto qualcosa che funzionava.")

        if self.rimozioni_sospette:
            print(f"\n  BARRIERE PLAUSIBILI SPARITE ({len(self.rimozioni_sospette)}) — da ispezionare:")
            for riga in self.rimozioni_sospette[:20]:
                print(f"    {riga}")
            if len(self.rimozioni_sospette) > 20:
                print(f"    ... e altre {len(self.rimozioni_sospette) - 20}")

        if self.valori_persi:
            print(f"\n  VALORI SCALARI PERSI ({len(self.valori_persi)}) — da ispezionare:")
            for riga in self.valori_persi[:20]:
                print(f"    {riga}")
            if len(self.valori_persi) > 20:
                print(f"    ... e altri {len(self.valori_persi) - 20}")

        if self.aggiunte_implausibili:
            print(f"\n  BARRIERE NUOVE FUORI INTERVALLO ({len(self.aggiunte_implausibili)}):")
            for riga in self.aggiunte_implausibili[:20]:
                print(f"    {riga}")

        if self.errori:
            print(f"\n  ERRORI DI PARSING ({len(self.errori)}):")
            for riga in self.errori[:10]:
                print(f"    {riga}")

        print()
        if self.ha_regressioni():
            print("  ESITO: REGRESSIONI RILEVATE — non sostituire il DB.")
            if self.perdite_controllo():
                print(f"    {self.perdite_controllo()} valori persi sul gruppo di controllo")
            if self.rimozioni_sospette:
                print(f"    {len(self.rimozioni_sospette)} barriere plausibili sparite")
            if self.aggiunte_implausibili:
                print(f"    {len(self.aggiunte_implausibili)} barriere nuove fuori intervallo")
        else:
            print("  ESITO: nessuna regressione. Il DB rigenerato si puo' sostituire.")
        print()


def main() -> int:
    parser_args = argparse.ArgumentParser(
        description="Ri-parsa le note EDGAR e confronta col DB attuale."
    )
    parser_args.add_argument(
        "--limit", type=int, default=None,
        help="Ri-parsa solo le prime N note (campione, per una verifica veloce).",
    )
    parser_args.add_argument(
        "--out", default=None,
        help="Percorso del DB rigenerato (default: accanto all'originale, .reparse.db).",
    )
    args = parser_args.parse_args()

    if not _DB.exists():
        _log.error("DB non trovato: %s", _DB)
        return 1

    out_path = Path(args.out) if args.out else _DB.with_suffix(".reparse.db")
    shutil.copy2(_DB, out_path)
    _log.info("Copia di lavoro: %s", out_path)

    src = sqlite3.connect(_DB)
    dst = sqlite3.connect(out_path)
    note = _leggi_note(src, args.limit)
    _log.info("Ri-parsing di %d note (rate-limited verso la SEC)...", len(note))

    parser = ProspectusParser()
    report = Report()

    for i, riga in enumerate(note, start=1):
        issuer = riga["issuer"] or "n/d"
        report.conta(issuer, "note")
        if i % 25 == 0:
            _log.info("  %d/%d", i, len(note))

        try:
            nuova = parser.parse({"url": riga["filing_url"]})
        except Exception as exc:  # noqa: BLE001 — un filing rotto non ferma il giro
            report.errori.append(f"{issuer} {riga['filing_url'][-40:]}: {exc}")
            continue

        # ── notional ──────────────────────────────────────────────────────────
        vecchio_n, nuovo_n = riga["notional_usd"], nuova.notional_usd
        if vecchio_n is None and nuovo_n is not None:
            report.conta(issuer, "notional_guadagnato")
        elif vecchio_n is not None and nuovo_n is None:
            report.conta(issuer, "notional_perso")
            report.valori_persi.append(
                f"{issuer} nota {riga['id']} notional {vecchio_n} -> None"
            )
        elif vecchio_n is not None and nuovo_n is not None and vecchio_n != nuovo_n:
            report.conta(issuer, "notional_cambiato")

        # ── initial_level ─────────────────────────────────────────────────────
        vecchio_i, nuovo_i = riga["initial_level"], nuova.initial_level
        if vecchio_i is None and nuovo_i is not None:
            report.conta(issuer, "initial_guadagnato")
        elif vecchio_i is not None and nuovo_i is None:
            report.conta(issuer, "initial_perso")
            report.valori_persi.append(
                f"{issuer} nota {riga['id']} initial_level {vecchio_i} -> None"
            )

        # ── barriere ──────────────────────────────────────────────────────────
        # Confronto insiemistico, non posizionale: appaiare due liste ordinate di
        # lunghezza diversa accosta elementi non correlati e produce "livelli
        # cambiati" che non sono cambiamenti. Interessa cosa è comparso e cosa è
        # sparito, non l'allineamento fra le due liste.
        vecchie = Counter(_barriere_di(src, riga["id"]))
        nuove = Counter((b.barrier_type, b.level_pct) for b in nuova.barriers)

        aggiunte = nuove - vecchie
        rimosse = vecchie - nuove
        report.per_issuer[issuer]["barriere_aggiunte"] += sum(aggiunte.values())
        report.per_issuer[issuer]["barriere_rimosse"] += sum(rimosse.values())

        # Un livello rimosso è una regressione solo se era plausibile: togliere
        # una barriera fantasma è il motivo per cui questa correzione esiste.
        for (tipo, lvl), n in rimosse.items():
            if _plausibile(lvl):
                report.rimozioni_sospette.extend(
                    [f"{issuer} nota {riga['id']} {tipo} {lvl}"] * n
                )
            else:
                report.per_issuer[issuer]["fantasmi_rimossi"] += n

        for (tipo, lvl), n in aggiunte.items():
            if not _plausibile(lvl):
                report.aggiunte_implausibili.extend(
                    [f"{issuer} nota {riga['id']} {tipo} {lvl}"] * n
                )

        # scrive nella copia, mai nell'originale
        dst.execute(
            "UPDATE notes SET notional_usd = ?, initial_level = ? WHERE id = ?",
            (nuovo_n, nuovo_i, riga["id"]),
        )

    dst.commit()
    src.close()
    dst.close()

    report.stampa()
    print(f"  DB rigenerato: {out_path}")
    print(f"  L'originale ({_DB.name}) non e' stato toccato.\n")

    return 2 if report.ha_regressioni() else 0


if __name__ == "__main__":
    sys.exit(main())
