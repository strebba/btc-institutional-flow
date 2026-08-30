"""Esporta il Desk Note come carosello PNG 1080x1350 per LinkedIn e X.

Renderizza la pagina in Chromium headless e fotografa ogni card come elemento,
quindi i PNG escono alla geometria nativa senza scalature né ritagli. Non
richiede che l'API sia in ascolto: compone l'edizione in-process e la carica
come contenuto, così lo stesso script gira in locale, in CI e sul container DO.

Uso:
    python3 scripts/export_desk_note.py                    # -> out/desk-note/
    python3 scripts/export_desk_note.py --out /tmp/cards
    python3 scripts/export_desk_note.py --url https://.../report?export=true
    python3 scripts/export_desk_note.py --only-on-event    # esce se non è notizia

Variabili d'ambiente:
    PLAYWRIGHT_BROWSERS_PATH  Già configurata negli ambienti che hanno Chromium
                              preinstallato; altrimenti serve `playwright install`.

Exit code:
    0 — PNG esportati (o nessun evento con --only-on-event)
    1 — rendering fallito o Playwright non disponibile
    2 — l'edizione è vuota: i dati non bastano a comporre nessuna card
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import setup_logging

_log = setup_logging("export_desk_note")

#: Geometria del viewport: larga quanto una card, alta a sufficienza per
#: contenerla senza che il layout si comprima.
_VIEWPORT = {"width": 1080, "height": 1400}


def _compose() -> tuple[str, int]:
    """Compone l'edizione in-process e la rende in HTML pronto per l'export.

    Returns:
        (html, numero di card). Zero card significa dati insufficienti.
    """
    from src.api.routers.report import _gather
    from src.report.narrative import build_desk_note
    from src.report.renderer import render_html

    fonti = _gather()
    note = build_desk_note(
        gex=fonti["gex"],
        barriers=fonti["barriers"],
        flows=fonti["flows"],
        signals=fonti["signals"],
        forecast=fonti["forecast"],
        macro=fonti["macro"],
    )
    for avviso in note.warnings:
        _log.warning("Dati incompleti: %s", avviso)
    return render_html(note, export=True), len(note.cards)


def _has_event() -> bool:
    """Vero se qualcosa è cambiato abbastanza da giustificare un'edizione."""
    from src.api.routers.report import _gather
    from src.report.events import ReportStateDB, detect_events, should_publish, snapshot_state

    fonti = _gather()
    corrente = snapshot_state(
        gex=fonti["gex"], barriers=fonti["barriers"], signals=fonti["signals"]
    )
    eventi = detect_events(corrente, ReportStateDB().load())
    for e in eventi:
        _log.info("Evento %s (%.2f): %s", e.key, e.severity, e.title)
    return should_publish(eventi)


def _launch(playwright):
    """Avvia Chromium, aggirando i disallineamenti di versione dei browser.

    Quando la versione di Playwright installata si aspetta una build di Chromium
    diversa da quella presente, il lancio predefinito fallisce anche se un
    browser perfettamente utilizzabile è lì. In quel caso si punta all'eseguibile
    presente invece di scaricarne un altro: gli ambienti CI e il container DO
    hanno il browser preinstallato e la rete verso il CDN spesso chiusa.
    """
    try:
        return playwright.chromium.launch()
    except Exception as exc:  # noqa: BLE001 — riprovo con un eseguibile esplicito
        _log.warning("Lancio predefinito fallito (%s), cerco un Chromium installato", exc)

    import os

    radice = Path(os.getenv("PLAYWRIGHT_BROWSERS_PATH", "/opt/pw-browsers"))
    candidati = sorted(radice.glob("chromium-*/chrome-linux/chrome"), reverse=True)
    candidati += sorted(
        radice.glob("chromium_headless_shell-*/chrome-linux/chrome-headless-shell"), reverse=True
    )
    for exe in candidati:
        if exe.exists():
            _log.info("Uso il Chromium in %s", exe)
            return playwright.chromium.launch(executable_path=str(exe))

    raise RuntimeError(
        f"Nessun Chromium utilizzabile sotto {radice}. Esegui 'playwright install chromium'."
    )


def _shoot(page, out_dir: Path) -> list[Path]:
    """Fotografa ogni card come elemento e restituisce i file scritti."""
    page.wait_for_selector(".card", timeout=15_000)
    # i webfont cambiano la metrica del testo: senza attenderli i PNG escono
    # con il fallback di sistema e un'interlinea diversa dalla pagina web
    try:
        page.evaluate("document.fonts && document.fonts.ready")
        page.wait_for_timeout(600)
    except Exception as exc:  # noqa: BLE001 — senza webfont si esporta comunque
        _log.warning("Attesa webfont fallita, uso i font di sistema: %s", exc)

    troncate = page.evaluate(
        "Array.from(document.querySelectorAll('.card'))"
        ".map((c,i) => c.scrollHeight > c.clientHeight + 1 ? i + 1 : 0)"
        ".filter(Boolean)"
    )

    cards = page.query_selector_all(".card")
    scritti: list[Path] = []
    for i, card in enumerate(cards, start=1):
        path = out_dir / f"desk-note-{i:02d}.png"
        card.screenshot(path=str(path))
        scritti.append(path)
        _log.info("Scritto %s%s", path, "  ⚠ TESTO TRONCATO" if i in troncate else "")

    if troncate:
        # una card tagliata pubblicata non si ritira: va vista prima, non dopo
        _log.error(
            "Card con testo troncato: %s. Accorcia la copy in src/report/facts.py "
            "prima di pubblicare.",
            ", ".join(str(i) for i in troncate),
        )
    return scritti


def main() -> int:
    parser = argparse.ArgumentParser(description="Esporta il Desk Note come PNG 1080x1350.")
    parser.add_argument("--out", default="out/desk-note", help="Cartella di destinazione.")
    parser.add_argument(
        "--url",
        default=None,
        help="Renderizza da un'istanza in ascolto invece che in-process.",
    )
    parser.add_argument(
        "--only-on-event",
        action="store_true",
        help="Esporta solo se il rilevamento eventi dice che è notizia.",
    )
    args = parser.parse_args()

    if args.only_on_event and not _has_event():
        _log.info("Nessun evento sopra soglia: niente da pubblicare.")
        return 0

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        _log.error("Playwright non installato: pip install playwright")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    html = None
    if not args.url:
        html, n_cards = _compose()
        if n_cards == 0:
            _log.error("Edizione vuota: i dati non bastano a comporre nessuna card.")
            return 2

    try:
        with sync_playwright() as p:
            browser = _launch(p)
            # deviceScaleFactor 1: la card è già 1080px, non serve raddoppiare
            page = browser.new_page(viewport=_VIEWPORT, device_scale_factor=1)
            if args.url:
                page.goto(args.url, wait_until="networkidle", timeout=45_000)
            else:
                page.set_content(html, wait_until="networkidle", timeout=45_000)
            scritti = _shoot(page, out_dir)
            browser.close()
    except Exception as exc:  # noqa: BLE001 — l'export è un job, non un servizio
        _log.error("Export fallito: %s", exc)
        return 1

    if not scritti:
        _log.error("Nessuna card trovata nella pagina.")
        return 2

    _log.info("Esportate %d card in %s", len(scritti), out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
