"""Fetch giornaliero dei flussi ETF da Farside — pensato per cron.

Scarica la pagina Farside, parsa i flussi, aggiorna la cache HTML su disco
e salva un CSV snapshot giornaliero in data/farside_YYYY-MM-DD.csv.

Uso manuale:
  python scripts/fetch_farside.py

Cron (ogni giorno alle 18:30 ora locale, Farside aggiorna ~18:00 UTC):
  30 18 * * 1-5  /path/to/.venv/bin/python /path/to/scripts/fetch_farside.py >> /path/to/logs/farside_cron.log 2>&1

Exit codes:
  0 = successo (dati scaricati e salvati)
  1 = fetch fallito (usata cache esistente o nessun dato)
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import setup_logging
from src.flows.scraper import FarsideScraper, _CACHE_FILE

_log = setup_logging("fetch_farside")


def main() -> int:
    _log.info("=== fetch_farside.py avviato ===")
    scraper = FarsideScraper()

    # Tenta il fetch live (curl_cffi bypassa Cloudflare)
    try:
        html  = scraper._fetch_html(scraper.FARSIDE_URL)
        flows = scraper._parse_table(html)
    except Exception as e:
        _log.error("Fetch Farside fallito: %s", e)
        return 1

    if not flows:
        _log.error("Nessun flusso parsato dalla pagina Farside")
        return 1

    # Aggiorna cache HTML
    scraper._write_cache(html)

    # Salva snapshot CSV giornaliero
    df      = scraper.to_dataframe(flows)
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f"farside_{date.today().isoformat()}.csv"
    df.to_csv(csv_path)

    _log.info(
        "OK: %d giorni di flussi salvati in %s (cache: %s)",
        len(df), csv_path, _CACHE_FILE,
    )

    # Stampa un summary a stdout per il log del cron
    if "IBIT" in df.columns:
        ibit_net = df["IBIT"].sum() / 1e9
        last_ibit = df["IBIT"].dropna().iloc[-1] / 1e6 if not df["IBIT"].dropna().empty else 0
        print(f"[OK] {date.today()} | giorni={len(df)} | IBIT net={ibit_net:.2f}B | ultimo giorno={last_ibit:.0f}M")

    return 0


if __name__ == "__main__":
    sys.exit(main())
