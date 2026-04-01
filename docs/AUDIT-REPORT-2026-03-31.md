# AUDIT REPORT — ibit-gamma-tracker
**Data:** 2026-03-31  
**Classificazione:** PROJ-AUDIT-ADV-v1.0  
**Eseguito da:** Claude Code (claude-sonnet-4-6)  

---

## Executive Summary

Il progetto è architetturalmente solido: la pipeline EDGAR→DB, la catena waterfall dei flussi ETF e il motore GEX/Deribit sono ben strutturati. Il layer test copre 185 casi e passa interamente (esclusi 3 file con dipendenze non installate). L'API FastAPI e la dashboard Streamlit sono operative.

**Blocker principale confermato**: nessuna tabella `gex_snapshots` esiste nel DB. Ogni esecuzione di backtest e regime analysis riceve un solo punto GEX (oggi) oppure `None`, rendendo la strategia sistematicamente **FLAT** nel backtest e il percentile GEX non persistente tra restart. Finché questo non viene risolto, il sistema rimane un tool di monitoraggio live senza capacità di segnale storico.

**Problemi secondari**:
- 3 dipendenze critiche (`beautifulsoup4`, `yfinance`, `curl_cffi`) non installate nell'ambiente di test → 3 file di test non collezionabili.
- Farside cache HTML datata 13 mar 2026 (18 giorni), CSV datata 5 mar 2026 (26 giorni) → flussi live probabilmente degradati a stima yfinance.
- `run_backtest()` in `app.py` non passa `gex_series` a `bt.run()` (discordanza con `scripts/run_analytics.py` che lo fa).
- 6 barriere attive su 10 prive di `level_price_btc` → non geolocalizzabili rispetto al prezzo BTC corrente.

---

## Fase 1: Inventario e Stato del Codice

### 1.1 Struttura file

| File | Righe | Ruolo principale |
|---|---|---|
| `src/api/main.py` | 442 | FastAPI: 4 endpoint (`/gex`, `/flows`, `/barriers`, `/signals`) + cache TTL |
| `src/dashboard/app.py` | 1266 | Streamlit: 5 tab + composite signal + tutti i loader |
| `src/dashboard/charts.py` | 538 | Funzioni Plotly per ogni chart |
| `src/analytics/backtest.py` | 377 | `Backtest.run()`, `_generate_signals()`, `BacktestMetrics` |
| `src/analytics/regime_analysis.py` | 289 | `RegimeAnalysis.analyze()`, `RegimeComparisonResult` |
| `src/analytics/event_study.py` | 379 | `EventStudy.run()`: analisi price action attorno ai barrier level |
| `src/analytics/granger.py` | ~150 | `GrangerCausality`: test flussi→prezzi |
| `src/gex/deribit_client.py` | ~200 | `DeribitClient`: fetch opzioni + spot Deribit |
| `src/gex/gex_calculator.py` | ~280 | `GexCalculator.calculate_gex()`, `gex_to_dict()` |
| `src/gex/regime_detector.py` | ~150 | `RegimeDetector.detect()`: classifica regime + alert |
| `src/gex/models.py` | ~80 | `GexSnapshot`, `GexByStrike` dataclass |
| `src/flows/scraper.py` | 529 | `FarsideScraper.fetch()`: waterfall Farside→cache→SoSoValue→EDGAR→yfinance |
| `src/flows/price_fetcher.py` | 358 | `PriceFetcher`: yfinance + SQLite store |
| `src/flows/correlation.py` | 294 | `FlowCorrelation.merge()`: join flussi + prezzi |
| `src/flows/edgar_nport.py` | 357 | `EdgarNportClient`: fallback N-PORT SEC |
| `src/flows/sosovalue.py` | ~150 | `SoSoValueClient`: API SoSoValue |
| `src/edgar/structured_notes_db.py` | 420 | `StructuredNotesDB`: CRUD note + barriere + `compute_btc_prices()` |
| `src/edgar/parser.py` | 536 | `NoteParser`: parsing HTML/XML note strutturate |
| `src/edgar/search.py` | 394 | `EdgarSearch`: ricerca filing SEC |
| `src/edgar/models.py` | ~80 | `StructuredNote`, `BarrierLevel` dataclass |
| `src/config.py` | ~60 | `get_settings()`, `setup_logging()` |

**Script:**
- `scripts/run_analytics.py` — pipeline completa con `_load_gex_series()` (fetch live Deribit) + passa `gex_series` al backtest ✅
- `scripts/run_edgar.py` — fetch + parse EDGAR, chiama `compute_btc_prices()`
- `scripts/run_gex.py` — fetch GEX live e stampa, ma **non salva nel DB** ❌
- `scripts/run_flows.py` — aggiorna prezzi + flussi nel DB
- `scripts/fetch_farside.py` — aggiorna cache Farside HTML

### 1.2 Stato dei test

```
185 passed, 23 warnings (3 file non collezionabili per import error)
```

| Stato | Numero |
|---|---|
| Passati | 185 |
| Falliti | 0 |
| Saltati | 0 |
| Errori di collection | 3 file |

**File non collezionabili** per dipendenze mancanti nell'ambiente system Python 3.9:
- `tests/test_edgar/test_parser.py` → `No module named 'bs4'` (beautifulsoup4)
- `tests/test_flows/test_price_fetcher.py` → `No module named 'yfinance'`
- `tests/test_flows/test_scraper.py` → `No module named 'bs4'`

**Test mancanti per moduli critici:**
- `tests/test_gex/test_gex_db.py` — NON ESISTE (il modulo non esiste ancora)
- `tests/test_analytics/test_backtest.py` — esiste, ma non testa il caso `gex_series=None` → strategia flat
- Nessun test di integrazione per la pipeline EDGAR→DB→API

### 1.3 Dipendenze

**Dipendenze non installate nell'ambiente di test** (`/Library/Python/3.9`):
- `beautifulsoup4` / `bs4` — necessaria per parser EDGAR e Farside scraper
- `yfinance` — necessaria per PriceFetcher e fallback flows
- `curl_cffi` — necessaria per bypass Cloudflare su Farside

**Nota**: il progetto gira correttamente in un virtual environment; il problema è che i test vengono eseguiti con il Python di sistema che non ha il venv attivato. Il resto del codice (fastapi, pandas, numpy, scipy, statsmodels) è installato nel sistema e i 185 test passano.

**Versioni pinned**: floating (`>=`) per tutte le dipendenze — accettabile per un progetto di ricerca, ma rischio di breaking changes con major version bumps.

---

## Fase 2: Verifica Persistenza Dati GEX (CRITICA)

### 2.1 Schema DB — BLOCKER CONFERMATO

**Tabelle esistenti:**
```
barrier_levels  notes  prices
```

**La tabella `gex_snapshots` NON ESISTE.** → **BLOCKER CRITICO**

### 2.2 Flusso salvataggio GEX

```
grep -rn "gex_snapshot|save_gex|store_gex|insert.*gex|INSERT.*gex" src/ scripts/
# Nessun risultato

grep -rn "gex_snapshots|gex_history|gex_daily" src/ scripts/
# Nessun risultato
```

Il GEX non viene mai scritto su disco. Il flusso completo è:

```
DeribitClient.fetch_all_options()
  → GexCalculator.calculate_gex() → GexSnapshot (in-memory)
  → RegimeDetector.detect() → RegimeState (in-memory)
  → gex_to_dict() → dict serializzato
  → _cache_set("gex", ...) → dict in cache in-process (TTL 5 min)
```

Nessun passaggio persiste nel DB.

### 2.3 Backtest: gex_series

**`scripts/run_analytics.py`** (CLI): ✅ funziona correttamente
```python
gex_series = _load_gex_series()   # fetch live da Deribit → pd.Series({today: snap.total_net_gex})
...
run_backtest(merged_df, gex_series, barriers)  # passa gex_series
```

**`src/dashboard/app.py`** (Streamlit): ❌ bug
```python
def run_backtest(merged_df, barriers):
    bt = Backtest()
    return bt, bt.run(merged_df, active_barriers=barriers if barriers else None)
    # gex_series NON passato → default None → _gex = 0.0 → strategia sempre FLAT
```

**`src/api/main.py`** (API `/api/signals`): ❌ bug identico
```python
results = bt.run(merged, active_barriers=active_barriers)
# gex_series NON passato
```

**Impatto nella `_generate_signals()`** (`backtest.py:94-101`):
```python
if gex_series is not None and not gex_series.empty:
    df = df.join(gex_series.rename("_gex"), how="left")
elif "total_gex" in df.columns:
    df["_gex"] = df["total_gex"]
else:
    df["_gex"] = 0.0  # ← SEMPRE questo ramo in dashboard e API
```

Con `_gex = 0.0` per tutti i giorni, la condizione `gex > long_gex_th` (threshold = 0) è **falsa per default** e la strategia non genera mai LONG → equity curve piatta.

### 2.4 Gap Analysis

| Componente | Dato necessario | Fonte attuale | Persistenza | Gap |
|---|---|---|---|---|
| Backtest storico (script) | GEX daily series | Deribit live (1 punto oggi) | None — solo in memoria durante run | Solo oggi, non storico |
| Backtest storico (dashboard) | GEX daily series | — | — | **Bug: gex_series non passato** |
| Backtest storico (API) | GEX daily series | — | — | **Bug: gex_series non passato** |
| Regime Analysis (script) | GEX daily series | Deribit live (1 punto) | None | Solo oggi |
| Regime Analysis (dashboard) | GEX daily series | Deribit live (1 punto) | In-memory cache 5 min | Perde a ogni restart |
| GEX percentile | Storico GEX | In-memory `RegimeDetector._history` | **Solo sessione corrente** | Perde a ogni restart |
| Segnale composito (dashboard) | GEX live | Deribit live | Cache 5 min | OK per segnale corrente |
| Segnale composito (API) | GEX live | Deribit live | Cache 5 min | OK per segnale corrente |
| Event Study | BTC prices + barriers | SQLite | ✅ OK | — |
| Granger Causality | Flows + Returns | SQLite + yfinance | ✅ OK | — |
| Barriere | level_price_btc | SQLite | 6/10 NULL | Conversione parziale |

---

## Fase 3: Flussi ETF e Prezzi

### 3.1 DB Prezzi

```
BTC-USD | 373 righe | 2025-03-03 → 2026-03-12
IBIT    | 259 righe | 2025-03-03 → 2026-03-12
```

Dati aggiornati al 2026-03-12 (19 giorni di ritardo rispetto all'audit — mancano ~14 giorni lavorativi).

### 3.2 Qualità dati flussi

**Cache Farside:**
- `data/farside_cache.html` — 630KB, datata **2026-03-13** (18 giorni fa)
- `data/farside_2026-03-05.csv` — datata **2026-03-05** (26 giorni fa)

**Waterfall configurata** in `FarsideScraper.fetch()`:
1. Farside HTML live (curl_cffi, bypass Cloudflare)
2. Cache disco HTML
3. SoSoValue API (richiede API key)
4. EDGAR N-PORT
5. yfinance tracking-error estimate (bassa qualità — avviso esplicito nel log)

Con `curl_cffi` non installato nel sistema e la cache HTML vecchia di 18 giorni, è probabile che la sorgente primaria fallisca e il sistema degradi a stima yfinance. Flussi recenti (ultimi 18 giorni) sono pertanto stime, non dati reali.

### 3.3 Merge data quality

Senza poter eseguire il codice (dipendenze mancanti nell'ambiente di sistema), si nota che la finestra temporale dei prezzi (fino al 2026-03-12) e della cache Farside (2026-03-13) è sostanzialmente allineata, suggerendo che l'ultimo aggiornamento sia avvenuto attorno al 13 marzo.

---

## Fase 4: EDGAR Pipeline

### 4.1 DB Note Strutturate

```
notes:         8 totali
  JPMorgan:    7
  Morgan Stanley: 1
  (NULL issuer): 2  ← potenziale parsing error

product_type:
  autocallable:   3
  barrier_note:   1
  buffered_note:  1
  leveraged_note: 1
  (NULL):         2  ← parsing fallito/incompleto

barrier_levels:
  10 totali, 10 active
  knock_in: 8
  autocall:  2

  level_price_btc NOT NULL: 4
  level_price_btc IS NULL:  6  ← 60% delle barriere senza prezzo BTC
```

**Nota critica**: 2 note con `issuer = NULL` e `product_type = NULL` indicano parsing fallito — i record sono presenti ma incompleti. Possibile causa: varianti di formato HTML non gestite dal parser.

### 4.2 Conversione prezzi BTC

`compute_btc_prices()` (`structured_notes_db.py:262`) aggiorna le barriere con `level_price_btc = level_price_ibit / ibit_btc_ratio`.

**Chiamata automatica**: sì, nell'endpoint `/api/barriers` (`main.py:286`) viene chiamata ad ogni fetch (non in cache), usando il ratio IBIT/BTC live da `PriceFetcher.get_ibit_btc_ratio()`. Questo significa che le 6 barriere NULL vengono aggiornate a ogni chiamata API se `level_price_ibit` è disponibile.

**Ipotesi 6 NULL persistenti**: probabilmente queste barriere hanno anche `level_price_ibit = NULL` (parsing fallito), quindi `compute_btc_prices()` non può calcolare il prezzo BTC.

---

## Fase 5: API e Dashboard

### 5.1 Dashboard Streamlit

- 5 tab: GEX, Flows, Barriers, Backtest, Regime Analysis
- `_REFRESH` = 900s (15 min) da `config/settings.yaml`
- Tutti i loader usano `@st.cache_data(ttl=_REFRESH)`
- **Bug identificato**: `run_backtest()` non passa `gex_series` (vedi Fase 2)
- Il composite signal `_composite_signal()` usa GEX live (corretto per segnale corrente)

### 5.2 API FastAPI

Endpoint disponibili:
- `GET /api/health` — health check
- `GET /api/gex` — snapshot GEX live Deribit (cache 5 min)
- `GET /api/flows` — flussi ETF + correlazioni (cache 15 min)
- `GET /api/barriers` — barriere attive con `compute_btc_prices()` dinamico (cache 1h)
- `GET /api/signals` — segnale composito + backtest (cache 5 min)

**Bug in `/api/signals`**: backtest chiamato senza `gex_series` → strategia sempre flat (stesso bug della dashboard).

### 5.3 Cache TTL

| Endpoint | TTL | Ragionevolezza |
|---|---|---|
| `gex` | 300s (5 min) | OK — fetch Deribit dura ~90s |
| `flows` | 900s (15 min) | OK — Farside non aggiorna più spesso |
| `barriers` | 3600s (1h) | OK — dati SEC statici |
| `signals` | 300s (5 min) | OK — dipende da gex + flows |
| Streamlit | 900s (15 min) | OK — allineato con flows |

**Race condition potenziale**: se `/api/signals` scade prima di `/api/gex`, il segnale viene ricalcolato con un GEX più vecchio rispetto al cache GEX. Impatto minimo in pratica.

---

## Issue Register

| # | Severità | Categoria | Descrizione | File/Linea | Impatto | Fix suggerito | Effort |
|---|---|---|---|---|---|---|---|
| 1 | **BLOCKER** | Data | Nessuna tabella `gex_snapshots` nel DB — storico GEX non persiste | DB schema | Backtest sempre flat; percentile GEX perso a ogni restart | Creare tabella + `src/gex/gex_db.py` + cron save | 4h |
| 2 | **BLOCKER** | Logic | `run_backtest()` in `app.py` non passa `gex_series` → `_gex=0.0` per tutti i giorni | `app.py:324` | Equity curve sempre piatta in dashboard | Passare GEX live come `pd.Series({today: gex})` | 30 min |
| 3 | **BLOCKER** | Logic | `/api/signals` non passa `gex_series` a `bt.run()` | `main.py:409` | Stesso impatto del #2 nell'API | Stesso fix del #2 | 30 min |
| 4 | **MAJOR** | Data | 6/10 barriere attive con `level_price_btc = NULL` | DB `barrier_levels` | Barriere non localizzabili su grafico prezzo BTC | Debug parser EDGAR per estrarre `level_price_ibit`; fallback su `initial_level * level_pct / 100` | 2h |
| 5 | **MAJOR** | Data | 2 note con `issuer = NULL` e `product_type = NULL` | DB `notes` | Dati incompleti; possibile undercount barriere attive | Debug `NoteParser` con filing specifici; aggiungere fallback regex | 2h |
| 6 | **MAJOR** | Infra | Cache Farside HTML datata 18 giorni; CSV datata 26 giorni — flussi recenti probabilmente da stima yfinance | `data/farside_cache.html` | Qualità flussi degradata per ultime 2-3 settimane | Schedulare `scripts/fetch_farside.py` giornalmente (cron/DO scheduler) | 1h |
| 7 | **MAJOR** | Infra | Prezzi DB fermi al 2026-03-12 (19 giorni fa) | `prices` table | Correlazioni e backtest su finestra troncata | Schedulare `scripts/run_flows.py` giornalmente | 30 min |
| 8 | **MAJOR** | Test | 3 file di test non collezionabili per mancanza dipendenze nel Python di sistema | `tests/test_edgar/test_parser.py` etc. | CI potrebbe sembrare verde ma non testa parser EDGAR e scraper | Attivare venv nel CI / aggiungere `pip install -r requirements.txt` nel Makefile | 1h |
| 9 | **MINOR** | Test | Nessun test per `gex_db.py` (da creare) e per il salvataggio GEX | — | Nessun safety net per la persistenza GEX | Creare `tests/test_gex/test_gex_db.py` con ≥10 test | 2h |
| 10 | **MINOR** | Logic | `run_analytics.py:_load_gex_series()` fa un fetch Deribit live (90s) a ogni run — non usa cache | `scripts/run_analytics.py:56` | Ogni run analytics è lento; impossibile testare con dati storici | Dopo la creazione di `gex_snapshots`, leggere da DB con fallback a live | 1h |
| 11 | **MINOR** | Data | `RegimeDetector._history` (lista in-memory) non persiste tra restart | `src/gex/regime_detector.py:44` | Percentile GEX calcolato su sola sessione corrente, non su storico | Dopo #1: caricare storico da `gex_snapshots` al boot | 1h |
| 12 | **MINOR** | UX | `app.py:243` ha un commento che spiega `compute_btc_prices()` come operazione manuale, ma è automatica in API | `app.py:243` | Documentazione interna fuorviante | Aggiornare commento | 10 min |
| 13 | **COSMETIC** | Test | Dipendenze test (`pytest`, `pytest-cov`, `responses`) elencate in `[project.dependencies]` in `pyproject.toml` — dovrebbero essere in `[project.optional-dependencies]` | `pyproject.toml` | Dipendenze test installate in produzione | Spostare in `[project.optional-dependencies].dev` | 20 min |

---

## Piano d'Azione

### BLOCCO A — Persistenza GEX (prerequisito per tutto il resto)

**Obiettivo**: il sistema accumula uno snapshot GEX giornaliero nel DB, usato da backtest e regime analysis.

1. **Creare tabella `gex_snapshots`** — migrare lo schema (vedi Appendice B)
2. **Creare `src/gex/gex_db.py`** — `GexDB` con `save_snapshot()` e `load_series()`
3. **Modificare `scripts/run_gex.py`** — aggiungere chiamata `GexDB().save_snapshot(snap)` dopo il calcolo
4. **Creare `scripts/cron_gex.py`** — wrapper schedulabile (vedi Appendice C)
5. **Fix bug #2**: modificare `run_backtest()` in `app.py` per caricare serie da DB
6. **Fix bug #3**: modificare `/api/signals` in `main.py` per caricare serie da DB
7. **Aggiornare `RegimeDetector`** — caricare storico da DB al boot (`GexDB().load_series()`)
8. **Aggiornare `scripts/run_analytics.py`** — `_load_gex_series()` legge da DB + 1 punto live oggi
9. **Scrivere `tests/test_gex/test_gex_db.py`** — ≥10 test (save, load, upsert, serie vuota, etc.)

**GATE CHECK A**: 
```bash
python3 scripts/cron_gex.py
sqlite3 data/structured_notes.db "SELECT COUNT(*) FROM gex_snapshots"
# Risultato atteso: ≥ 1
```

---

### BLOCCO B — Bug Fix e Qualità Dati

1. **Fix #4 — Barriere NULL**: esaminare le 6 barriere con `level_price_btc = NULL`; aggiungere fallback in `NoteParser` per calcolare `level_price_ibit = initial_level * level_pct / 100` quando mancante
2. **Fix #5 — Note NULL**: identificare i 2 filing con parsing fallito, debug `NoteParser` su quei documenti specifici
3. **Fix #6 e #7 — Dati obsoleti**: configurare cron su DO App Platform o `scripts/` per aggiornamento giornaliero automatico:
   - `scripts/fetch_farside.py` ogni giorno alle 20:00 ET
   - `scripts/run_flows.py` ogni giorno alle 21:00 ET
4. **Fix #8 — Test environment**: creare `Makefile` con target `test` che attiva venv prima di `pytest`
5. **Fix #13 — pyproject.toml**: spostare `pytest`, `pytest-cov`, `responses` in `[project.optional-dependencies]`

**GATE CHECK B**: 
```bash
python3 -m pytest tests/ -v --tb=short
# Tutti 185+ test verdi, 0 errori di collection
```

---

### BLOCCO C — Hardening e Migliorie

1. **Backfill GEX storico**: script one-shot per popolare `gex_snapshots` con date storiche usando Deribit historical data API (se disponibile) o mark dati mancanti con NULL
2. **Alert su dati obsoleti**: aggiungere check in `app.py` che mostra warning se i dati prezzi sono più vecchi di 3 giorni
3. **Test di integrazione**: `tests/integration/test_api.py` che avvia l'API e testa tutti gli endpoint end-to-end
4. **Pin dipendenze**: generare `requirements-lock.txt` con `pip freeze` per reproducibilità
5. **Coverage target**: raggiungere ≥80% con `pytest --cov=src`
6. **Race condition cache**: aggiungere `gex` snapshot al response di `/api/signals` per evitare doppio fetch

**GATE CHECK C**: 
```bash
# Dashboard operativa
streamlit run src/dashboard/app.py &
# Tutti i tab si renderizzano con dati reali
# Backtest mostra equity curve non flat
# Regime percentile basato su ≥30 giorni di storico
```

---

## Appendice A: Output Comandi Chiave

```
# Tabelle DB
sqlite3 data/structured_notes.db ".tables"
→ barrier_levels  notes  prices

# Prezzi
sqlite3 data/structured_notes.db "SELECT ticker, COUNT(*), MIN(date), MAX(date) FROM prices GROUP BY ticker"
→ BTC-USD|373|2025-03-03|2026-03-12
→ IBIT|259|2025-03-03|2026-03-12

# Barriere
→ 10 active, 4 con level_price_btc, 6 NULL
→ knock_in: 8, autocall: 2

# Test
→ 185 passed, 3 collection errors (bs4, yfinance non installati)

# GEX save: nessun risultato
grep -rn "gex_snapshots|save_gex|INSERT.*gex" src/ scripts/
→ (vuoto)
```

---

## Appendice B: Schema SQL proposto per `gex_snapshots`

```sql
CREATE TABLE gex_snapshots (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    date                TEXT    NOT NULL UNIQUE,   -- YYYY-MM-DD (1 snapshot/giorno)
    timestamp           TEXT    NOT NULL,          -- ISO datetime del calcolo
    spot_price          REAL    NOT NULL,
    total_net_gex       REAL    NOT NULL,          -- USD raw (es. 450_000_000)
    gamma_flip_price    REAL,
    put_wall            REAL,
    call_wall           REAL,
    max_pain            REAL,
    regime              TEXT,                      -- positive_gamma|negative_gamma|neutral
    total_call_oi       REAL,
    total_put_oi        REAL,
    put_call_ratio      REAL,
    n_instruments       INTEGER,
    created_at          TEXT    NOT NULL
);

CREATE INDEX idx_gex_snapshots_date ON gex_snapshots(date);
```

---

## Appendice C: `scripts/cron_gex.py`

```python
#!/usr/bin/env python3
"""Cron script: salva uno snapshot GEX giornaliero nel DB.

Uso:
    python3 scripts/cron_gex.py

Scheduling (crontab):
    0 18 * * 1-5  /path/to/venv/bin/python3 /path/to/scripts/cron_gex.py

Logica:
    - Fetcha opzioni BTC da Deribit
    - Calcola GEX snapshot
    - Salva in gex_snapshots (UPSERT su date)
    - Aggiorna RegimeDetector con storico DB
"""
from __future__ import annotations

import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# Aggiungi root al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings, setup_logging
from src.gex.deribit_client import DeribitClient
from src.gex.gex_calculator import GexCalculator
from src.gex.regime_detector import RegimeDetector

_log = setup_logging("cron_gex")


def _get_db_path() -> Path:
    cfg = get_settings()
    return Path(cfg.get("db", {}).get("path", "data/structured_notes.db"))


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS gex_snapshots (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            date              TEXT    NOT NULL UNIQUE,
            timestamp         TEXT    NOT NULL,
            spot_price        REAL    NOT NULL,
            total_net_gex     REAL    NOT NULL,
            gamma_flip_price  REAL,
            put_wall          REAL,
            call_wall         REAL,
            max_pain          REAL,
            regime            TEXT,
            total_call_oi     REAL,
            total_put_oi      REAL,
            put_call_ratio    REAL,
            n_instruments     INTEGER,
            created_at        TEXT    NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_gex_date ON gex_snapshots(date)")
    conn.commit()


def save_snapshot(conn: sqlite3.Connection, snap, regime: str) -> None:
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    now_iso = datetime.now(tz=timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO gex_snapshots
            (date, timestamp, spot_price, total_net_gex, gamma_flip_price,
             put_wall, call_wall, max_pain, regime, total_call_oi,
             total_put_oi, put_call_ratio, n_instruments, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(date) DO UPDATE SET
            timestamp        = excluded.timestamp,
            spot_price       = excluded.spot_price,
            total_net_gex    = excluded.total_net_gex,
            gamma_flip_price = excluded.gamma_flip_price,
            put_wall         = excluded.put_wall,
            call_wall        = excluded.call_wall,
            max_pain         = excluded.max_pain,
            regime           = excluded.regime,
            total_call_oi    = excluded.total_call_oi,
            total_put_oi     = excluded.total_put_oi,
            put_call_ratio   = excluded.put_call_ratio,
            n_instruments    = excluded.n_instruments
    """, (
        today,
        snap.timestamp.isoformat(),
        snap.spot_price,
        snap.total_net_gex,
        snap.gamma_flip_price,
        snap.put_wall,
        snap.call_wall,
        snap.max_pain,
        regime,
        snap.total_call_oi,
        snap.total_put_oi,
        snap.put_call_ratio,
        len(snap.gex_by_strike),
        now_iso,
    ))
    conn.commit()
    _log.info("Snapshot GEX salvato per %s: spot=%.0f GEX=%.1fM regime=%s",
              today, snap.spot_price, snap.total_net_gex / 1e6, regime)


def main() -> None:
    db_path = _get_db_path()
    _log.info("DB: %s", db_path)

    try:
        client = DeribitClient()
        calc   = GexCalculator()
        spot   = client.get_spot_price()
        opts   = client.fetch_all_options("BTC")
        snap   = calc.calculate_gex(opts, spot)
    except Exception as exc:
        _log.error("Fetch Deribit fallito: %s", exc)
        sys.exit(1)

    detector = RegimeDetector()
    state    = detector.detect(snap)

    with sqlite3.connect(db_path) as conn:
        _ensure_table(conn)
        save_snapshot(conn, snap, state.regime)

    print(f"[OK] GEX snapshot salvato: spot={snap.spot_price:.0f} "
          f"GEX={snap.total_net_gex/1e6:+.1f}M regime={state.regime}")


if __name__ == "__main__":
    main()
```
