# Architettura del sistema

## Panoramica

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         FONTI DATI ESTERNE                               │
│  SEC EDGAR EFTS API   │   Deribit Public API   │   yfinance / Farside   │
└──────────┬────────────┴──────────┬─────────────┴──────────┬─────────────┘
           │                       │                          │
           ▼                       ▼                          ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────────────┐
│  Modulo 1        │   │  Modulo 2        │   │  Modulo 3                │
│  EDGAR Scraper   │   │  GEX Calculator  │   │  ETF Flow Tracker        │
│  src/edgar/      │   │  src/gex/        │   │  src/flows/              │
└────────┬─────────┘   └────────┬─────────┘   └────────────┬─────────────┘
         │                      │                           │
         │  SQLite              │  GexSnapshot              │  AggregateFlows
         │  structured_notes.db │  (in-memory)              │  + OHLCV cache
         └──────────────────────┴───────────────────────────┘
                                        │
                                        ▼
                           ┌─────────────────────────┐
                           │  Modulo 4               │
                           │  Statistical Analysis   │
                           │  src/analytics/         │
                           │                         │
                           │  · Granger causality    │
                           │  · Event study (CAR)    │
                           │  · Regime analysis      │
                           │  · Backtest             │
                           └────────────┬────────────┘
                                        │
                          ┌─────────────┴──────────────┐
                          ▼                            ▼
               ┌────────────────────┐  ┌────────────────────────┐
               │  Modulo A          │  │  Modulo B              │
               │  Forecast Engine   │  │  Alert System          │
               │  src/forecast/     │  │  src/alerts/           │
               └─────────┬──────────┘  └────────────┬───────────┘
                         │                          │
                         └──────────┬───────────────┘
                                    ▼
                           ┌─────────────────────────┐
                           │  Modulo 5               │
                           │  Streamlit Dashboard    │
                           │  src/dashboard/         │
                           └────────────┬────────────┘
                                       │
                                       ▼
                          ┌─────────────────────────┐
                          │  Modulo 6 (API + UI)    │
                          │  FastAPI + nginx        │
                          │  src/api/               │
                          └─────────────────────────┘
```

## Deployment (produzione)

Container **unico** su DO App Platform, gestito da `supervisord` (3 processi):

```
                    ┌──────────────────────────────────────┐
                    │         Container DO (porta 8080)    │
  https://…         │                                      │
  wagmi-lab.com ──► │  nginx :8080 (reverse proxy)         │
  (iframe embed)    │    ├── /api/*     → uvicorn :8000    │
                    │    └── /_stcore/* → streamlit :8501  │
                    │                       (WebSocket)     │
                    │  uvicorn :8000  — FastAPI (loopback)  │
                    │  streamlit :8501 — dashboard          │
                    │                                      │
                    │  /app/data/structured_notes.db        │
                    └──────────────────────────────────────┘
```

- **nginx** rimuove `X-Frame-Options` (impostato hardcoded da Streamlit) e aggiunge
  `Content-Security-Policy: frame-ancestors` → consente l'embed in iframe su wagmi-lab.com
- **supervisord** riavvia automaticamente i processi in caso di crash
- I dati sono condivisi via **DB versionato nel repo** (App Platform non supporta volumi):
  refresh EDGAR settimanale → commit su `main` → redeploy
- Config: `nginx.conf`, `supervisord.conf`, `.do/app.yaml`, `.streamlit/config.toml`

### URL

| Risorsa | URL |
|---------|-----|
| Dashboard (embed) | `https://btc-institutional-flow-tpw9m.ondigitalocean.app/?embed=true` |
| API docs | `https://btc-institutional-flow-tpw9m.ondigitalocean.app/api/docs` |
| Health | `https://btc-institutional-flow-tpw9m.ondigitalocean.app/api/health` |

## Flusso dati

### Pipeline principale

```
1. SEC EDGAR → HTML filing → Parser → StructuredNote + BarrierLevel → SQLite
2. Deribit API → OptionData[] → GexCalculator → GexSnapshot → GammaRegime
3. yfinance → OHLCV → SQLite cache → PriceFetcher
4. FarsideScraper → AggregateFlows[] → FlowCorrelation.merge() → merged_df
   (pipeline condivisa: src/api/data_pipeline.get_flow_context())
5. merged_df + GexSnapshot → Analytics modules → metrics/charts
6. Tutti i dati → Dashboard Streamlit / FastAPI / Desk Note
```

### Struttura dati centrale: `merged_df`

Il DataFrame `merged_df` è il cuore del sistema. Viene prodotto da `FlowCorrelation.merge()` e contiene:

| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| `btc_close` | float | Prezzo BTC di chiusura giornaliero |
| `btc_return` | float | Log return giornaliero BTC |
| `btc_vol_7d` | float | Volatilità rolling 7 giorni annualizzata |
| `ibit_close` | float | Prezzo IBIT di chiusura |
| `ibit_btc_ratio` | float | Rapporto IBIT/BTC (≈0.0006) |
| `ibit_flow` | float | Flusso netto IBIT giornaliero (USD) |
| `total_flow` | float | Flusso totale ETF Bitcoin (USD) |
| `ibit_flow_3d` | float | Flusso IBIT rolling 3 giorni |
| `total_flow_3d` | float | Flusso totale rolling 3 giorni |
| `btc_return_next1d` | float | Return BTC giorno successivo (per analisi predittiva) |

## Moduli

### `src/config.py`

- `get_settings()` — carica `config/settings.yaml` + override da `.env`, con `@lru_cache`
- `setup_logging(name)` — logger configurato su stderr + file `logs/tracker.log`

### `src/edgar/`

| File | Responsabilità |
|------|---------------|
| `search.py` | Query EFTS API con paginazione, deduplicazione per accession number |
| `parser.py` | Regex su HTML prospectus: barriere, notional, initial level, prodotto |
| `structured_notes_db.py` | CRUD SQLite: note, barriere, barrier/macro snapshots, refresh runs |
| `barrier_utils.py` | Clustering barriere, confluenza GEX↔barriere, `barrier_sign()` |
| `models.py` | `StructuredNote`, `BarrierLevel` dataclass |

**Dettaglio URL EDGAR:**
```
_id field = "{adsh}:{document_filename}"
CIK = _source.ciks[-1]  (emittente, non parent holding)
URL = https://www.sec.gov/Archives/edgar/data/{int(CIK)}/{acc_clean}/{doc_filename}
```

### `src/gex/`

**Formula GEX:**
```python
GEX = sign × gamma × OI × contract_size × spot² × 0.01
# sign = +1 per call (dealer short call → long gamma)
# sign = -1 per put (dealer short put → short gamma)
# contract_size = 1.0 BTC su Deribit
```

**Metriche calcolate:**
- `gamma_flip_price` — strike dove il GEX cumulativo cambia segno
- `put_wall` — strike con il GEX più negativo (supporto)
- `call_wall` — strike con il GEX più positivo (resistenza)
- `max_pain` — prezzo che massimizza le perdite per i compratori di opzioni

### `src/flows/`

**Strategia di fallback per i dati:**
```
1. Farside Investors (HTML scraping) → bloccato da Cloudflare 403
2. SoSoValue API → implementata (con retry)
3. yfinance volume estimate → ATTIVO
   flow_estimate = sign(return) × volume × close × 0.08
```

**Metriche macro:** `macro_fetcher.fetch_macro_data()` prova CoinGlass (5 fattori:
funding, OI, long/short, put/call, liquidazioni), poi CoinGecko
(`/api/v3/derivatives`) come ripiego (funding + OI). L'annualizzazione del funding
sta solo in `funding.py` (×3×365).

**Storico OI a 7 giorni:** tabella `macro_snapshots` nel DB versionato, alimentata da
`scripts/cron_macro.py` + workflow GitHub giornaliero.

**Fix yfinance multi-index:**
```python
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [c[0] for c in df.columns]
```

### `src/analytics/`

| Modulo | Test statistico | H0 |
|--------|----------------|-----|
| `pillars.py` | CompositeSignal a 4 pilastri (GEX/Barrier/Flows/Macro) | — |
| `factor_scorers.py` | Libreria scoring a 8 fattori (ex `signal_model`) | — |
| `signal_validation.py` | Spearman IC + null model + alpha decay | IC = 0 |
| `granger.py` | F-test (statsmodels) | flows non precedono returns |
| `event_study.py` | t-test a un campione | CAR = 0 intorno al barrier level |
| `regime_analysis.py` | Welch t-test | mean_return(pos_gamma) = mean_return(neg_gamma) |
| `walk_forward.py` | Rolling train→test OOS | — |
| `backtest.py` | Sharpe, Drawdown, Win Rate | — |

### `src/dashboard/`

- **`app.py`**: orchestratore — carica GEX/flussi/barriere una volta in `st.session_state`,
  poi `st.navigation(position="top")` con 7 pagine **lazy** (Panoramica di default)
- **`app_pages/`**: thin wrapper `st.Page` → `tabs/` (funzioni `_tab_*`)
- **`components.py`**: design system in `st.html` (classi `wx-`, stile Desk Note)
- **`charts.py`**: funzioni pure `DataFrame → go.Figure`, riusabili fuori dalla dashboard
- **`data_loader.py`**: funzioni `@st.cache_data(ttl=900)` condivise

### `src/forecast/`

Spine predizione → esito → calibrazione: `jobs.py` (predict/verify/calibrate, invocate
dal scheduler in-process in `src/api/scheduler.py`), `prediction_db.py` (predictions,
outcomes, weight_versions), `calibration.py` (proposta pesi **human-gated**),
`sources/dealer_flow.py`.

### `src/alerts/`

Alert Telegram via APScheduler: daily recap, ETF flow check, comandi `/recap`, `/signal`,
`/status`, `/help`. HTML escaping, retry con backoff, notifica errori.

### `src/report/` — Desk Note

Report a card pubblicabile (`facts.py`, `narrative.py`, `events.py`, `renderer.py`,
`formatting.py`). Pubblicazione **su evento**, non a calendario; il motore non inventa
(un estrattore senza dati restituisce `None`).

### `src/api/` — FastAPI

- **`main.py`**: orchestratore (~225 righe), middleware API key, lifespan scheduler
- **`routers/`** (7): `health`, `gex`, `flows`, `barriers`, `signals`, `forecast`, `report`
- **`cache.py`**: TTL cache in-memory + lock · **`scheduler.py`**: 3 APScheduler
  (alert, IFI + barrier snapshot, forecast)
- **`data_pipeline.py`**: `get_flow_context()` — pipeline flussi condivisa
- **`helpers.py`**: serializzazione `_ok()` / `_sanitize()` (numpy/pandas → JSON)

| Endpoint | Contenuto |
|----------|-----------|
| `GET /api/health`, `/api/health/edgar`, `/api/health/scheduler` | Health check |
| `GET /api/gex` | Snapshot GEX, regime, walls, profilo per strike |
| `GET /api/flows` | ETF flows, correlazione, Granger |
| `GET /api/barriers`, `/api/notes`, `/api/notes/by-url` | Barriere e note EDGAR |
| `GET /api/signals`, `/api/pillars/series` | Composite signal e serie pilastri |
| `GET /api/macro`, `/api/ifi` (deprecato) | Macro e IFI |
| `/api/predictions/*`, `/api/calibration`, `/api/forecast/status` | Forecast spine |
| `GET /report`, `/api/report/cards`, `/api/report/events` | Desk Note |

## Database SQLite

### `data/structured_notes.db`

DB SQLite unico (versionato in git). Contiene: `notes`, `barrier_levels`, `prices`
(OHLCV BTC/IBIT), `gex_snapshots`, `barrier_snapshots`, `macro_snapshots`, `refresh_runs`.

`data/runtime.db` (gitignorato) contiene i dati runtime: `predictions`, `outcomes`,
`weight_versions`, `alerts`. `StructuredNotesDB` e `GexDB` ignorano `DB_PATH` e puntano
sempre al DB versionato; `PredictionDB`/`AlertDB` rispettano `DB_PATH`.

```sql
CREATE TABLE notes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filing_url  TEXT UNIQUE NOT NULL,
    issuer      TEXT,
    product_type TEXT,
    notional_usd REAL,
    initial_level REAL,
    maturity_date TEXT,
    ...
);

CREATE TABLE barrier_levels (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id         INTEGER REFERENCES notes(id),
    barrier_type    TEXT,    -- knock_in | autocall | buffer | knock_out
    level_pct       REAL,    -- % del livello iniziale
    level_price_ibit REAL,   -- prezzo IBIT assoluto
    level_price_btc  REAL,   -- prezzo BTC corrispondente
    status          TEXT     -- active | triggered | expired
);

CREATE TABLE prices (
    ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL,
    return_pct REAL, updated_at TEXT,
    PRIMARY KEY (ticker, date)
);
```

## Configurazione (`config/settings.yaml`)

```yaml
edgar:
  base_url: "https://efts.sec.gov/LATEST/search-index"
  user_agent: "ibit-gamma-tracker/1.0 (…)"   # override via EDGAR_USER_AGENT
  rate_limit_rps: 8
  page_size: 100
  search_terms: ["IBIT", "iShares Bitcoin Trust", "FBTC", "BITB", "ARKB"]
  forms: ["424B2", "424B3"]
  start_date: "2024-01-01"

deribit:
  base_url: "https://www.deribit.com/api/v2/public"
  rate_limit_rps: 15
  gex_threshold_usd: 1_000_000

flows:
  farside_url: "https://farside.co.uk/bitcoin-etf-flow-all-data/"
  lookback_days: 365

coinglass:
  api_key: ""                # override via COINGLASS_API_KEY
  timeout_s: 15

backtest:
  barrier_exclusion_pct: 5.0
  transaction_cost_bps: 80
  trading_days_per_year: 365

analytics:
  granger_max_lags: 10
  event_window_days: 5
  barrier_proximity_pct: 2.0

alerts:
  telegram_enabled: true     # richiede TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
  daily_recap: {hour_utc: 12, minute_utc: 45}

forecast:
  enabled: true
  horizon_days: 5

dashboard:
  refresh_interval_s: 900
  theme: {background: "#000000", positive: "#00FF9D", negative: "#FF0033", …}
```
