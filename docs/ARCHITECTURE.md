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
                                       ▼
                          ┌─────────────────────────┐
                          │  Modulo 5               │
                          │  Streamlit Dashboard    │
                          │  src/dashboard/         │
                          └─────────────────────────┘
```

## Flusso dati

### Pipeline principale

```
1. SEC EDGAR → HTML filing → Parser → StructuredNote + BarrierLevel → SQLite
2. Deribit API → OptionData[] → GexCalculator → GexSnapshot → RegimeState
3. yfinance → OHLCV → SQLite cache → PriceFetcher
4. FarsideScraper → AggregateFlows[] → FlowCorrelation.merge() → merged_df
5. merged_df + GexSnapshot → Analytics modules → metrics/charts
6. Tutti i dati → Dashboard Streamlit → visualizzazione real-time
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
| `structured_notes_db.py` | CRUD SQLite: note, barriere, status update |
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
2. SoSoValue API → non implementata
3. yfinance volume estimate → ATTIVO
   flow_estimate = sign(return) × volume × close × 0.08
4. CSV manuale → FarsideScraper.from_csv(path)
```

**Fix yfinance multi-index:**
```python
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [c[0] for c in df.columns]
```

### `src/analytics/`

| Modulo | Test statistico | H0 |
|--------|----------------|-----|
| `granger.py` | F-test (statsmodels) | flows non precedono returns |
| `event_study.py` | t-test a un campione | CAR = 0 intorno al barrier level |
| `regime_analysis.py` | Welch t-test | mean_return(pos_gamma) = mean_return(neg_gamma) |
| `backtest.py` | Sharpe, Drawdown, Win Rate | — |

### `src/dashboard/`

- **`app.py`**: orchestrazione Streamlit, `@st.cache_data(ttl=900)` per tutti i moduli
- **`charts.py`**: funzioni pure `DataFrame → go.Figure`, riusabili fuori dalla dashboard

## Database SQLite

### `data/structured_notes.db`

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
```

### `data/prices.db`

```sql
CREATE TABLE btc_ohlcv (
    date TEXT PRIMARY KEY,
    open REAL, high REAL, low REAL, close REAL, volume REAL
);
CREATE TABLE ibit_ohlcv (
    date TEXT PRIMARY KEY,
    open REAL, high REAL, low REAL, close REAL, volume REAL
);
```

## Configurazione (`config/settings.yaml`)

```yaml
edgar:
  base_url: "https://efts.sec.gov/LATEST/search-index"
  rate_limit_s: 0.5          # secondi tra richieste
  max_results: 500
  search_queries: ["IBIT", "iShares Bitcoin Trust"]
  form_types: ["424B2", "424B3"]

deribit:
  base_url: "https://www.deribit.com/api/v2"
  rate_limit_s: 0.07         # ~15 req/s
  gex_threshold_usd: 1000000 # threshold per classificazione regime

backtest:
  long_gex_threshold: 0
  long_flow_threshold_usd_m: 100    # M$
  short_gex_threshold: 0
  short_flow_threshold_usd_m: -200  # M$
  barrier_exclusion_pct: 5.0

analytics:
  granger_max_lags: 10
  event_window_days: 5
  barrier_proximity_pct: 2.0

dashboard:
  refresh_interval_s: 900    # 15 minuti
  theme:
    background: "#1a1a2e"
    positive:   "#00ff88"
    negative:   "#ff4444"
    neutral:    "#4488ff"
```
