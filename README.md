# ibit-gamma-tracker

> Toolkit per l'analisi dell'impatto del dealer hedging su note strutturate IBIT sul prezzo BTC.

Basato sulla teoria di Arthur Hayes: i dump/pump di BTC sono causati dal delta hedging meccanico dei dealer che emettono note strutturate (autocallable, barrier note, buffered note) legate all'ETF iShares Bitcoin Trust (IBIT) di BlackRock.

---

## Teoria

I dealer che emettono note strutturate IBIT devono coprire la loro esposizione delta/gamma in modo continuativo:

- In **positive gamma** (GEX > 0): i dealer comprano quando BTC scende, vendono quando sale → **effetto stabilizzante**, bassa volatilità
- In **negative gamma** (GEX < 0): i dealer vendono quando BTC scende, comprano quando sale → **effetto amplificante**, alta volatilità
- Le **barriere knock-in** (es. 70% del livello iniziale): se il prezzo scende sotto la barriera, il dealer deve vendere massicciamente per coprirsi → **accelerazione del sell-off**
- I **forti inflow IBIT** precedono i rialzi di BTC di ~5-7 giorni (confermato con Granger causality, p≈0.03)

---

## Architettura

```
btc-institutional-flow/
│
├── src/
│   ├── edgar/          # Modulo 1 — SEC EDGAR scraper + parser note strutturate
│   │   ├── search.py          # EFTS API full-text search su 424B2/424B3
│   │   ├── parser.py          # Estrazione barriere/notional da HTML JPMorgan
│   │   ├── structured_notes_db.py  # SQLite storage note e barriere
│   │   └── models.py          # StructuredNote, BarrierLevel dataclass
│   │
│   ├── flows/          # Modulo 3 — ETF Flow Tracker
│   │   ├── scraper.py         # Farside + fallback yfinance volume estimate
│   │   ├── price_fetcher.py   # BTC/IBIT OHLCV da yfinance (SQLite cache)
│   │   ├── correlation.py     # Merge flussi+prezzi, correlazione rolling
│   │   └── models.py          # EtfFlowData, AggregateFlows, MergedRecord
│   │
│   ├── gex/            # Modulo 2 — Gamma Exposure Calculator
│   │   ├── deribit_client.py  # Deribit public API (greeks, OI, spot)
│   │   ├── gex_calculator.py  # GEX formula, gamma flip, put/call wall, max pain
│   │   ├── regime_detector.py # Regime classify + alert generation
│   │   └── models.py          # GexSnapshot, GexByStrike, RegimeState
│   │
│   ├── analytics/      # Modulo 4 — Statistical Analysis
│   │   ├── pillars.py         # CompositeSignal a 4 pilastri (single source of truth)
│   │   ├── factor_scorers.py   # Libreria scoring 8 fattori (ex signal_model)
│   │   ├── signal_validation.py # Information Coefficient, alpha decay, null model IC
│   │   ├── backtest.py        # Backtest + null models (random, always_long, momentum)
│   │   ├── walk_forward.py    # Walk-forward validation rolling train→test
│   │   ├── factor_decomposition.py # OLS alpha/beta decomposition
│   │   ├── sensitivity.py     # Parameter sensitivity ±20%
│   │   ├── granger.py         # Granger causality + find_optimal_lag()
│   │   ├── event_study.py     # CAR intorno ai barrier levels
│   │   ├── regime_analysis.py # Welch t-test positive vs negative gamma
│   │   └── confluence_backtest.py # Probe confluenza barriere↔GEX
│   │
│   ├── dashboard/      # Modulo 5 — Streamlit Dashboard
│   │   ├── app.py             # Main app multi-tab (6 tab)
│   │   ├── data_loader.py     # Funzioni @st.cache_data condivise
│   │   ├── tabs/              # 6 moduli: barrier_map, gex, flows, signals, edgar, validation
│   │   ├── charts.py          # Plotly chart builders
│   │   ├── header.py / sidebar.py / static/style.css
│   │
│   └── config.py       # Settings loader (YAML + .env), logging setup
│
├── scripts/            # Entry point CLI
│   ├── run_edgar.py           # Scarica e parsa filing EDGAR
│   ├── run_flows.py           # Scarica flussi ETF e prezzi
│   ├── run_gex.py             # Calcola GEX live da Deribit
│   └── run_analytics.py       # Esegue tutti gli analytics
│
├── tests/              # ~679 test unitari (pytest)
├── config/settings.yaml       # Tutti i parametri configurabili
└── data/               # SQLite DB locali (auto-creati)
```

---

## Dashboard

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ₿ ibit-gamma-tracker                                                   │
│  BTC: $84,200  │  GEX: +41.5M$  │  Put Wall: $75k  │  Call Wall: $90k  │
│  ┌──────────────────────────────┐                                        │
│  │  Regime: POSITIVE GAMMA      │  ⚠️ NEAR CALL_WALL                    │
│  └──────────────────────────────┘                                        │
├──────────┬──────────┬──────────┬──────────┬──────────────────────────────┤
│  📊 GEX  │ 💸 Flows │ 🔬 Analy │ 📈 Back  │  🏛️ EDGAR Barriers          │
├──────────┴──────────┴──────────┴──────────┴──────────────────────────────┤
│  [GEX profile bar chart per strike]    [Livelli chiave: walls + flip]    │
│                                                                           │
│  Gamma Flip: $75k  │  Max Pain: $82k  │  Put/Call OI: 0.94              │
└──────────────────────────────────────────────────────────────────────────┘
```

**6 tab:**
- **Barrier Map** — Mappa visiva dei livelli critici EDGAR con confluenza GEX
- **GEX** — Profilo Gamma Exposure, regime, gamma flip, put/call wall
- **ETF Flows** — Flussi IBIT e multi-ETF, correlazione rolling, Granger causality
- **Segnali** — CompositeSignal a 4 pilastri (GEX/Barrier/Flows/Macro) con gauge e backtest vs null models
- **EDGAR Monitor** — Note strutturate SEC, barriere attive, event study CAR
- **Validation** — Information Coefficient (potere predittivo), Walk-Forward, Factor Decomposition, Parameter Sensitivity

---

## Installazione

### Prerequisiti

- Python 3.9+
- Git

### Setup

```bash
git clone git@github.com:strebba/btc-institutional-flow.git
cd btc-institutional-flow

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

### Configurazione

```bash
cp .env.example .env
# Edita .env se vuoi sovrascrivere i parametri in config/settings.yaml
```

Le variabili rilevanti in `.env`:

```env
# Opzionale — tutti hanno default in settings.yaml
EDGAR_USER_AGENT="ibit-gamma-tracker/1.0 (tua@email.com)"
DERIBIT_BASE_URL="https://www.deribit.com/api/v2"
```

---

## Utilizzo

### 1. Popola il database EDGAR

Scarica e parsa i filing 424B2/424B3 di note strutturate IBIT:

```bash
python scripts/run_edgar.py
# oppure con limite:
python scripts/run_edgar.py --max 20
```

Output: `data/structured_notes.db` con note strutturate e barriere attive.

### 2. Scarica prezzi e flussi ETF

```bash
python scripts/run_flows.py
```

Scarica 365 giorni di OHLCV BTC/IBIT (yfinance), stima i flussi IBIT se Farside non è raggiungibile.

### 3. Calcola GEX live

```bash
python scripts/run_gex.py
```

Chiama l'API pubblica di Deribit, calcola GEX per ~948 opzioni BTC, stampa regime e alert.

### 4. Esegui gli analytics

```bash
python scripts/run_analytics.py              # tutto
python scripts/run_analytics.py --granger    # solo Granger
python scripts/run_analytics.py --regime     # solo regime analysis
python scripts/run_analytics.py --events     # solo event study
python scripts/run_analytics.py --backtest   # solo backtest
```

### 5. Avvia la dashboard

```bash
streamlit run src/dashboard/app.py
```

Apri http://localhost:8501

---

## Risultati con dati reali

| Test | Risultato |
|------|-----------|
| **Information Coefficient** | IC del CompositeSignal vs forward BTC return — validazione rolling con null model |
| **GEX live** | +$41.5M → regime **POSITIVE_GAMMA**, Put Wall $60k (-12%), Call Wall $75k (+9%) |
| **EDGAR filing** | 547 filing 424B2/424B3 trovati, JPMorgan dominante emittente |
| **Note strutturate** | 8 note parsed (autocallable, barrier note), 10 barriere attive |
| **Walk-Forward** | Rolling train (2 anni) → test (3 mesi) per validazione OOS |
| **Factor Decomposition** | OLS regression per separare alpha puro da beta mascherato |

---

## Test

```bash
pytest                    # tutti i test (~679)
pytest tests/test_edgar/  # solo EDGAR
pytest tests/test_gex/    # solo GEX
pytest tests/test_flows/  # solo Flows
pytest tests/test_analytics/  # solo Analytics (302 test)
pytest tests/test_forecast/   # solo Forecast
```

---

## Fonti dati

| Fonte | Tipo | Note |
|-------|------|------|
| [SEC EDGAR EFTS](https://efts.sec.gov/LATEST/search-index) | Filing 424B2/424B3 | Gratuito, nessuna API key |
| [Deribit](https://www.deribit.com/api/v2/public/) | Opzioni BTC live | Gratuito, no auth |
| [yfinance](https://pypi.org/project/yfinance/) | Prezzi BTC/IBIT | Gratuito |
| [Farside Investors](https://farside.co.uk/bitcoin-etf-flow-all-data/) | Flussi ETF | Cloudflare-bloccato → fallback yfinance |

---

## Licenza

MIT
