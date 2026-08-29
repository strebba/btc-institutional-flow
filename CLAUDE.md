# CLAUDE.md — btc-institutional-flow (ibit-gamma-tracker)

Toolkit Python per l'impatto del **dealer hedging** su note strutturate IBIT sul prezzo BTC
(tesi Arthur Hayes). Espone un **backend FastAPI** + una dashboard Streamlit.

> Esiste una `memory/MEMORY.md` nel repo con dettagli di architettura e bug-fix storici della
> dashboard Streamlit — leggerla per il dettaglio, **non duplicarla** qui.
>
> Esiste `CONTEXT.md` (root) con il glossario dei termini canonici del dominio.

## Comandi essenziali (Makefile)

```bash
make install        # pip install -e ".[dev]"
make run-api        # FastAPI → http://localhost:8000  (= python run_api.py)
make run-dashboard  # streamlit run src/dashboard/app.py
make compose-up     # replica ambiente DO: nginx:8080 + API + dashboard (docker compose)
make test           # pytest tests/ -v  (~834 test)
make test-unit      # pytest tests/unit/ -v -q (esclude integration)
make lint           # ruff check src/ tests/
make update-all     # update-gex + update-flows + update-edgar (cron data refresh)
```

Lint/type: `ruff` (configurato in `pyproject.toml`), `.pre-commit-config.yaml`.
Venv locale in `.venv`.

## Ruolo nell'ecosistema

Il FastAPI di questo repo (`run_api.py`, porta 8000) è il **BTC API consumato da PTF-Dashboard**
(lì configurato come `VITE_BTC_API_URL`). Modifiche end-to-end ai dati BTC della dashboard toccano
entrambi i repo. Su DO il backend e la dashboard Streamlit girano nello **stesso container**
(nginx + supervisord), con la dashboard embeddata nel sito Wix **wagmi-lab.com**.

## Architettura (moduli `src/`)

| Modulo | Ruolo |
|--------|-------|
| `src/edgar/` | SEC EDGAR scraper/parser note strutturate (424B2/424B3) → SQLite |
| `src/gex/` | Gamma Exposure da Deribit (`gex_calculator.py`, `deribit_client.py`): GEX, gamma flip, put/call wall, max pain |
| `src/flows/` | ETF flow tracker (Farside + yfinance, Coinglass, SoSoValue), price fetcher BTC/IBIT, correlazioni, EDGAR N-PORT, `macro_fetcher.py` (dati macro unificati) |
| `src/analytics/` | Segnale composito a 4 pilastri (`pillars.py` single source of truth) + `factor_scorers.py` (ex signal_model) + backtest (+ transaction costs 80bps, null models) + IFI + Granger (+ `find_optimal_lag` anti data-snooping) + regime analysis + `signal_validation.py` (Information Coefficient, alpha decay) |
| `src/dashboard/` | Dashboard Streamlit — `app.py` orchestratore, `data_loader.py` (cached), `tabs/` (6 moduli con validation tab), `charts.py` (Plotly), `header.py`, `sidebar.py`, `static/style.css` |
| `src/api/` | FastAPI — `main.py` orchestratore (~225 righe), `routers/` (7 file: health, gex, flows, barriers, signals, forecast, report), `cache.py`, `helpers.py`, `auth.py`, `scheduler.py`, `schemas.py` |
| `src/alerts/` | Alert Telegram (ETF flow check, daily recap, error notification, comandi /recap /status /help) via `apscheduler` + GEX alert monitor |
| `src/forecast/` | Predizioni dealer-flow, calibrazione pesi, validazione esiti, multi-source (EMA, portfolio, dealer-flow) |
| `src/report/` | **Desk Note** — report a card pubblicabili. `facts.py` (estrattori + salienza), `narrative.py` (selezione e composizione), `events.py` (trigger di pubblicazione + `ReportStateDB`), `renderer.py` (HTML per web e PNG), `formatting.py` (numeri all'italiana), `fonts/` (IBM Plex incorporato) |

DB: SQLite in `data/` (`structured_notes.db` versionato + `runtime.db` gitignorato).
`StructuredNotesDB` e `GexDB` puntano **sempre** a `structured_notes.db` (path hardcodato,
ignorano `DB_PATH`). `SignalDB`, `PredictionDB`, `AlertDB` rispettano `DB_PATH` (default
`structured_notes.db`, override `data/runtime.db` in dev). Config: `config/settings.yaml` +
`config/weights.yaml` via `src.config.get_settings()`. Scheduler/cron in `scripts/` (16 script).

## Skills disponibili

Tutte le skill sono disponibili globalmente e richiamabili via `skill` tool.
**Regola generale**: quando lavori su un file/modulo elencato sotto, carica la skill corrispondente
prima di iniziare — fornisce pattern, best practice, e reference aggiornati.

### Project-installed (`.agents/skills/`)

| Skill | Trigger | File/Task |
|-------|---------|-----------|
| `fastapi-python` | Qualsiasi modifica a `src/api/` | `main.py`, `routers/*`, `schemas.py`, `auth.py`, `cache.py`, `scheduler.py` |
| `developing-with-streamlit` | Qualsiasi modifica a `src/dashboard/` | `app.py`, `tabs/*`, `charts.py`, `data_loader.py`, `header.py`, `sidebar.py` |
| `tdd` | Scrivere/aggiornare test (`tests/`) | Red-green-refactor, test first |
| `systematic-debugging` | Qualsiasi bug o test failure | Root cause tracing, defense-in-depth |
| `codebase-design` | Refactoring, nuovo modulo/seam | Deep module design, interfacce |
| `domain-modeling` | Modellare nuovi concetti dominio | ADR, ubiquitous language, CONTEXT.md |
| `find-skills` | Cercare nuove skill utili | Ricerca ecosistema skills.sh |

### Globale — Core dominio

| Skill | Trigger | File/Task |
|-------|---------|-----------|
| `crypto-derivatives` | GEX, gamma flip, dealer positioning, options flow, funding rate, barriere, max pain | `src/gex/*`, `src/edgar/barrier_utils.py`, `src/analytics/pillars.py` (pilastro gex/barrier), `tabs/gex.py`, `tabs/barrier_map.py` |
| `quantitative-research` | Backtesting, alpha generation, factor models, regime detection, walk-forward, statistical arbitrage | `src/analytics/backtest.py`, `src/analytics/factor_scorers.py`, `src/analytics/regime_analysis.py`, `src/analytics/walk_forward.py`, `src/analytics/pillars.py` |
| `Time Series Analysis` | Trend, autocorrelation, Granger causality, forecasting, ARIMA, ACF/PACF | `src/analytics/granger.py`, `src/forecast/*`, `src/flows/correlation.py`, `src/analytics/signal_validation.py` |
| `portfolio-risk` | VaR, max drawdown, Sharpe/Sortino, correlation matrix, rolling metrics | `src/analytics/backtest.py`, `src/analytics/regime_analysis.py`, `src/analytics/sensitivity.py` |
| `scipy-best-practices` | Ottimizzazione, stat avanzata, interpolazione, signal processing | `src/analytics/*`, `src/forecast/calibration.py`, qualsiasi uso di `scipy.*` |
| `plotly` | Qualsiasi grafico Plotly | `src/dashboard/charts.py`, `tabs/gex.py`, `tabs/barrier_map.py`, `tabs/flows.py`, `tabs/signals.py` |

### Globale — Infrastruttura & operatività

| Skill | Trigger | File/Task |
|-------|---------|-----------|
| `bingx-swap-market` | Funding rate, OI, order book da BingX (fonte alternativa ai dati macro) | `src/flows/macro_fetcher.py`, `src/analytics/factor_scorers.py` |
| `error-monitoring` | Error logging, Sentry, health check, structured logging backend | `src/api/main.py`, `src/alerts/`, qualsiasi gestione errori produzione |

### Skill NON usare (stack non corrispondente)

`playwright-e2e` (React), `postgres-optimization` (SQLite), `react-performance`, `supabase-realtime`, `supabase-security`, `bingx-fund-account`, `bingx-swap-account`, `bingx-swap-trade`, `customize-opencode`

## Deploy

**DO App Platform** (`btc-institutional-flow-tpw9m.ondigitalocean.app`, `.do/app.yaml`):
container unico con **supervisord** che gestisce 3 processi:
- `nginx` :8080 → reverse proxy pubblico (root del dominio → dashboard Streamlit)
- `uvicorn` :8000 → FastAPI backend (solo loopback, `/api/*`)
- `streamlit` :8501 → dashboard (solo loopback, tema Wagmi Lab da `.streamlit/config.toml`)

nginx: `/api/*` e `/report` → FastAPI, `/*` → Streamlit (con WebSocket `/_stcore/*`); rimuove
`X-Frame-Options` e imposta `Content-Security-Policy: frame-ancestors` per **wagmi-lab.com**
(embed iframe). App Platform non supporta volumi → i dati sono condivisi via **DB versionato
nel repo** (refresh EDGAR → commit → redeploy). Config nginx: `nginx.conf`; processi:
`supervisord.conf`. Modifiche alla spec DO: applicare da Console → Settings → App Spec.
Local mirror: `docker compose up -d --build` → http://localhost:8080 (dash) e /api/docs (API).
La dashboard gira anche in locale standalone (`make run-dashboard`, porta 8501).

## Refresh dati EDGAR (note IBIT)

Il DB `data/structured_notes.db` è **versionato** (fonte di verità: filesystem DO effimero).
Refresh incrementale: `scripts/cron_edgar.py` (env `EDGAR_LOOKBACK_DAYS`, default 30); full:
`make update-edgar`. Automazione: `.github/workflows/edgar-refresh.yml` (lunedì + backup
mercoledì 06:30 UTC, committa il DB su `main` → deploy DO). Lo User-Agent SEC è in
`config/settings.yaml` (email reale) — non servono variabili esterne. Override opzionale
via env var `EDGAR_USER_AGENT`. In caso di fallimento, il workflow invia una notifica
Telegram (richiede `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` nei Repository secrets).
Endpoint di monitoraggio: `GET /api/health/edgar`. La salute si misura sull'**ultimo
refresh riuscito** (tabella `refresh_runs`, soglia 10 giorni), non sull'ultima nota
scritta: il workflow gira due volte a settimana e in una finestra tranquilla può
legittimamente non trovare filing nuovi — misurare l'età delle note faceva sembrare
rotta una pipeline sana. `notes_age_days` resta esposto come informazione sul mercato
primario, e `reason` dice a parole cosa sta succedendo.
I supplement *preliminari* hanno `is_preliminary=1` e `initial_level`/`notional` = NULL;
`/api/barriers` mostra solo i finali.

I search terms includono anche FBTC/BITB/ARKB: il parser estrae il ticker reale del sottostante
(`_detect_underlying`, colonna `notes.underlying`), ma `get_active_barriers()`,
`compute_btc_prices()` e `update_barrier_statuses()` operano **solo sulle note IBIT** (default) —
i prezzi/ratio IBIT non si applicano agli altri ETF. `data/runtime.db` (predizioni/cache runtime,
usato da `make run-api` via `DB_PATH`) è invece **ignorato** da git, separato dal seed versionato.

## Desk Note (report a card)

Report pubblicabile a sei card generato dagli endpoint esistenti — nato per
rispondere ai post a carosello dei competitor, che le persone leggono mentre la
dashboard richiede di saperla leggere.

```bash
python3 scripts/export_desk_note.py                 # -> out/desk-note/*.png (1080x1350)
python3 scripts/export_desk_note.py --only-on-event # esporta solo se e' notizia
```

- `GET /report` — la pagina (via nginx, fuori da `/api/`). `?export=true` toglie
  la riduzione responsive e l'intestazione: e' cio' che fotografa Playwright.
- `GET /api/report/cards` — le card in JSON, sorgente di tutti i renderer.
- `GET /api/report/events` — cosa e' cambiato (sola lettura, non consuma gli eventi).
- `POST /api/report/events/commit` — fissa la linea di base dopo la pubblicazione.

**Pubblicazione su evento, non a calendario**: l'edizione esce quando il regime
gamma si ribalta, lo spot attraversa il gamma flip o una barriera SEC, un muro
viene superato, o il segnale cambia lato delle soglie 40/65. Soglia in
`events.PUBLISH_THRESHOLD`.

**Il motore non inventa mai.** Un estrattore senza dati restituisce `None` e
costa una card, non l'edizione; sotto le soglie di materialita'
(`facts._MIN_FLOW_USD_M`, `_MIN_GEX_USD_M`) un numero non diventa un fatto —
quando Farside e' giu' i flussi arrivano a `0.0`, che e' "non lo sappiamo", non
"zero". I pilastri scoperti finiscono in `DeskNote.warnings`, che e' cio' che
tiene ferma la card del punteggio finche' CoinGlass non torna.
