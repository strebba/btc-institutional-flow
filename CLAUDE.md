# CLAUDE.md — btc-institutional-flow (ibit-gamma-tracker)

Toolkit Python per l'impatto del **dealer hedging** su note strutturate IBIT sul prezzo BTC
(tesi Arthur Hayes). Espone un **backend FastAPI** + una dashboard Streamlit.
Traccia anche **ETH** (dal 2026-10: GEX, flussi ETF, macro — niente barriere né EDGAR).

**Solo dati osservati, nessun punteggio** (2026-10-07): rimossi segnale composito a 4
pilastri, IFI, backtest, pagine Segnali/Validation, Desk Note e comando Telegram `/signal`.
Non reintrodurre score/verdetti LONG/CAUTION senza una decisione esplicita di Stefano.

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
make test           # pytest tests/ -v  (~990 test)
make test-unit      # pytest tests/ --ignore=tests/integration/ -v -q
make lint           # .venv/bin/ruff check src/ tests/
make typecheck      # .venv/bin/mypy src/ --ignore-missing-imports
make update-all     # update-gex + update-flows + update-edgar + update-macro (cron refresh)
```

Lint: `ruff` (configurato in `pyproject.toml`, pin esatto in CI), `.pre-commit-config.yaml`.
Type-check: `mypy` è disponibile via `make typecheck` ma non è (ancora) un gate CI —
baseline di ~70 errori pre-esistenti al 2026-09, mai risolti perché il tool non era
nemmeno installato prima d'ora. Da burn-down prima di attivarlo in `ci.yml`.
Venv locale in `.venv`.

## Ruolo nell'ecosistema

Il FastAPI di questo repo (`run_api.py`, porta 8000) è il **BTC API consumato da PTF-Dashboard**
(lì configurato come `VITE_BTC_API_URL`). Modifiche end-to-end ai dati BTC della dashboard toccano
entrambi i repo. Su DO il backend e la dashboard Streamlit girano nello **stesso container**
(nginx + supervisord), con la dashboard embeddata nel sito Wix **wagmi-lab.com**.

## Architettura (moduli `src/`)

| Modulo | Ruolo |
|--------|-------|
| `src/assets.py` | Registro `AssetSpec` (BTC, ETH): unico punto per simboli Deribit/CoinGlass/CoinGecko, URL Farside, lead ETF (IBIT/ETHA), nomi di colonna del `merged_df` e `features` disponibili. Ovunque `asset="BTC"` di default; BTC conserva i nomi legacy (`btc_close`, `ibit_flow`) |
| `src/edgar/` | SEC EDGAR scraper/parser note strutturate (424B2/424B3) → SQLite |
| `src/gex/` | Gamma Exposure da Deribit (`gex_calculator.py`, `deribit_client.py`): GEX, gamma flip, put/call wall, max pain |
| `src/flows/` | ETF flow tracker (Farside + yfinance, Coinglass, SoSoValue), price fetcher BTC/IBIT, correlazioni, EDGAR N-PORT, `macro_fetcher.py` (dati macro unificati), `coingecko_client.py` (ripiego funding/OI) |
| `src/analytics/` | Granger (+ `find_optimal_lag` anti data-snooping), regime analysis (pagina GEX), event study (pagina EDGAR); `factor_scorers.py` resta solo come dipendenza del forecast spine |
| `src/dashboard/` | Dashboard Streamlit — `app.py` orchestratore + `st.navigation` lazy (solo la pagina attiva calcola), `app_pages/` (5 pagine, thin wrapper sulle funzioni `_tab_*`), `tabs/` (contenuto delle 5 sezioni), `data_loader.py` (cached), `charts.py` (Plotly), `components.py` (design system), `header.py`, `sidebar.py`, `static/` (font IBM Plex self-hosted) |
| `src/api/` | FastAPI — `main.py` orchestratore, `routers/` (6 file: health, gex, flows, barriers, macro, forecast), `cache.py`, `helpers.py`, `data_pipeline.py` (`get_flow_context`, pipeline flussi condivisa), `scheduler.py`. Nessun `schemas.py`/Pydantic sulle risposte: gli endpoint restituiscono dict via il wrapper `_ok()` |
| `src/alerts/` | Alert Telegram — **solo livelli GEX e flussi ETF**: daily recap (regime, spot, net GEX, flip, muri + flussi), alert eventi flussi ETF, error notification, comandi /recap /status /help, via `apscheduler` |
| `src/forecast/` | Predizioni dealer-flow, calibrazione pesi, validazione esiti |

DB: SQLite in `data/` (`structured_notes.db` versionato + `runtime.db` gitignorato).
`gex_snapshots` e `macro_snapshots` hanno la colonna `asset` (unicità data+asset); la
migrazione dallo schema pre-ETH gira da sola all'avvio, è atomica e idempotente.
`StructuredNotesDB` e `GexDB` puntano **sempre** a `structured_notes.db` (path hardcodato,
ignorano `DB_PATH`). `PredictionDB`, `AlertDB` rispettano `DB_PATH` (default
`structured_notes.db`, override `data/runtime.db` in dev). Config: `config/settings.yaml` +
`config/weights.yaml` via `src.config.get_settings()`. Script CLI in `scripts/` (12).

## Dashboard: tema e navigazione (2026-09)

Tema **nativo Streamlit** in `.streamlit/config.toml` (nero `#000` + neon `#00FF9D`,
palette Wagmi Lab), niente CSS inline. Font **IBM Plex Sans/Mono self-hosted** da
`src/dashboard/static/` (serviti via `server.enableStaticServing=true` +
`[[theme.fontFaces]]` → `/app/static/*`), nessuna
dipendenza da fonts.gstatic.com. I colori dei grafici Plotly restano in
`config/settings.yaml → dashboard.theme` (allineati al tema).

Asset: selettore BTC/ETH in cima alla sidebar (`st.segmented_control`, `bind="query-params"`
→ `?asset=eth`), letto da `app.py` prima di caricare i dati; lo spec va in
`st.session_state["asset_spec"]`. Pagine visibili per asset in `navigation.py`
(`visible_pages`): con ETH solo Panoramica, GEX, ETF Flows.

Navigazione: `st.navigation(position="top")` + `st.Page` in `src/dashboard/app_pages/`
(**5 pagine**, thin wrapper sulle funzioni `_tab_*` di `tabs/`). **Panoramica è la
default** (answer-first: livelli GEX + flussi + derivati + barriera più vicina), poi
GEX, ETF Flows, Barrier Map, EDGAR.
**Solo la pagina attiva viene eseguita** — prima `st.tabs` era eager ed eseguiva
Granger/event-study a ogni load. `app.py` carica GEX/flussi/barriere una volta e li mette in
`st.session_state`. Il refresh manuale invalida anche `run_signal_ic` (presente nella
lista `fn.clear()` dei loader rimasti). `_PAGES_DIR` (in `navigation.py`) usa `Path(__file__).resolve().parent`.

Design system: `src/dashboard/components.py` (`tape`, `eyebrow`, `hero`, `pillar_bars`)
in `st.html` con CSS proprio (classi `wx-`: tape e eyebrow mono, label
neon). **Non tocca i widget nativi Streamlit** — niente override di classi interne.
`inject_style()` è chiamato una volta in `app.py`.

## Skills disponibili

Tutte le skill sono disponibili globalmente e richiamabili via `skill` tool.
**Regola generale**: quando lavori su un file/modulo elencato sotto, carica la skill corrispondente
prima di iniziare — fornisce pattern, best practice, e reference aggiornati.

### Project-installed (`.agents/skills/`)

| Skill | Trigger | File/Task |
|-------|---------|-----------|
| `fastapi-python` | Qualsiasi modifica a `src/api/` | `main.py`, `routers/*`, `auth.py`, `cache.py`, `scheduler.py` |
| `developing-with-streamlit` | Qualsiasi modifica a `src/dashboard/` | `app.py`, `app_pages/*`, `tabs/*`, `charts.py`, `data_loader.py`, `header.py`, `sidebar.py` |
| `tdd` | Scrivere/aggiornare test (`tests/`) | Red-green-refactor, test first |
| `systematic-debugging` | Qualsiasi bug o test failure | Root cause tracing, defense-in-depth |
| `codebase-design` | Refactoring, nuovo modulo/seam | Deep module design, interfacce |
| `domain-modeling` | Modellare nuovi concetti dominio | ADR, ubiquitous language, CONTEXT.md |
| `find-skills` | Cercare nuove skill utili | Ricerca ecosistema skills.sh |

### Globale — Core dominio

| Skill | Trigger | File/Task |
|-------|---------|-----------|
| `crypto-derivatives` | GEX, gamma flip, dealer positioning, options flow, funding rate, barriere, max pain | `src/gex/*`, `src/edgar/barrier_utils.py`, `tabs/gex.py`, `tabs/barrier_map.py` |
| `quantitative-research` | Backtesting, alpha generation, factor models, regime detection, walk-forward, statistical arbitrage | `src/analytics/regime_analysis.py`, `src/analytics/factor_scorers.py`, `src/forecast/*` |
| `Time Series Analysis` | Trend, autocorrelation, Granger causality, forecasting, ARIMA, ACF/PACF | `src/analytics/granger.py`, `src/forecast/*`, `src/flows/correlation.py` |
| `portfolio-risk` | VaR, max drawdown, Sharpe/Sortino, correlation matrix, rolling metrics | `src/analytics/regime_analysis.py` |
| `scipy-best-practices` | Ottimizzazione, stat avanzata, interpolazione, signal processing | `src/analytics/*`, `src/forecast/calibration.py`, qualsiasi uso di `scipy.*` |
| `plotly` | Qualsiasi grafico Plotly | `src/dashboard/charts.py`, `tabs/gex.py`, `tabs/barrier_map.py`, `tabs/flows.py` |

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

nginx: `/api/*` → FastAPI, `/*` → Streamlit (con WebSocket `/_stcore/*`); rimuove
`X-Frame-Options` e imposta `Content-Security-Policy: frame-ancestors` per **wagmi-lab.com**
(embed iframe). App Platform non supporta volumi → i dati sono condivisi via **DB versionato
nel repo** (refresh EDGAR → commit → redeploy). Config nginx: `nginx.conf`; processi:
`supervisord.conf`. Modifiche alla spec DO: applicare da Console → Settings → App Spec.
Local mirror: `docker compose up -d --build` → http://localhost:8080 (dash) e /api/docs (API).
La dashboard gira anche in locale standalone (`make run-dashboard`, porta 8501).

## Fonti macro: CoinGlass preferito, CoinGecko di ripiego

`macro_fetcher.fetch_macro_data()` prova prima **CoinGlass** (tutti e cinque i
fattori, richiede `COINGLASS_API_KEY`). Se il funding manca ancora, ripiega su
**CoinGecko** — `GET /api/v3/derivatives`, che risponde anche senza chiave — e
ne ricava funding rate pesato per open interest e OI aggregato: due fattori su
cinque, ma il funding da solo pesa 0,30 del pilastro.

**Convenzione del funding — una sola, per entrambe le fonti.** CoinGlass e
CoinGecko restituiscono tutte e due punti percentuali per 8 ore (`0.01` = 0,01%),
quindi si annualizzano con `×3×365` e basta. Verificato confrontando le due
metriche OI-weighted nello stesso istante: 0,005477 e 0,007499, rapporto 0,73× —
se le convenzioni fossero diverse sarebbe ~100×.

La conversione sta **solo** in `src/flows/funding.py`: era duplicata in cinque
punti e in quattro applicava un `×100` di troppo, che dava 599% invece di 6% e
faceva leggere allo scorer "flush imminente" dove il mercato era tiepido. Non
reintrodurla nei chiamanti.

Per ETH valgono le stesse fonti con i simboli dello spec (`fetch_macro_data(asset="ETH")`);
`cron_macro.py` scrive uno snapshot per asset con un solo download CoinGecko.

`source_status` ha quattro stati: `ok`, `partial_coingecko` (ripiego attivo),
`no_api_key`, `unavailable`.

CoinGecko non ha storico, quindi la **variazione a 7 giorni dell'OI** arriva
dalla tabella `macro_snapshots` nel DB versionato, alimentata da
`scripts/cron_macro.py` + `.github/workflows/macro-snapshot.yml` (giornaliero,
committa il DB). Lo storico non può stare in `runtime.db`: il filesystem DO è
effimero e la finestra non maturerebbe mai. I due workflow che scrivono il DB
condividono il gruppo di concorrenza `db-write`.

## API multi-asset

`/api/gex`, `/api/flows`, `/api/macro` accettano `?asset=btc|eth` (default `btc`, 422 sugli
altri). Senza parametro la risposta è quella storica consumata da PTF-Dashboard (più il
campo additivo `asset`). Cache: BTC usa le chiavi storiche, ETH `<chiave>:eth`; lock di
dedup Deribit per asset. `/api/barriers` resta solo BTC.
Soglia di neutralità GEX per asset in `deribit.gex_threshold_usd_by_asset` (ETH 100k,
da ritarare quando ci sarà storico ETH). `cron_gex.py --asset {BTC,ETH,all}`.

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
(`_detect_underlying`, colonna `notes.underlying`), ma `get_active_barriers()` e
`compute_btc_prices()` operano **solo sulle note IBIT** (default) —
i prezzi/ratio IBIT non si applicano agli altri ETF. `data/runtime.db` (predizioni/cache runtime,
usato da `make run-api` via `DB_PATH`) è invece **ignorato** da git, separato dal seed versionato.
