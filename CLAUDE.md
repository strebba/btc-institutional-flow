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
make test           # pytest tests/ -v  (~513 test)
make test-unit      # pytest tests/unit/ -v -q (esclude integration)
make lint           # ruff check src/ tests/
make update-all     # update-gex + update-flows + update-edgar (cron data refresh)
```

Lint/type: `ruff` (configurato in `pyproject.toml`), `.pre-commit-config.yaml`.
Venv locale in `.venv`.

## Ruolo nell'ecosistema

Il FastAPI di questo repo (`run_api.py`, porta 8000) è il **BTC API consumato da PTF-Dashboard**
(lì configurato come `VITE_BTC_API_URL`). Modifiche end-to-end ai dati BTC della dashboard toccano
entrambi i repo.

## Architettura (moduli `src/`)

| Modulo | Ruolo |
|--------|-------|
| `src/edgar/` | SEC EDGAR scraper/parser note strutturate (424B2/424B3) → SQLite |
| `src/gex/` | Gamma Exposure da Deribit (`gex_calculator.py`, `deribit_client.py`): GEX, gamma flip, put/call wall, max pain |
| `src/flows/` | ETF flow tracker (Farside + yfinance, Coinglass, SoSoValue), price fetcher BTC/IBIT, correlazioni, EDGAR N-PORT, `macro_fetcher.py` (dati macro unificati) |
| `src/analytics/` | Segnale composito a 4 pilastri (`pillars.py` single source of truth) + `factor_scorers.py` (ex signal_model) + backtest (+ transaction costs 80bps) + IFI + Granger + regime analysis |
| `src/dashboard/` | Dashboard Streamlit — `app.py` orchestratore, `data_loader.py` (cached), `tabs/` (5 moduli), `charts.py` (Plotly), `header.py`, `sidebar.py`, `static/style.css` |
| `src/api/` | FastAPI — `main.py` orchestratore (~225 righe), `routers/` (6 file: health, gex, flows, barriers, signals, forecast), `cache.py`, `helpers.py`, `auth.py`, `scheduler.py`, `schemas.py` |
| `src/alerts/` | Alert Telegram (ETF flow check, daily recap, error notification, comandi /recap /status /help) via `apscheduler` + GEX alert monitor |
| `src/forecast/` | Predizioni dealer-flow, calibrazione pesi, validazione esiti, multi-source (EMA, portfolio, dealer-flow) |

DB: SQLite in `data/` (`structured_notes.db` versionato + `runtime.db` gitignorato).
`StructuredNotesDB` e `GexDB` puntano **sempre** a `structured_notes.db` (path hardcodato,
ignorano `DB_PATH`). `SignalDB`, `PredictionDB`, `AlertDB` rispettano `DB_PATH` (default
`structured_notes.db`, override `data/runtime.db` in dev). Config: `config/settings.yaml` +
`config/weights.yaml` via `src.config.get_settings()`. Scheduler/cron in `scripts/` (16 script).

## Deploy

`Procfile` → `python run_api.py --host 0.0.0.0 --port 8000`. `Dockerfile` + `docker-compose.yml`
disponibili. Git repo attivo (branch `main`). La dashboard Streamlit gira in locale
(`make run-dashboard`), non deployata su DO.

## Refresh dati EDGAR (note IBIT)

Il DB `data/structured_notes.db` è **versionato** (fonte di verità: filesystem DO effimero).
Refresh incrementale: `scripts/cron_edgar.py` (env `EDGAR_LOOKBACK_DAYS`, default 30); full:
`make update-edgar`. Automazione: `.github/workflows/edgar-refresh.yml` (lunedì + backup
mercoledì 06:30 UTC, committa il DB su `main` → deploy DO). Lo User-Agent SEC è in
`config/settings.yaml` (email reale) — non servono variabili esterne. Override opzionale
via env var `EDGAR_USER_AGENT`. In caso di fallimento, il workflow invia una notifica
Telegram (richiede `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` nei Repository secrets).
Endpoint di monitoraggio: `GET /api/health/edgar` (freschezza DB, conteggi).
I supplement *preliminari* hanno `is_preliminary=1` e `initial_level`/`notional` = NULL;
`/api/barriers` mostra solo i finali.

I search terms includono anche FBTC/BITB/ARKB: il parser estrae il ticker reale del sottostante
(`_detect_underlying`, colonna `notes.underlying`), ma `get_active_barriers()`,
`compute_btc_prices()` e `update_barrier_statuses()` operano **solo sulle note IBIT** (default) —
i prezzi/ratio IBIT non si applicano agli altri ETF. `data/runtime.db` (predizioni/cache runtime,
usato da `make run-api` via `DB_PATH`) è invece **ignorato** da git, separato dal seed versionato.
