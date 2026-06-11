# CLAUDE.md — btc-institutional-flow (ibit-gamma-tracker)

Toolkit Python per l'impatto del **dealer hedging** su note strutturate IBIT sul prezzo BTC
(tesi Arthur Hayes). Espone un **backend FastAPI** + una dashboard Streamlit.

> Esiste già una `memory/MEMORY.md` nel repo con dettagli di architettura e bug-fix storici della
> dashboard Streamlit — leggerla per il dettaglio, **non duplicarla** qui.

## Comandi essenziali (Makefile)

```bash
make install        # pip install -e ".[dev]"
make run-api        # FastAPI → http://localhost:8000  (= python run_api.py)
make run-dashboard  # streamlit run src/dashboard/app.py
make test           # pytest tests/ -v  (~506 test)
make update-all     # update-gex + update-flows + update-edgar (cron data refresh)
```

Lint/type: `ruff` (configurato in `pyproject.toml`). Venv locale in `.venv`.

## Ruolo nell'ecosistema

Il FastAPI di questo repo (`run_api.py`, porta 8000) è il **BTC API consumato da PTF-Dashboard**
(lì configurato come `VITE_BTC_API_URL`). Modifiche end-to-end ai dati BTC della dashboard toccano
entrambi i repo.

## Architettura (moduli `src/`)

| Modulo | Ruolo |
|--------|-------|
| `src/edgar/` | SEC EDGAR scraper/parser note strutturate (424B2/424B3) → SQLite |
| `src/gex/` | Gamma Exposure da Deribit (`gex_calculator.py`, `deribit_client.py`): GEX, gamma flip, put/call wall, max pain |
| `src/flows/` | ETF flow tracker (Farside + yfinance), price fetcher BTC/IBIT, correlazioni |
| `src/analytics/` | Segnale composito (GEX+Flows+Barriers) + backtest |
| `src/dashboard/` | Dashboard Streamlit (`app.py`, `charts.py`) |
| `src/api/` | FastAPI app (`src.api.main:app`) servita da `run_api.py` |

DB: SQLite in `data/` (note, barrier_levels, prices). Config: `config/settings.yaml` via
`src.config.get_settings()`. Scheduler/cron in `scripts/` (`cron_gex.py`, `cron_signal.py`, ecc.).

## Deploy

`Procfile` → `python run_api.py --host 0.0.0.0 --port 8000`. `Dockerfile` + `docker-compose.yml`
disponibili. Git repo attivo (branch `main`).

## Refresh dati EDGAR (note IBIT)

Il DB `data/structured_notes.db` è **versionato** (fonte di verità: filesystem DO effimero).
Refresh incrementale: `scripts/cron_edgar.py` (env `EDGAR_LOOKBACK_DAYS`, default 14); full:
`make update-edgar`. Automazione: `.github/workflows/edgar-refresh.yml` (lunedì, committa il DB su
`main` → deploy). **Richiede** la Repository variable `EDGAR_USER_AGENT` (email reale) — senza, il
job fallisce di proposito (ToS SEC). In locale usare la stessa env var (placeholder `example.com` →
WARNING in `get_settings()`). I supplement *preliminari* hanno `is_preliminary=1` e `initial_level`/
`notional` = NULL; `/api/barriers` mostra solo i finali.

I search terms includono anche FBTC/BITB/ARKB: il parser estrae il ticker reale del sottostante
(`_detect_underlying`, colonna `notes.underlying`), ma `get_active_barriers()`,
`compute_btc_prices()` e `update_barrier_statuses()` operano **solo sulle note IBIT** (default) —
i prezzi/ratio IBIT non si applicano agli altri ETF. `data/runtime.db` (predizioni/cache runtime,
usato da `make run-api` via `DB_PATH`) è invece **ignorato** da git, separato dal seed versionato.
