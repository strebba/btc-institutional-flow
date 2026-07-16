# ibit-gamma-tracker — Project Memory

## Code Quality & Validation Overhaul (session 2026-07-16 — 19 fixes)

**625 test passing** dopo tutti i fix (19 pre-existing lxml failures).

### Fase 1 — Fix critici
1. **DB duale risolto**: `StructuredNotesDB` e `GexDB` ora hanno path hardcodato a `data/structured_notes.db`, ignorano `DB_PATH`. L'API in dev (`make run-api`) usando `runtime.db` ora vede correttamente barriere e GEX storici. `SignalDB`, `PredictionDB`, `AlertDB` continuano a rispettare `DB_PATH`.
2. **Async lock documentato**: `_gex_fetch_lock` è `threading.Lock`, corretto perché tutti i chiamanti sono endpoint sync eseguiti in threadpool. Documentato il vincolo.
3. **Barrier sign corretto**: `_barrier_direction()` in `pillars.py` allineato a `barrier_sign()`. Autocall/knock-out sopra spot = supportivo (0.65, dealer compra su dip), knock-in/buffer sotto = accelerante (0.15). Aggiunta costante `_DIR_SUPPORTIVE`.
4. **data_loader ora testato**: Estratto `_get_backtest_context()` helper condiviso che sostituisce 4 blocchi duplicati. Creato `tests/test_dashboard/test_data_loader.py` con 7 test.

### Fase 2 — Fix statistici
5. **Binomial p-value**: `scipy.stats.binom.sf()` al posto di `math.comb()` → niente overflow per n>170.
6. **Volatilità BTC √365**: `price_fetcher.py` corretto da `252**0.5` a `365**0.5`. `backtest.py` già usava 365.
7. **Transaction costs**: 80 bps dedotti su ogni cambio posizione nel backtest. Configurato in `settings.yaml:backtest.transaction_cost_bps`.
8. **Magic 0.5 configurabile**: `scraper.py` legge `premium_to_flow_ratio` da `settings.yaml:flows.yfinance_fallback`.
9. **Regime coverage + data-age**: `backtest.py:run()` emette warning se < 1 anno di dati. `regime_coverage()` già esistente.

### Fase 3 — DRY
10. **Macro data fetch unificato**: Nuovo `src/flows/macro_fetcher.py` con `MacroData` dataclass + `fetch_macro_data()`. Sostituisce 2 dei 3 siti duplicati (`/api/signals` e `data_loader.load_macro()`). `/api/macro` tenuto separato (serve storico 90gg per charting).
11. **Backtest context helper**: `_get_backtest_context(days)` in `data_loader.py` sostituisce 4 blocchi identici di 15 righe.
12-13. **Skip**: `_to_score`/`score_to_signal` dedup e CoinGlass parsing helper — il costo di un terzo modulo supera il beneficio.

### Fase 4 — QoL
14. `ruff>=0.9` in dev-dependencies (`pyproject.toml`)
15. Marker `integration` registrato in `pyproject.toml`
16. `st.dataframe(width="stretch")` → `use_container_width=True` in `tabs/signals.py`
17. `@st.cache_data` su `load_prices_and_flows()` in `data_loader.py`
18. Backtest in try/except isolato in `/api/signals` — non contagia più l'endpoint
19. `misfire_grace_time` 6h → 24h in `scheduler.py`

### File creati
- `src/flows/macro_fetcher.py`
- `tests/test_dashboard/test_data_loader.py`
- `tests/test_dashboard/__init__.py`

### Metriche
- Righe nette: ~-150 (rimosse duplicazioni, aggiunta funzionalità)
- Test aggiunti: 7 nuovi + 2 estesi (barrier direction)
- DRY violations risolte: 4/6 critiche
- Bug fissati: barrier sign invertito, binomial overflow, volatilità BTC, transaction costs, DB path duale

## Codebase Restructuring (session 2026-07-13 — comprehensive refactor)

**513 test passing** dopo il refactor completo (12 pre-existing lxml failures in ambiente locale).

### Domain Modeling
- `CONTEXT.md` creato con glossario di 30+ termini canonici del dominio
- `RegimeState` → `GammaRegime` (src/gex/models.py + 10 file aggiornati)
- `signal_model.py` → `factor_scorers.py` (18 file aggiornati, rename + import propagate)
- `signal_model` rimane come alias di scoring library riusata da `pillars.py`

### API Splitting (src/api/)
- `main.py`: 1932 → 225 righe (-88%)
- Nuovi moduli: `auth.py`, `cache.py`, `helpers.py`, `scheduler.py`, `schemas.py`
- 6 router in `routers/`: `health.py`, `gex.py`, `flows.py`, `barriers.py`, `signals.py`, `forecast.py`
- `GET /api/health/scheduler` — nuovo endpoint per monitorare stato APScheduler
- Cache + lock anti-concorrenza spostati in `cache.py` con interfaccia pubblica

### Dashboard Splitting (src/dashboard/)
- `app.py`: 1615 → 220 righe (-86%), ora orchestratore puro
- Nuovi moduli: `data_loader.py` (9 `@st.cache_data`), `tabs/` (5 moduli: barrier_map, gex, flows, signals, edgar), `header.py`, `sidebar.py`, `static/style.css`
- CSS estratto da inline `st.markdown` a `static/style.css`
- `sys.path.insert` mantenuto in `app.py` (pip vecchio in venv non supporta editable install)

### Test
- `tests/conftest.py`: aggiunto `_clear_settings_cache` autouse fixture per `lru_cache`
- `tests/integration/test_contract_signal.py`: nuovo contract test API↔Dashboard (3 test)
- Test count: 513 passing (582 total con quelli esclusi per lxml)

### CI/CD + Quality
- `.pre-commit-config.yaml`: ruff check + format
- `Makefile`: aggiunti `test-unit`, `test-integration`, `lint`, `typecheck`
- `.gitignore`: aggiunti `.agents/`, `skills-lock.json`

### Key renames
- `RegimeState` → `GammaRegime` (src/gex/models.py)
- `signal_model.py` → `factor_scorers.py` (src/analytics/)
- `_get_gex_data` → `src.api.routers.gex._get_gex_data()` + re-exported in `barriers.py`

## Architecture
- **Language**: Python 3.9+, Streamlit dashboard
- **DB**: SQLite at `data/structured_notes.db` (notes + barrier_levels + prices tables),
  **versioned in git** (source of truth — DO filesystem is ephemeral; weekly refresh via
  `.github/workflows/edgar-refresh.yml` commits it to `main`). Runtime data lives in
  `data/runtime.db` (gitignored, `DB_PATH` env).
- **Packages**: `src/edgar`, `src/flows`, `src/gex`, `src/analytics`, `src/dashboard`
- **Config**: `config/settings.yaml` (loaded via `src/config.get_settings()`)
- **Tests**: ~506 tests in `tests/`, run with `.venv/bin/pytest tests/ -v`
- **Launch**: `streamlit run src/dashboard/app.py`

## Key File Paths
- Dashboard: `src/dashboard/app.py`
- Charts: `src/dashboard/charts.py`
- GEX calc: `src/gex/gex_calculator.py`
- GEX client: `src/gex/deribit_client.py`
- Flows scraper: `src/flows/scraper.py`
- Price fetcher: `src/flows/price_fetcher.py`
- EDGAR DB: `src/edgar/structured_notes_db.py`

## Bugs Fixed (session 2026-03-08)
1. **`gex_to_dict()` key mismatch**: Added `total_net_gex` (raw USD) and `n_instruments`
   alongside existing `total_net_gex_m` / `num_strikes` in `gex_calculator.py`
2. **`fetch_and_store` missing method**: `app.py` called non-existent `PriceFetcher.fetch_and_store()`.
   Fixed to use `pf.fetch("BTC-USD")` + `pf.fetch("IBIT")` directly.
3. **Deribit timeout**: Changed from 15s to 30s in `deribit_client.py`

## Dashboard Tab Structure (current)
1. 🎯 Barrier Map — visual chart of BTC barrier levels + contextual alerts
2. 📊 GEX — gamma exposure profile, regime box, regime analysis
3. 💰 ETF Flows — IBIT flows, KPI, alert boxes, Granger expander
4. 🚦 Segnali — 4-pillar composite (GEX/Barrier/ETF Flows/Macro): top-level gauge +
   4 sub-gauges + readable table + backtest results
5. 🔍 EDGAR Monitor — structured notes KPI, filings table, event study

## EDGAR Notes (2026-06)
- EDGAR search terms include FBTC/BITB/ARKB besides IBIT; the parser extracts the real
  underlying ticker (`_detect_underlying` in `parser.py`, column `notes.underlying`).
- IBIT-price semantics: `get_active_barriers()` (default `underlying="IBIT"`),
  `compute_btc_prices()` and `update_barrier_statuses()` only touch IBIT notes.
- Issuer comes from the EDGAR **filer** (`entity_name`) via `_known_issuer_or_none()`
  allowlist; filings whose filer is not a known note issuer are discarded in `parse_batch`
  (beware: new issuers are silently dropped until added to `_ISSUER_CANONICAL`).
- Preliminary supplements (`is_preliminary=1`) have NULL `initial_level`/`notional` and are
  excluded from `/api/barriers`.

## 4-Pillar Signal Architecture (2026-06)
- `src/analytics/pillars.py` is the **single source of truth** for the composite signal,
  consumed by both `/api/signals` and the Streamlit `_tab_signals`. It consolidates the
  three previous parallel engines (binary dashboard rules, `SignalModel`, `IFIModel`).
- Four pillars, each a 0-100 sub-score, blended with weight-rescaling
  (`PILLAR_WEIGHTS`, sums to 1.0): **gex** (Deribit regime + gamma-flip), **barrier**
  (EDGAR notes, directional + notional-weighted via prox kernel σ=10%), **etf_flows**
  (reuses IFI flow_momentum/trend/price_momentum + flow_3d), **macro** (reuses
  `signal_model` contrarian scorers: funding/oi/long-short/put-call/liquidations).
- `CompositeSignal.compute(inputs)` = live; `.compute_series(df, active_barriers)` =
  vectorized backtest, returns columns `*_score` + `composite_score` (preserves IFI's
  charting capability). `Backtest.run(..., composite=...)` is the new branch; the
  `signal_model=` and legacy branches are kept intact (do not break existing tests).
- **Barriers are now a weighted pillar**, not just a veto. `get_active_barriers()` SELECT
  now also returns `notional_usd` (needed for notional weighting).
- `SignalModel` / `IFIModel` remain as **reusable scoring libraries** (their `_score_*`
  funcs are imported by `pillars.py`); they are no longer top-level signals.
- `/api/signals` is backward-compatible: legacy `components`/`weights`/`score`/`signal`
  preserved (`components` = 7 legacy factors via `CompositeResult.legacy_components`),
  new additive field `pillars`. New endpoints: `/api/pillars/series?pillar=...` (generalized
  IFI replacement), `/api/notes` + `/api/notes/by-url` (EDGAR drill-down). `/api/ifi` is
  soft-deprecated (tag `deprecated`) but still served.

## Important Patterns
- `gex_to_dict()` returns both `total_net_gex` (raw USD) and `total_net_gex_m` (millions)
- `PriceFetcher` has no `fetch_and_store()` — use `fetch(ticker)` directly
- All Streamlit cache functions use `ttl=_REFRESH` (900s default)
- Composite signal logic is in `pillars.CompositeSignal`; the dashboard helper is
  `compute_composite()` in `app.py` (the old `_composite_signal()` was removed)
- `barrier_map()` chart function added to `charts.py`; pillar gauges:
   `composite_gauge()` + `pillar_gauges()` in `charts.py`

## Bugs Fixed (session 2026-07-08 — Phase 2)

1. **`/api/notes` filtro underlying asimmetrico**: `(n.underlying or "IBIT") == underlying`
   → `.upper()` su entrambi i lati per case-insensitivity. Il default None→IBIT rimane
   per backward compatibility ma le note senza underlying sono esplicitamente convertite.

2. **`shares_outstanding = 50_000_000` hardcoded**: Sostituito con `_get_ibit_shares_outstanding()`
   in `scraper.py:97` — waterfall: EDGAR N-PORT (dato SEC ufficiale) → yfinance
   (`Ticker.info["sharesOutstanding"]`) → fallback 50M. Cache 7 giorni.
   Aggiunta `get_latest_shares_outstanding()` in `edgar_nport.py:359`.

3. **`_tab_flows` ETF column overflow >6**: Layout a griglia adattivo: se ci sono più di 6 ETF,
   le metriche overflowano su righe multiple da 6 colonne ciascuna (`app.py:1050`).

4. **`price_fetcher._store_df` INSERT OR IGNORE → UPSERT**: `INSERT ... ON CONFLICT(ticker,date)
   DO UPDATE` — le correzioni retroattive dei prezzi yfinance vengono ora applicate
   (`price_fetcher.py:126`). Test `test_idempotent` aggiornato di conseguenza.

5. **`/api/pillars/series` TTL cache assente**: Aggiunto `"pillars_series": 900` nel dict
   `_TTL` (`main.py:54`). Il compute_series + Farside scrape è costoso e merita 15 min.

6. **`_error()` pattern atipico**: Rinominato in `_http_error()` (con alias `_error` per
   backward compat). Funzione invariata ma il nome documenta che restituisce un'eccezione
   da raise-are (`main.py:334`).

7. **`_tab_gex` regime analysis fallisce silenziosamente**: Sostituito `_log.debug(...)` con
   `st.info("Regime analysis non disponibile: dati storici GEX insufficienti.")` in
   `app.py:933`. L'utente ora sa quando il dato manca.

8. **`event_study_car` CI band**: Sostituita la banda uniforme (non corretta: CI calcolato
   solo sul giorno finale) con un error bar verticale al giorno finale della finestra,
   che mostra `[ci_lower, ci_upper]` come 95% CI. `charts.py:825`.

## Bugs Fixed (session 2026-07-08 — Phase 3)

1. **Dashboard caricamento parallelo**: I 3 fetch sequenziali (GEX, flows, barriers)
   ora usano `ThreadPoolExecutor(max_workers=3)` in `app.py:1481`. Tempo totale di
   caricamento = max(Deribit, Farside, SQLite) invece di sum. Aggiunto import
   `concurrent.futures` top-level.

2. **Timeout esterni**: Verificati tutti i moduli HTTP — tutti hanno timeout espliciti
   (Deribit 60s, CoinGlass 15s, Farside 30s, EDGAR N-PORT 20-30s). Nessuna modifica
   necessaria.

3. **`signal_model.compute_series()` ottimizzato**: Sostituito `scores.iloc[i] = ...`
   nel loop con list comprehension + `pd.Series(...)`. Evita copy-on-write warning di
   pandas e riduce overhead di assegnazione. Aggiunto docstring che rimanda a
   `CompositeSignal.compute_series()` per backtest vettorializzati.

4. **Gamma flip con interpolazione lineare**: `_find_gamma_flip()` ora interpola
   linearmente tra due strike quando il GEX cumulativo cambia segno
   (`gex_calculator.py:191`). Precisione migliorata da ±$1000 a ±$50 tipicamente.

5. **Clustering barriere deterministico**: `detect_clusters()` riscritto con algoritmo
   a scansione lineare (`barrier_utils.py:89`). Ordina per prezzo, merge barriere
   adiacenti entro `proximity_pct%`. Risultato indipendente dall'ordine di input.
   Nessuna dipendenza esterna aggiuntiva.

6. **`barrier_map` spacing orizzontale**: Ogni barriera ora ha una posizione x unica
   (`x = i/N`) invece di `x=[0,1]` per tutte. Evita la sovrapposizione totale quando
   ci sono 500+ barriere. `charts.py:100`.

## Bugs Fixed (session 2026-07-08 — Phase 4)

1. **Tracking storico barriere**: Nuova tabella `barrier_snapshots` in
   `structured_notes_db.py` (migrazione v3). Metodi: `snapshot_active_barriers()`
   (salva istantanea giornaliera) e `get_barrier_history(days)` (recupera storico
   come DataFrame). `barrier_snapshots` ha UNIQUE(barrier_id, snapshot_date).

2. **`_barrier_series()` con storico**: `pillars.py` ora accetta `barrier_history`
   opzionale. Se fornito, per ogni data storica recupera le barriere attive in quel
   giorno (nearest-neighbor ±1 giorno). Nuovo helper `_barrier_series_from_history()`.
   `compute_series()` aggiornato per propagare il parametro.

3. **`Backtest.run()` accetta `barrier_history`**: `backtest.py` aggiornato:
   `run()` e `_generate_signals()` accettano e propagano `barrier_history` a
   `CompositeSignal.compute_series()`. `run_backtest()` nella dashboard carica
   automaticamente lo storico da `StructuredNotesDB.get_barrier_history()`.

4. **Cron job `cron_snapshot_barriers.py`**: Nuovo script per snapshot giornaliero
   delle barriere attive. Da eseguire via cron/scheduler per accumulare lo storico
   necessario al backtest accurato del pilastro Barrier.

## Bugs Fixed (session 2026-07-08 — Phase 5)

1. **Indicatore freschezza dati in sidebar**: Sostituito "Ultimo aggiornamento"
   statico con indicatori OK/— per GEX, Flussi, Barriere. Colore verde = dati
   disponibili, rosso = mancanti. `app.py:495`.

2. **Ridondanza "Statistiche GEX" rimossa**: Sezione 4-metriche duplicava Gamma Flip,
   Put/Call OI già visibili nell'header KPI e nella GEX row. Mantenuti solo Max Pain
   e Strumenti BTC, spostati sotto il grafico `gex_walls`. `app.py:908`.

3. **Legenda Barrier Map condizionale**: Le 3 descrizioni statiche sostituite con
   conteggi reali per tipo (n barriere + nozionale). I tipi senza barriere vengono
   nascosti. Aggiunto supporto per `knock_out`. `app.py:648`.

4. **`flows_stacked_chart` color cycle automatico**: Sostituiti colori hardcoded con
   `plotly.colors.qualitative.Plotly` cycle. Ticker noti (branding) mantengono colore
   fisso, ticker nuovi ottengono colore dal cycle. `charts.py:585`.

## Bugs Fixed (session 2026-07-08 — Phase 1)

1. **`compute_composite()` crash in `_tab_signals`**: Avvolto in try/except con fallback
   warning in `app.py:1207`. La chiamata a `compute_composite()` nel Tab Segnali non
   aveva protezione — un DataFrame vuoto o errore in un pilastro bloccava l'intero tab.

2. **API cache non thread-safe**: Aggiunto `_cache_lock = threading.Lock()` in
   `main.py:44` a protezione di `_cache_get()` e `_cache_set()`. Il dict `_cache`
   era letto/scritto senza sincronizzazione in un server multi-thread.

3. **`load_macro()` mancante dal refresh manuale**: Aggiunta `load_macro` alla lista
   `fn.clear()` in `app.py:1497`. I dati CoinGlass rimanevano stantii dopo refresh
   manuale perché non venivano invalidati.

4. **Double JSON encoding in `_ok()`**: Classe `_NumpyEncoder` rimossa. `_sanitize()`
   unificata (main.py:300): ora gestisce np.integer, np.floating, np.bool_,
   np.ndarray, pd.Timestamp, e float NaN/Inf in un unico passaggio ricorsivo.
   `_ok()` chiama `_sanitize(payload)` direttamente e passa il risultato a
   `JSONResponse` → una sola serializzazione invece di tre.
    Aggiunto `import pandas as pd` a top-level per `pd.Timestamp` support.

## Bugs Fixed (session 2026-07-08 — Dashboard verification fixes)

1. **`flows_stacked_chart` crash per duplicato `legend`**: `_LAYOUT_BASE` contiene
   già `legend=dict(...)`. `flows_stacked_chart` passava `legend=dict(...)` esplicito
   PRIMA di `**_LAYOUT_BASE`, causando `TypeError: got multiple values for 'legend'`.
   Fix: diviso in due chiamate `update_layout` — la prima con `**_LAYOUT_BASE`, la
   seconda override solo legend. `charts.py:616`.

2. **`.last()` deprecato in pandas**: Due chiamate in `_tab_flows` (`merged_df.last("7D")`
   e `merged_df.last("30D")`) usavano la sintassi deprecata. Sostituite con boolean
   indexing: `df[df.index >= df.index.max() - pd.Timedelta(days=N)]`. `app.py:1012,1172`.

3. **Doppio messaggio "nessuna barriera"**: Quando le barriere esistevano ma senza
   `level_price_btc`, apparivano due messaggi informativi ridondanti. Unificati in
   un unico messaggio contestuale (barriere assenti vs prezzo BTC non calcolabile)
    con `return` dopo ciascuno. `app.py:688`.

## Housekeeping (session 2026-07-13)

1. **Commit Phase 5**: 20 file unstaged + 1 untracked (`cron_snapshot_barriers.py`)
   committati in `9da2515`. Include barrier snapshots (DB v3), freshness indicators,
   thread-safe cache, NumpyEncoder/`_sanitize()` unificata, ecc.

2. **Sync `pyproject.toml` ↔ `requirements.txt`**: `pyproject.toml` mancava
   `fastapi`, `uvicorn[standard]`, `apscheduler`, `httpx`. Aggiunti a production
   deps. `responses>=0.25` rimosso da dev deps (0 import nel codebase).
   `requirements.txt` allineato.

3. **Bug fix**: `get_latest_shares_outstanding()` in `edgar_nport.py:370` referenziava
   `EdgarNPortFetcher()` (classe inesistente). Corretto in `EdgarNportClient()`.

4. **File obsoleti rimossi**: `.DS_Store` (x2, root + `src/`), `farside_2026-03-05.csv`,
   `farside_cache.html`, log stantii (`dashboard.log`, `streamlit.log`, `forecast_api.out`,
   `ngrok.log`). Tag `archive/barrier-gex-confluence` rimosso.

5. **CLAUDE.md**: aggiunti moduli `src/alerts/` e `src/forecast/` mancanti dalla
   tabella architettura.

6. **Lint fix** (15 violazioni ruff → 0):
   - `structured_notes_db.py`: `import pandas as pd` mancante (F821)
   - `edgar_nport.py`: `EdgarNPortFetcher` → `EdgarNportClient` (F821)
   - `run_gex.py`, `api/main.py`: `import json` inutilizzato rimosso (F401)
   - `reparse_goldman.py`: E402 + F841 + F541 fix
   - `run_flows.py`: F541 f-string vuoto

7. **Test**: 582/582 passati, ruff pulito (0 errori).

## EDGAR Workflow Hardening (session 2026-07-13)

1. **Root cause**: 5 workflow consecutivi falliti (15 giu → 13 lug) perché
   `EDGAR_USER_AGENT` Repository variable era assente. DB fermo a 633 note.

2. **User-Agent in `settings.yaml`**: Sostituito placeholder `you@example.com` con email
   reale. Il workflow non dipende più da variabili esterne; override opzionale via env
   `EDGAR_USER_AGENT`. `config.py` warning ammorbidito (non bloccante).

3. **Workflow `edgar-refresh.yml` riscritto**:
   - Rimosso guard step `exit 1` (non più necessario).
   - Lookback default 30gg (da 14) — recupera fino a 4 settimane di buchi.
   - Backup cron mercoledì 06:30 UTC (oltre al lunedì).
   - `workflow_dispatch` con input `lookback_days` per catch-up manuale.
   - Step `Notify Telegram on failure` via `curl` + Bot API (richiede
     `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` nei Repository secrets).
   - Rimosso `[skip ci]` dal messaggio di commit automatico — bloccava il deploy DO.

4. **Endpoint `/api/health/edgar`**: Nuovo health check con `last_update`,
   `total_notes`, `total_barriers`, `active_barriers`, `stale_days`.
   `healthy = stale_days <= 14`.

5. **Catch-up**: Workflow eseguito con `lookback_days=60` → DB recuperato:
   655 note (+22), 682 barriere (+25), ultima issue_date 2026-07-06.

6. **Test**: 582/582 pass, ruff clean.

## Telegram Bot Hardening (session 2026-07-13)

1. **HTML escaping**: Tutti i valori dinamici nei template (`_esc()` via `html.escape`)
   applicati a regime, IFI, ticker, alert text. Previene break del formato HTML se
   dati contengono `<`, `>`, `&`.

2. **Startup/heartbeat message**: All'avvio scheduler il bot invia
   `🟢 BTC Institutional Flow — Bot avviato` con orario recap e soglie configurate.
   Metodo `send_startup_message()` in `GexAlertMonitor`.

3. **Error notification via Telegram**: Se `send_daily_recap()` fallisce per mancanza
   dati GEX, il bot invia un alert errore (`ALERT_ERROR = "error_notification"`)
   con cooldown 6h. L'utente non resta più al buio quando la pipeline dati è rotta.

4. **`GexDB.get_last_regime_label()`**: Metodo pubblico (sostituisce accesso a
   `_conn()` privato da `GexAlertMonitor._last_regime_label()`).

5. **Retry + truncation in `TelegramClient`**: `_post_with_retry()` con 3 tentativi
   a exponential backoff (1s, 2s, 4s) su `send_message` e `send_to`. Truncation
   automatica a 4000 char (`_TELEGRAM_MAX_LENGTH`) con `…` per evitare errori 400.

6. **Webhook riutilizza `_alert_monitor`**: Il webhook handler non crea più istanze
   nuove di `GexAlertMonitor` e `TelegramClient` a ogni `/recap`. Usa l'istanza
   globale salvata da `_start_alert_scheduler()`.

7. **Comandi `/status` e `/help`**: `/status` → ultimo recap, n° snapshot GEX,
   regime corrente, prossimo recap. `/help` → lista comandi.

8. **`scripts/notify_telegram.py`**: Script CLI condiviso per notifiche da CI/CD.
   Sostituisce il `curl` raw nei workflow GitHub Actions. Accetta messaggio da
   argomenti o stdin.

9. **CI notification su `ci.yml`**: Aggiunta notifica Telegram su fallimento
   (prima assente). Entrambi i workflow (`ci.yml`, `edgar-refresh.yml`) ora usano
   `scripts/notify_telegram.py`.

10. **Test**: 125/125 core tests passati (56 alerts + 69 api/gex). Lint ruff pulito.

## Useful Commands (aggiuntivi)
- `make test` — 582 test in ~25s
- `.venv/bin/ruff check src/ tests/ scripts/` — lint
- `pip install -e ".[dev]"` — ora installa anche FastAPI/uvicorn grazie al sync
- `gh workflow run "edgar-refresh.yml" -f lookback_days=60 --ref main` — catch-up manuale
- `python3 scripts/notify_telegram.py "test message"` — notifica Telegram da CLI

## Telegram Bot Commands
- `/recap` — Recap GEX + IFI + ETF flows aggiornato (anche da webhook in gruppo)
- `/status` — Stato bot: ultimo recap, snapshot GEX, regime, prossimo recap
- `/help` — Lista comandi disponibili
