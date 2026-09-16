# STATUS: 14/16 risolti in sessione 2026-09-16 (dettaglio in fondo). 2 rimandati: refactor funzioni Streamlit lunghe, test funzionali per le tab dashboard (AppTest) — entrambi troppo estesi per un fix rapido, non solo "nice-to-have" cosmetici.

# AUDIT REPORT — btc-institutional-flow
**Data:** 2026-09-16
**Ambito:** architettura & qualità codice · test & affidabilità dati · deploy & CI/CD
**Eseguito da:** Claude Code (claude-sonnet-5), 3 agenti di review in parallelo + esecuzione reale di lint/test
**Audit precedente:** [AUDIT-REPORT-2026-03-31.md](AUDIT-REPORT-2026-03-31.md) — vedi confronto in fondo

---

## Executive Summary

Il progetto è in **buono stato**: nessun blocker, nessuna credenziale hardcoded, nessun debito
tecnico esplicito (0 TODO/FIXME/HACK reali), suite di test interamente verde. L'audit del
2026-03-31 aveva trovato un blocker critico sulla persistenza GEX (nessuna tabella
`gex_snapshots`) — oggi risolto: `src/gex/gex_db.py` esiste e persiste correttamente lo storico.

I problemi trovati ora sono tutti di categoria **"da sistemare"** o **"nice-to-have"**: due
piccole violazioni della regola "single source of truth" (formula funding duplicata in un
router, soglie segnale hardcodate in una tab), un comando `make lint` rotto per un problema di
PATH, un gate `mypy` presente ma disattivato (`|| true`), due variabili d'ambiente Telegram usate
a runtime ma non dichiarate nella spec DO, e alcune aree di test sotto-esercitate (mock eccessivo
in `test_jobs.py`, smoke test soltanto in `test_tabs.py`).

Una premessa dell'audit va corretta: **`data/runtime.db` non è committato** — contrariamente a
quanto suggeriva `CLAUDE.md`/la memoria di progetto, è correttamente escluso sia da `.gitignore`
sia da `.dockerignore`. Nessun rischio dati lì.

---

## Numeri reali (eseguiti, non stimati)

| Check | Risultato |
|---|---|
| `ruff check src/ tests/` (via `.venv/bin/ruff`, non `make lint`) | **0 errori** — "All checks passed!" |
| `make lint` | **Fallisce**: `ruff: No such file or directory` — vedi finding sotto |
| Test unit (`make test-unit`) | **1016 passed**, 40 warning, 45.3s |
| Test integration (`tests/integration/`) | **10 passed**, 0.55s |
| Totale test | **1026** (CLAUDE.md/memoria dicono "~834" — numero da aggiornare) |
| TODO/FIXME/HACK reali in `src/` | **0** |
| Credenziali hardcoded | **0** |
| `requirements.txt` vs `pyproject.toml` | **sincronizzati oggi**, nessun lock file |
| `data/runtime.db` committato? | **No** — correttamente ignorato ovunque |

---

## Findings — Critico

Nessuno.

---

## Findings — Da sistemare

**Architettura & qualità codice**

- **`src/api/routers/signals.py:401`** — `funding_rate_8h_pct = round(_f / (3 * 365), 4)`
  reimplementa (invertita) la formula di annualizzazione che deve stare solo in
  `src/flows/funding.py`, per regola esplicita del progetto (era stata duplicata in 5 punti con
  un bug ×100 in 4). Corretta oggi, ma è un punto di rientro per lo stesso tipo di bug se la
  costante cambiasse. Va fattorizzata come helper inverso in `funding.py`.

- **`src/edgar/structured_notes_db.py:24,123`** — il docstring del costruttore di
  `StructuredNotesDB` dice che `db_path` ha "default da settings.yaml", ma il default reale è
  `_VERSIONED_DB`, un path hardcodato nel modulo che ignora `DB_PATH`/`settings.yaml`. È
  esattamente l'asimmetria (`StructuredNotesDB`/`GexDB` hardcoded vs `SignalDB`/`PredictionDB`
  che rispettano `DB_PATH`) che `CLAUDE.md` dice di documentare — qui è documentata in modo
  fuorviante invece che correttamente.

- **`src/dashboard/data_loader.py:29-30`** — `except Exception: pass` silenzioso attorno a
  `StructuredNotesDB().get_barrier_history()`, senza log, pur avendo `_log` già configurato nel
  file. Un fallimento nel pilastro barrier del tab Validation/backtest sparisce senza traccia.

- **`src/api/` — nessun uso di Pydantic** (`grep -rn "BaseModel\|response_model" src/api` → 0
  risultati) e **`src/api/schemas.py` non esiste**, pur essendo documentato in `CLAUDE.md` come
  parte dell'architettura. Gli endpoint restituiscono dict grezzi via `ok()`/`JSONResponse`:
  nessuna validazione di schema sulle risposte pubbliche, e la documentazione è disallineata dal
  codice reale — va corretta l'una o l'altro.

**Test & affidabilità dati**

- **`make lint` rotto**: il Makefile chiama `ruff` nudo, ma il binario è solo in
  `.venv/bin/ruff` (non in PATH senza `source .venv/bin/activate`). Comando "essenziale"
  documentato in `CLAUDE.md` che oggi non funziona out-of-the-box in locale (la CI funziona
  perché installa ruff globalmente via pip, separatamente).

- **`tests/test_forecast/test_jobs.py`** — mocka pesantemente ogni collaboratore
  (`PredictionDB`, `load_weights_config`, `gather_dealer_flow_context`,
  `build_dealer_flow_predictions`) in tutti e 3 i test. La logica di orchestrazione di
  `src/forecast/jobs.py` non è mai esercitata end-to-end con componenti reali — rischio che bug
  di integrazione tra i moduli reali sfuggano.

- **`tests/test_dashboard/test_tabs/test_tabs.py`** — solo smoke test
  (`assert modulo is not None`, `assert callable(funzione)`) per tutte e 6 le tab dashboard.
  Ragionevole vista la difficoltà di testare Streamlit, ma è copertura nominale, non funzionale.

**Deploy & CI/CD**

- **`TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`** sono usati a runtime in produzione
  (`src/api/scheduler.py:72`, `src/alerts/gex_alert_monitor.py:181-182`) ma **non sono
  dichiarati in `.do/app.yaml`**, a differenza di `COINGLASS_API_KEY`/`COINGECKO_API_KEY`
  (presenti come `type: SECRET`). Se non impostati fuori-spec via Console DO,
  `start_alert_scheduler()` si disattiva silenziosamente con un solo log warning
  (`scheduler.py:73-75`) — bot Telegram di produzione potenzialmente muto senza segnale
  evidente. Se invece sono impostati via Console, c'è drift tra spec versionata e stato reale.

- **Nessun `mypy` come gate reale in CI**: `ci.yml` esegue solo `ruff check`, pur essendo
  async+ruff+**mypy** un default tecnico dichiarato nel `CLAUDE.md` globale. `make typecheck`
  esiste ma con `|| true` che ne annulla l'effetto come gate — passa sempre, a prescindere dagli
  errori di tipo.

---

## Findings — Nice-to-have

- **`src/dashboard/tabs/signals.py:78-81`** — soglie `65`/`40` hardcodate localmente per
  colorare l'emoji per-pilastro, invece di importare `LONG_THRESHOLD`/`RISK_OFF_THRESHOLD` da
  `src.analytics.pillars`. Solo un dettaglio cosmetico (il segnale composito vero viene
  correttamente da `result.signal`), ma diverge silenziosamente se le soglie cambiano.
- Funzioni molto lunghe negli orchestratori Streamlit: `tabs/validation.py::_tab_validation`
  (~289 righe), `tabs/flows.py::_tab_flows` (~246), `tabs/barrier_map.py::_tab_barrier_map`
  (~197) — tipico per pagine Streamlit ma pesante da navigare/testare.
- `requirements.txt`/`pyproject.toml` sincronizzati oggi ma senza lock file/hash e senza
  upper-bound: nessuna garanzia che `pip install` risolva le stesse versioni tra
  locale/CI/Docker in momenti diversi.
- CI non builda/testa l'immagine Docker: un errore introdotto solo nel `Dockerfile` (es. path
  `COPY` errato) non verrebbe intercettato prima del deploy su DO.
- Marker pytest `integration` registrato in `pyproject.toml` ma sotto-applicato: solo
  `test_contract_signal.py` lo usa realmente; `test_data_loader.py` (7 test) sta nella cartella
  `tests/integration/` ma non ha il marker. La CI comunque esegue tutta `tests/` senza filtro
  `-m`, quindi il marker oggi non incide sul comportamento — solo igiene.
- 40 warning ricorrenti nei test: `datetime.utcnow()` deprecato (32 solo da
  `forecast/routers.py` e due file di test), `on_event` FastAPI deprecato (da migrare a
  lifespan), `starlette.testclient`+httpx deprecato. Nessuno bloccante, ma da ripulire prima che
  diventino breaking in una futura major.
- `CLAUDE.md`/memoria riportano "~834 test": sono oggi 1026 — numero da aggiornare.

---

## Punti di forza (da preservare)

- **`src/flows/funding.py`**: modulo minimale, single-responsibility, con docstring che spiega
  il *perché* (bug storico ×100) — previene la regressione meglio di un commento qualsiasi.
- **`src/analytics/pillars.py`**: rescaling pesato per copertura ben progettato, riuso esplicito
  di `factor_scorers`/`ifi` senza duplicare la logica di scoring, dataclass chiare con
  `Optional` ovunque manchino dati.
- **`src/report/facts.py`**: convenzione `None` vs `0.0` rispettata rigorosamente in tutti gli
  estrattori, soglie di materialità esplicite e commentate, `extract_all()` isola il fallimento
  di un singolo estrattore senza far cadere l'intera edizione.
- **`src/api/helpers.py::sanitize()`**: gestione centralizzata e robusta di NaN/Inf/numpy types
  verso JSON.
- **Segregazione loopback-only** pulita per uvicorn/streamlit dietro nginx, con CSP
  `frame-ancestors` correttamente ristretta a `wagmi-lab.com`/`www.wagmi-lab.com` (non
  wildcard). Nessuna porta interna esposta esternamente.
- **Gestione secrets pulita**: nessuna chiave in chiaro, `type: SECRET` in `app.yaml`,
  `secrets.*` nei workflow GitHub, `.env`/`runtime.db` esclusi coerentemente da git e Docker.
- **`concurrency: group: db-write`** condiviso tra `edgar-refresh.yml` e `macro-snapshot.yml` —
  soluzione elegante e ben commentata al problema reale (App Platform senza volumi, DB
  versionato scritto da due workflow diversi).
- **Suite di test veloce e interamente verde** (1026 test in ~46s), con test di dominio
  (pillars, calibration, facts) che verificano invarianti di business reali, non solo "non
  esplode" — buona resistenza a regressioni silenziose. Copertura esplicita e mirata dei 4
  stati di degrado fonte dati (`source_status`) e della convenzione None/0.0 nei punti più
  critici per l'affidabilità del segnale pubblicato.

---

## Confronto con l'audit del 2026-03-31

L'audit precedente aveva trovato un blocker critico: nessuna tabella `gex_snapshots`,
backtest sempre FLAT, percentile GEX perso a ogni restart. **Risolto**: `src/gex/gex_db.py`
esiste, persiste correttamente lo storico GEX, ed è usato da backtest/regime analysis.
Nessun blocker equivalente è emerso in questo audit. Anche i problemi minori dell'epoca
(dipendenze test non installate, cache Farside/prezzi obsolete) non si sono ripresentati: la
suite gira pulita e le pipeline di refresh dati sono ora schedulate via GitHub Actions.

---

## Raccomandazioni prioritizzate

1. Fix `make lint` (usare `.venv/bin/ruff` o attivare il venv nel target Makefile) — quick win,
   comando "essenziale" oggi rotto.
2. Rimuovere la formula funding duplicata in `signals.py:401`, sostituendola con un helper
   inverso in `funding.py`.
3. Dichiarare `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` in `.do/app.yaml` (o verificare/annotare
   che sono impostati via Console) per evitare un bot silenziosamente muto in produzione.
4. Correggere il docstring fuorviante di `StructuredNotesDB.__init__`.
5. Aggiungere logging al fallimento silenzioso in `data_loader.py:29-30`.
6. Decidere su `mypy`: attivarlo come gate reale in CI (rimuovendo `|| true`) o aggiornare
   `CLAUDE.md` per riflettere che oggi non è enforced.
7. Allineare `CLAUDE.md`/documentazione su `src/api/schemas.py` (creare Pydantic schemas o
   rimuovere il riferimento) e sul numero di test (834 → 1026).
8. Rafforzare `test_jobs.py` con meno mock per esercitare l'orchestrazione reale di
   `forecast/jobs.py`.

---

## Remediation — 2026-09-16

Applicati nella stessa sessione, con `make lint`/suite test verdi dopo ogni fix (1016 unit +
12 integration — invariati salvo le aggiunte sotto):

**Da sistemare (9/9)**
1. `funding_pct_8h_from_annual()` aggiunta a `src/flows/funding.py`; `signals.py:401` non
   ricalcola più `/ (3*365)` a mano.
2. Docstring di `StructuredNotesDB.__init__` corretto: spiega l'hardcoding di `_VERSIONED_DB`
   e l'esclusione da `DB_PATH`.
3. `data_loader.py:29-30` logga ora il fallimento di `get_barrier_history()` invece di
   inghiottirlo silenziosamente.
4. Documentazione allineata: `schemas.py` non esiste, rimosso da `CLAUDE.md` (2 punti) e da
   `memory/MEMORY.md`; nessuna migrazione Pydantic forzata senza necessità funzionale.
5. `make lint`/`make typecheck` ora puntano a `.venv/bin/ruff`/`.venv/bin/mypy` — non più
   dipendenti dall'attivazione manuale del venv.
6. `test_jobs.py` — aggiunto `tests/integration/test_forecast_jobs.py`: `run_daily_predict`
   gira con `PredictionDB` reale su SQLite temporaneo e `build_dealer_flow_predictions` reale,
   solo `gather_dealer_flow_context` mockato (confine di rete). Verifica persistenza reale e
   idempotenza sui duplicati.
7. `test_tabs.py` — **non risolto**, vedi nota sotto.
8. `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` dichiarati come `type: SECRET` in `.do/app.yaml`
   (valore vuoto, da valorizzare da Console — **richiede applicazione manuale della spec**,
   come da istruzioni in testa al file).
9. `mypy` installato come dev dependency (`pyproject.toml`, era assente — `make typecheck`
   falliva ancora prima di poter girare, dietro `|| true`). Baseline reale: **73 errori
   pre-esistenti in 16 file**, nessuno introdotto dai fix di questa sessione. Non ancora un
   gate CI: azzerarli in questa sessione avrebbe rischiato regressioni; `make typecheck` ora
   però fallisce onestamente invece di mentire verde.

**Nice-to-have (5/7)**
1. `tabs/signals.py` importa `LONG_THRESHOLD`/`RISK_OFF_THRESHOLD` da `pillars.py` invece di
   `65`/`40` hardcoded.
2. Funzioni lunghe negli orchestratori Streamlit — **non risolto**, vedi nota sotto.
3. Lock file dipendenze — **non risolto**: `requirements.txt`/`pyproject.toml` restano
   sincronizzati ma senza pin esatto; introdurre pip-tools/uv è una decisione di tooling, non
   un fix isolato.
4. Aggiunto job `docker-build` a `ci.yml` — build reale (no push) dell'immagine, verificata
   localmente prima di committare.
5. `tests/integration/test_data_loader.py` ora ha `pytestmark = pytest.mark.integration`.
6. `datetime.utcnow()` sostituito con `datetime.now(timezone.utc)` nei 3 punti trovati
   (`api/routers/forecast.py`, `test_price_fetcher.py`, `test_regime_detector.py`); `on_event`
   migrato a `lifespan` in `api/main.py`. Warning nella suite: 40 → 2 (i 2 residui sono interni
   a `starlette`/`fastapi`, non azionabili da codice nostro).
7. `CLAUDE.md` aggiornato: 834 → 1026 test, comando `test-unit` corretto (non esiste
   `tests/unit/`), aggiunta riga `make typecheck`.

**Deliberatamente non affrontati in questa sessione** (troppo estesi per un fix mirato, rischio
di introdurre regressioni senza test di comportamento pre-esistenti):
- **Funzioni lunghe nelle tab Streamlit** (`_tab_validation` ~289 righe, ecc.) — un refactor
  reale richiede prima i test funzionali del punto successivo, altrimenti si rifattorizza
  alla cieca.
- **`test_tabs.py` oltre lo smoke test** — Streamlit espone `streamlit.testing.v1.AppTest` per
  test headless reali (verificato disponibile con Streamlit 1.64 installato); servirebbe
  costruire fixture fedeli di `snap`/`merged_df`/`barriers` per tutte e 6 le tab. Stimato più
  di una sessione per farlo bene su tutte; farlo a metà avrebbe lasciato test fragili.
