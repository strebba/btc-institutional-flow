# Cutover a Postgres: DigitalOcean unica produzione

Obiettivo: **DigitalOcean è l'unica produzione**, tutti i dati (predizioni,
esiti, pesi, GEX, prezzi, note EDGAR, macro) vivono in **Postgres (Supabase)**,
e il **Mac mini resta solo per i test**.

Questa PR prepara il codice; **nessun passo qui sotto è stato eseguito**. Ogni
passo segnato con 🔐 richiede l'ok esplicito di Stefano.

## Situazione di partenza (audit 29/10/2026)

| | DigitalOcean | Mac mini |
|---|---|---|
| Commit | `main` | `2bc2644` |
| Predizioni | **0** (filesystem effimero, si azzera a ogni deploy) | **347** in `data/runtime.db` |
| Job giornaliero | nessuno (scheduler forecast in-process, ma senza persistenza) | 9:40: `cron_gex`, `run_coinglass`, `run_flows`, cache vault, `/daily-review` |
| Alert Telegram | spenti (secret vuoti) | **unica fonte** |
| Dati condivisi | `data/structured_notes.db` committato dai workflow (un commit + un deploy al giorno) | copia locale |

## Cosa cambia nel codice

- **`DATABASE_URL`** sceglie il backend (`src/db/`): vuota ⇒ SQLite in `data/`
  come prima (sviluppo, test, fallback); `postgres://` o `postgresql://` ⇒
  Postgres. Qualsiasi altro schema fa fallire l'avvio, così non si finisce a
  scrivere in silenzio su un file effimero.
- **Schema versionato** in `src/db/migrations/*.sql`, applicato
  automaticamente e in modo idempotente alla prima connessione (tabella
  `schema_migrations`, advisory lock contro le corse).
- **Migrazione dati** `scripts/migrate_sqlite_to_postgres.py`: idempotente,
  `--dry-run`, tutto o niente, id preservati.
- **Job giornaliero** `.github/workflows/daily-pipeline.yml` →
  `scripts/cron_daily.py`, ogni giorno alle 07:15 UTC (08:15 a Roma d'inverno):
  GEX, prezzi, verifica CoinGlass/Farside, predict e verify (calibrate il
  lunedì), tutto su Postgres. Sostituisce il LaunchAgent `com.wagmi.forecast-daily`.
- **`macro-snapshot.yml` e `edgar-refresh.yml`** non committano più il DB:
  scrivono su Postgres. Il repo smette di ricevere un commit (e un deploy) al giorno.
- **`.do/app.yaml`**: nuovi env `DATABASE_URL` (SECRET, vuota) e
  `FORECAST_SCHEDULER_ENABLED=false`, perché lo scheduler forecast
  in-process non deve fare da secondo scrittore. Telegram e CoinGlass erano già
  predisposti come SECRET vuoti.
- **`/api/health`** espone `"database": "postgres" | "sqlite"` (mai l'URL), per
  verificare il cutover.
- **API locale** (`start_api.sh`, `run_api.py`) ascolta di default solo su
  `127.0.0.1`. Override: `API_HOST=0.0.0.0`.

### Perché GitHub Actions e non un Job di DO App Platform

| | GitHub Actions | DO App Platform Job |
|---|---|---|
| Costo | **0 €**: il repo è pubblico, i minuti sono gratuiti | componente in più fatturato (dal piano basic in su) |
| Spec DO | invariata | va modificata e riapplicata |
| Pattern | identico a `edgar-refresh.yml` e `macro-snapshot.yml` (secret, notifica Telegram su errore) | nuovo |
| Rischi | il cron può ritardare anche di oltre 30 minuti; i workflow schedulati si **disattivano dopo 60 giorni** senza attività sul repo. Ora che i commit giornalieri del DB spariscono, serve almeno un commit ogni 60 giorni o una riattivazione manuale | più affidabile sull'orario |

Il ritardo è accettabile: le predizioni sono datate al giorno e c'è
l'idempotenza per giorno (`idx_pred_day_unique`).

## Passi di cutover (in ordine)

### 1. 🔐 Creare il progetto Supabase

1. Su supabase.com: New project, regione **eu-central-1 (Frankfurt)**, la
   stessa area del DO `fra`.
2. Generare una password robusta e salvarla nel password manager.
3. Prendere la connection string dal **Session pooler**: Project Settings →
   Database → Connection string → *Session pooler*, porta 5432, IPv4.
   La connessione diretta `db.<ref>.supabase.co` è solo IPv6 e i runner
   GitHub non la raggiungono. Anche il *Transaction pooler* (porta 6543)
   funziona, perché il codice disattiva i prepared statement.
4. Formato:
   `postgresql://postgres.<ref>:<password>@aws-0-eu-central-1.pooler.supabase.com:5432/postgres?sslmode=require`

Lo schema non va creato a mano: lo crea il primo processo che si connette, oppure la migrazione del passo 3.

### 2. 🔐 Impostare le variabili

**GitHub** (Settings → Secrets and variables → Actions):

| Secret | Uso |
|---|---|
| `DATABASE_URL` | obbligatorio per daily-pipeline, macro-snapshot ed edgar-refresh: senza, i job falliscono di proposito |
| `COINGLASS_API_KEY` | verifica CoinGlass nel job giornaliero (facoltativo: se manca, quel passo fallisce senza bloccare il job) |
| `COINGECKO_API_KEY` | facoltativo, come oggi |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | già usati per le notifiche di fallimento |

**DigitalOcean** (Console → Settings → App Spec, oppure Environment Variables
del componente `web`):

| Variabile | Valore |
|---|---|
| `DATABASE_URL` | la stessa stringa del passo 1, come SECRET |
| `FORECAST_SCHEDULER_ENABLED` | `false` |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | 🔐 **solo al passo 6**, per non avere alert doppi con il Mac |
| `COINGLASS_API_KEY` | facoltativo, completa il pilastro macro |

> La spec in `.do/app.yaml` **non viene applicata automaticamente**: va
> incollata da Console → Settings → App Spec. In alternativa si aggiungono
> solo le variabili sopra dalla UI.

### 3. 🔐 Migrare i dati

Da eseguire su una macchina con accesso alle copie dei DB del Mac. Copiare i
file con `sqlite3 ... ".backup"` o a LaunchAgent fermi (il WAL deve essere
consolidato):

```bash
# copie consistenti dal Mac (solo lettura sui file originali)
sqlite3 ~/…/btc-institutional-flow/data/runtime.db ".backup /tmp/mac-runtime.db"
sqlite3 ~/…/btc-institutional-flow/data/structured_notes.db ".backup /tmp/mac-notes.db"
git -C <clone aggiornato a main> show main:data/structured_notes.db > /tmp/main-notes.db

# 1) anteprima: nessuna scrittura
DATABASE_URL='postgresql://…' python3 scripts/migrate_sqlite_to_postgres.py \
    --runtime-db /tmp/mac-runtime.db \
    --notes-db   /tmp/main-notes.db \
    --gex-db     /tmp/mac-notes.db \
    --dry-run

# 2) esecuzione reale (stessi argomenti + --yes)
DATABASE_URL='postgresql://…' python3 scripts/migrate_sqlite_to_postgres.py \
    --runtime-db /tmp/mac-runtime.db --notes-db /tmp/main-notes.db \
    --gex-db /tmp/mac-notes.db --yes
```

- `runtime.db` del Mac contiene predictions (347), outcomes, weight_versions,
  alert_state e prices.
- Note EDGAR e `macro_snapshots` vanno prese da `main`, perché sono i
  workflow GitHub ad aggiornarle; i GEX dal Mac, dove gira `cron_gex`.
- Facoltativo: un secondo passaggio con `--gex-db /tmp/main-notes.db` aggiunge
  eventuali GEX presenti solo nel DB versionato. È idempotente: i giorni già
  copiati vengono saltati.
- Controllo: la colonna "conflitti id/orfane" deve essere 0. L'exit code è 1
  se non lo è.
- Rilanciarla è sicuro: copia solo ciò che manca e non sovrascrive.
- **Fare la migrazione prima del merge**, o comunque prima del primo run di
  `daily-pipeline`. Se il job scrive la predizione del giorno prima della
  migrazione, la riga del Mac per quello stesso giorno viene saltata e
  segnalata come conflitto.

### 4. 🔐 Merge della PR

Dopo il merge:

- **DO** ridispiega il codice (`deploy_on_push`). Senza `DATABASE_URL`
  l'app continua su SQLite come oggi, quindi il merge da solo non cambia nulla
  in produzione.
- **GitHub** attiva `daily-pipeline.yml`. Lanciarlo una volta a mano (Actions
  → Daily pipeline → Run workflow) e controllare il riepilogo.

### 5. Verifica su DO

```bash
APP=https://btc-institutional-flow-tpw9m.ondigitalocean.app
curl -s $APP/api/health          | jq .data.database        # "postgres"
curl -s "$APP/api/predictions"   | jq .data.count             # > 0 (default: ultime 50)
curl -s "$APP/api/calibration"   | jq .data.window_days
curl -s $APP/api/forecast/status | jq .data                  # total ≈ 347+, fresh=true
```

`scheduler_running` in `/api/forecast/status` sarà `false` di proposito:
predict e verify li fa il workflow.

### 6. 🔐 Spostare consumatori e alert dal Mac a DO

1. **Vault `/daily-review`**: puntare le letture da `http://localhost:8000` a
   `https://btc-institutional-flow-tpw9m.ondigitalocean.app/api/predictions` e
   `/api/calibration`. Se su DO è impostata `API_KEY`, aggiungere l'header
   `X-API-Key`.
2. **Disattivare i LaunchAgent del Mac**:
   ```bash
   launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.wagmi.forecast-daily.plist
   launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.wagmi.forecast-api.plist
   # per sicurezza, rinominare i plist (es. .plist.disabled) così non ripartono al login
   ```
3. **Telegram su DO**: valorizzare `TELEGRAM_BOT_TOKEN` e `TELEGRAM_CHAT_ID`
   nell'app DO **dopo** aver spento il Mac, per non ricevere alert doppi.
4. **API locale sul Mac** (solo test): aggiornare il clone a `main`; ora
   `start_api.sh` ascolta su `127.0.0.1`. Per i test locali lasciare
   `DATABASE_URL` vuota, e quindi SQLite: il Mac **non** deve scrivere sul
   Postgres di produzione.

### Rollback

- Togliere `DATABASE_URL` da DO: l'app torna su SQLite al deploy successivo.
- Riattivare i LaunchAgent del Mac: `runtime.db` non viene toccato dalla
  migrazione, che apre le sorgenti in sola lettura.
- I workflow `macro-snapshot` ed `edgar-refresh` senza `DATABASE_URL`
  falliscono e non perdono dati. Per tornare al commit del DB va fatto il
  revert di questa PR.

## Costi

| Voce | Costo |
|---|---|
| GitHub Actions | 0 € (repo pubblico) |
| DigitalOcean | invariato (stessa istanza `basic-xs`, nessun componente nuovo) |
| Supabase **Free** | 0 €: 500 MB di DB (oggi i dati sono pochi MB), **nessun backup automatico**, il progetto va in pausa dopo una settimana di bassa attività. Il job giornaliero e l'API dovrebbero bastare a evitarlo, ma non è garantito |
| Supabase **Pro** | da 25 $/mese: backup giornalieri (7 giorni), nessuna pausa. Consigliato se le predizioni devono essere a prova di perdita |

## Rischi noti

- **Pausa del progetto Free** ⇒ l'API su DO risponde errore e il job fallisce
  (arriva la notifica Telegram). Mitigazione: piano Pro, oppure un dump
  periodico (`pg_dump`).
- **Cron GitHub**: ritardi e disattivazione dopo 60 giorni di inattività del
  repo (GitHub manda un'email prima).
- **Nessun pool di connessioni**: una connessione per operazione. Va bene con
  il pooler di Supabase e con i volumi attuali. Se la latenza dell'API
  diventa un problema si può aggiungere `psycopg_pool`.
- **Cambio di comportamento**: `get_barrier_history` filtrava con
  `date('now','-365')`, che in SQLite restituisce NULL, quindi lo storico era
  sempre vuoto. Ora usa una data calcolata in Python e restituisce davvero
  l'ultimo anno di snapshot.
