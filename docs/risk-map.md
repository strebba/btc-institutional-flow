# Mappa di rischio (senza score)

La mappa non è un punteggio direzionale e non usa i pesi del dealer-flow.
Dice due cose, entrambe calcolate solo sulla volatilità e sulla distanza
dei livelli, con le barre giornaliere Binance già chiuse al momento della
previsione (la barra del giorno UTC in corso è esclusa).

Il backtest del 10/11/2026 ha trovato che gli input del dealer-flow (score,
GEX, flussi ETF, funding, OI, long/short) non aggiungono capacità previsiva
su direzione, ampiezza del movimento o tocco dei livelli. Quello che resta
in uso è la volatilità e la distanza misurata in quella volatilità.

## Cosa significa ogni numero

Orizzonte fisso: **5 giorni**.

| Campo | Significato |
| --- | --- |
| `vol_method` | `garch` se il GARCH(1,1) sui rendimenti giornalieri converge, altrimenti `ewma` (λ = 0,94). |
| `sigma_5d_log` | deviazione standard attesa del log-return a 5 giorni. |
| `move_1sigma_5d_usd` / `_pct` | movimento di 1 σ, in dollari (`prezzo × σ`) e in percento (`100 × σ`). Il percento è un log-return, non un rendimento semplice. |
| `expected_range_5d_usd` / `_pct` | range atteso fra massimo e minimo dei 5 giorni, `E[ln(H/L)] = σ × √(8/π)`. In dollari è `prezzo ×` quel valore. È il range del percorso, non la distanza del close. |
| `range_median_90d_pct` | mediana degli ultimi 90 range a 5 giorni già chiusi. «Normale» vuol dire questo, non una soglia scelta a mano. |
| `prob_range_gt_median_90d` | probabilità che il range dei prossimi 5 giorni superi quella mediana. È la frequenza dei range passati divisi per il σ noto allora (stessa unità di oggi). Sotto le 30 finestre si usa la formula browniana qui sotto. |
| `prob_range_gt_median_90d_brownian` | la stessa probabilità dalla legge del range di un browniano senza drift, `1 − F(mediana / σ)`. Nessun coefficiente. Sulle barre giornaliere 2023–2026 discrimina (AUC circa 0,62) ma è meno calibrata della frequenza (Brier 0,27 contro 0,24 della frequenza e 0,25 della costante). |
| `prob_range_gt_median_90d_empirical` | uguale al numero mostrato quando la storia basta. Resta esposto anche da solo, così si vede il confronto. |
| `distance_pct` | `(livello / prezzo − 1) × 100`, con segno. |
| `distance_vol` | `|ln(livello / prezzo)| / σ_5g`. È l'unica informazione che entra nella probabilità di tocco. |
| `touch_prob_5d` | probabilità che il prezzo tocchi il livello entro 5 giorni. |
| `touch_prob_reflection` | formula chiusa `2 (1 − Φ(z))`, sempre presente. |

`touch_prob_5d` è la **frequenza storica** delle escursioni passate (massimo
rialzo e massimo ribasso di ogni finestra, messi insieme) che hanno superato
la stessa distanza in unità di σ. Servono almeno 60 escursioni già chiuse;
prima di allora si usa la formula di riflessione. Il backtest sui max pain
salvati misurava questa idea (distanza / volatilità): Brier 0,086 e AUC 0,94.
L'AUC viene dall'ordinamento per distanza: formula e tabella empirica
ordinano allo stesso modo. Cambia la calibrazione (il Brier). La scelta
empirica è quella tarata sulla storia, senza coefficienti liberi.

Lo script `scripts/verify_risk_map.py` non ha i max pain delle coorti né le
barre orarie: rifà il conto su una griglia di distanze (0,25–2 σ) e sulle
barre giornaliere. Su quel disegno, dal 2023, la probabilità di tocco batte
la costante (Brier circa 0,17 contro 0,23, AUC circa 0,79; solo sulle
distanze vicine, z ≤ 1, Brier circa 0,21 e AUC circa 0,71). Non sono le
cifre 0,086 / 0,94: il campione e la finestra sono diversi.

I livelli considerati, se il prezzo c'è: put wall, call wall, gamma flip,
max pain, e ogni barriera EDGAR con `level_price_btc`.

## Dove sta

- Sulle **righe nuove** di `predictions`, nel JSON `metadata` (chiavi
  `risk_map` e `levels_pit`). Le righe già scritte restano con `metadata`
  NULL: non si aggiornano e gli esiti non si ricalcolano.
- `GET /api/predictions` aggiunge `risk_map` e `levels_pit` accanto ai campi
  esistenti (score compreso). Il campo `metadata` grezzo resta.
- `GET /api/forecast/risk-map` e `GET /api/forecast/status` espongono l'ultima
  mappa salvata. L'endpoint non ricalcola e non chiama Binance: legge solo
  ciò che il predict del mattino ha scritto.

Il brief può mostrare «range atteso 5gg» (`expected_range_5d_usd` /
`expected_range_5d_pct`) e «probabilità di tocco per livello»
(`levels[].touch_prob_5d`) senza togliere gli score.

## `barrier_snapshots`

La tabella è lo storico delle **barriere EDGAR**, non dei muri GEX. Era
vuota (o non cresceva) per tre motivi, nel codice:

1. Il job di produzione `scripts/cron_daily.py` non la scriveva. L'unico
   richiamo era lo scheduler interno dell'API alle 22:30 UTC, senza
   recupero se il processo non era sveglio a quell'ora. Lo script
   `cron_snapshot_barriers.py` citato nella memoria del repo non esiste.
2. Anche `scripts/cron_edgar.py` aggiornava `barrier_levels` e non faceva
   lo snapshot.
3. `snapshot_active_barriers` usava la data locale invece di UTC, contava
   i tentativi anche quando `INSERT OR IGNORE` non inseriva nulla, e su
   Postgres un singolo INSERT fallito abortiva la transazione: le righe
   buone dello stesso giorno venivano annullate.

Ora il conteggio è il numero di righe nuove, la data è UTC, ogni barriera
sta in un savepoint, e lo snapshot gira in `cron_daily` (passo non critico)
e alla fine di `cron_edgar`. Non c'è un backfill: uno snapshot di oggi non
è il livello di un giorno passato.

Put wall, call wall, gamma flip, max pain e il GEX del mattino stanno in
`levels_pit` dentro il metadata della coorte. `gex_snapshots` continua a
avere una sola riga per giorno (un aggiornamento pomeridiano la
sovrascrive); il metadata del predict non viene riscritto.

## Limiti

- Non è una previsione di direzione e non va letta come un segnale di
  ingresso. Un σ alto allarga il range; non dice da che parte.
- Il browniano è senza drift e senza salti. Su Bitcoin il range reale ha
  code più pesanti: `prob_range_gt_median_90d` può essere troppo bassa o
  troppo alta. Il campo empirico e lo script di verifica servono a vederlo.
- Le barre sono giornaliere (00:00 UTC), non la finestra oraria 08:00–08:00
  del backtest. Il prezzo di riferimento delle distanze è lo spot Deribit
  del predict, la volatilità è Binance: sono due prezzi vicini, non lo
  stesso print. Se `api.binance.com` risponde 451, si usa lo stesso libro
  su `data-api.binance.vision`.
- La probabilità di tocco è per un solo livello, non la probabilità di
  toccarne almeno uno in una lista.
- Il GARCH si rifitta ogni 7 giorni sui rendimenti passati e di nuovo
  sull'ultimo giorno. Non vede la barra ancora aperta.
- Non entra `etf_flow_3d_usd`, non entrano i pesi, e `freeze_weights` non
  cambia: la calibrazione dei pesi dealer-flow resta quella di prima.
