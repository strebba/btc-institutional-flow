# CONTEXT.md — Domain Glossary

Glossario dei termini canonici del dominio. Usa questi termini esattamente in
tutto il codice, i commit, i test e le discussioni.

## Asset

| Termine | Definizione |
|---------|-------------|
| **Asset** | Sottostante tracciato: `BTC` o `ETH`. Descritto da un `AssetSpec` in `src/assets.py`, l'unico punto in cui stanno simboli, URL e nomi di colonna per-asset. Default `BTC` ovunque |
| **LeadEtf** | ETF spot di riferimento dell'asset: `IBIT` per BTC, `ETHA` per ETH. Il suo flusso è la colonna `<lead>_flow` del `merged_df` (`ibit_flow`, `etha_flow`) |
| **AssetFeatures** | Cosa è disponibile per l'asset. BTC: gex, flows, macro, analytics, barriers, edgar. ETH: solo gex, flows, macro |

## Note Strutturate

| Termine | Definizione |
|---------|-------------|
| **StructuredNote** | Filing SEC 424B2/424B3 emesso da una banca (JPM, Morgan Stanley, Goldman Sachs, …) il cui rendimento dipende dal prezzo di IBIT (ETF Bitcoin di BlackRock) |
| **BarrierLevel** | Singolo prezzo knock-in / autocall / buffer su una nota, espresso in % dell'initial level e/o in USD (IBIT o BTC) |
| **BarrierCluster** | Gruppo di BarrierLevel entro 2% di distanza, aggregati per tipo e segno direzionale |
| **BarrierDirection** | Score direzionale 0-1 di una barriera: knock_in/buffer sotto spot = accelerante (0.15), autocall/knock_out sopra spot = supportivo (0.65), altrimenti = neutro (0.50). Allineato a `barrier_sign()` per coerenza dealer-flow |
| **Underlying** | Ticker del sottostante reale della nota (IBIT, FBTC, BITB, ARKB) — rilevato dal parser via `_detect_underlying` |
| **PreliminarySupplement** | Filing con `is_preliminary=1`, `initial_level` e `notional` a NULL — escluso da `/api/barriers` |
| **Issuer** | Filer SEC canonico, derivato da `_known_issuer_or_none()` allowlist — filing di emittenti non noti sono scartati |

## Gamma Exposure (GEX)

| Termine | Definizione |
|---------|-------------|
| **GexSnapshot** | Fotografia completa del GEX a un timestamp: total net GEX, gamma flip, put/call wall, max pain, profilo per strike |
| **GexByStrike** | GEX aggregato per un singolo strike price (call GEX, put GEX, net GEX, OI) |
| **GammaRegime** | Classificazione: `positive_gamma` (dealer stabilizzano) / `negative_gamma` (dealer amplificano) / `neutral` |
| **GammaFlip** | Prezzo al quale il GEX cumulativo cambia segno — sopra = gamma positiva, sotto = negativa |
| **PutWall** | Strike con massimo GEX negativo — supporto meccanico |
| **CallWall** | Strike con massimo GEX positivo — resistenza meccanica |
| **MaxPain** | Strike che minimizza il payoff totale delle opzioni a scadenza |

## Flussi ETF

| Termine | Definizione |
|---------|-------------|
| **EtfFlow** | Flusso netto giornaliero in USD per un singolo ticker ETF |
| **AggregateFlows** | Flusso aggregato multi-ticker di un asset (IBIT, FBTC, … per BTC; ETHA, FETH, … per ETH), con `lead_flow_usd` del LeadEtf |
| **MergedRecord** | Riga del `merged_df`: flussi + prezzi uniti tramite `FlowCorrelation.merge()` |
| **FlowDataSource** | Sorgente dati flussi nella waterfall: CoinGlass, Farside, SoSoValue, EDGAR N-PORT, yfinance |
| **IbitBtcRatio** | Rapporto `IBIT / BTC-USD` usato per convertire prezzi barriera da IBIT a BTC |

## Analisi statistiche

> Segnale composito a 4 pilastri, IFI, backtest, Information Coefficient, walk-forward,
> factor decomposition e sensitivity sono stati **rimossi il 2026-10-07**: il prodotto
> mostra solo dati osservati (livelli GEX, flussi ETF, barriere, derivati), senza punteggi.

| Termine | Definizione |
|---------|-------------|
| **GrangerLead** | Lag ottimale flussi→rendimenti da `find_optimal_lag()` su training set e validato su holdout — mitigazione data snooping |
| **FactorScorers** | Libreria di scoring a 8 fattori (`src/analytics/factor_scorers.py`), usata solo dal Forecast Spine |
| **AnnualizationFactor** | Le crypto tradano 365 giorni/anno — `sqrt(365)` in regime_analysis e correlation |

## Forecast Spine

| Termine | Definizione |
|---------|-------------|
| **Prediction** | Previsione verificabile prodotta da una source (dealer_flow, ema, portfolio) con target type, orizzonte e confidence |
| **Outcome** | Esito misurato di una Prediction (hit/miss, Brier score, signed error) |
| **TargetType** | `direction` (up/down/flat), `level` (reach/break/respect), `prob` (evento probabilistico) |
| **WeightsVersion** | Snapshot immutabile dei pesi attivi usati per generare predizioni — human-gated activation |
| **Calibration** | Processo che propone nuovi pesi dai risultati storici — mai auto-attiva, richiede `/api/weights/{id}/activate`. Usa `scipy.stats.binom.sf` per p-value senza overflow |
| **MacroData** | Dataclass unificato da `src/flows/macro_fetcher.py` con funding rate, OI, long/short, liquidazioni. Singola fonte di verità per `/api/macro` e dashboard |

## Infrastruttura

| Termine | Definizione |
|---------|-------------|
| **StructuredNotesDB** | SQLite versionato in git (`data/structured_notes.db`) — fonte di verità per note/barriere EDGAR. Path hardcodato, ignora `DB_PATH` |
| **RuntimeDB** | SQLite gitignorato (`data/runtime.db`) — segnali, predizioni, alert. Usato in dev via `DB_PATH` env var |
| **GexDB** | SQLite per snapshot GEX — path hardcodato a `data/structured_notes.db`, ignora `DB_PATH` |
| **CacheStore** | TTL cache in-memory con lock per ridurre chiamate upstream (Deribit, Farside) |
| **SchedulerManager** | Orchestrator dei 3 APScheduler in-process (alert Telegram, manutenzione/snapshot barriere, forecast) |
