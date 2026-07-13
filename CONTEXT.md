# CONTEXT.md — Domain Glossary

Glossario dei termini canonici del dominio. Usa questi termini esattamente in
tutto il codice, i commit, i test e le discussioni.

## Note Strutturate

| Termine | Definizione |
|---------|-------------|
| **StructuredNote** | Filing SEC 424B2/424B3 emesso da una banca (JPM, Morgan Stanley, Goldman Sachs, …) il cui rendimento dipende dal prezzo di IBIT (ETF Bitcoin di BlackRock) |
| **BarrierLevel** | Singolo prezzo knock-in / autocall / buffer su una nota, espresso in % dell'initial level e/o in USD (IBIT o BTC) |
| **BarrierCluster** | Gruppo di BarrierLevel entro 2% di distanza, aggregati per tipo e segno direzionale |
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
| **AggregateFlows** | Flusso aggregato multi-ticker (IBIT, FBTC, BITB, ARKB, …) |
| **MergedRecord** | Riga del `merged_df`: flussi + prezzi uniti tramite `FlowCorrelation.merge()` |
| **FlowDataSource** | Sorgente dati flussi nella waterfall: CoinGlass, Farside, SoSoValue, EDGAR N-PORT, yfinance |
| **IbitBtcRatio** | Rapporto `IBIT / BTC-USD` usato per convertire prezzi barriera da IBIT a BTC |

## Segnale Composito

| Termine | Definizione |
|---------|-------------|
| **CompositeSignal** | Output 0-100 del modello a 4 pilastri (`src/analytics/pillars.py`), single source of truth |
| **Pillar** | Uno dei 4 componenti: `gex` (25%), `barrier` (25%), `etf_flows` (30%), `macro` (20%) |
| **PillarScore** | Sotto-score 0-100 di un singolo pilastro con componenti, peso, e motivazione |
| **PillarWeights** | Pesi nominali dei pilastri — riscalati se uno o più pilastri non hanno dati |
| **SignalThreshold** | Score ≥ 65 = LONG, 40-64 = CAUTION, < 40 = RISK_OFF |
| **FactorScorers** | Libreria di scoring a 8 fattori (`src/analytics/factor_scorers.py`, ex `signal_model`) — riusata dai pilastri |
| **IFIModel** | Institutional Flow Index 0-100 a 6 fattori — deprecato, sostituito dai pilastri |

## Forecast Spine

| Termine | Definizione |
|---------|-------------|
| **Prediction** | Previsione verificabile prodotta da una source (dealer_flow, ema, portfolio) con target type, orizzonte e confidence |
| **Outcome** | Esito misurato di una Prediction (hit/miss, Brier score, signed error) |
| **TargetType** | `direction` (up/down/flat), `level` (reach/break/respect), `prob` (evento probabilistico) |
| **WeightsVersion** | Snapshot immutabile dei pesi attivi usati per generare predizioni — human-gated activation |
| **Calibration** | Processo che propone nuovi pesi dai risultati storici — mai auto-attiva, richiede `/api/weights/{id}/activate` |

## Infrastruttura

| Termine | Definizione |
|---------|-------------|
| **StructuredNotesDB** | SQLite versionato in git (`data/structured_notes.db`) — fonte di verità per note/barriere EDGAR |
| **RuntimeDB** | SQLite gitignorato (`data/runtime.db`) — cache, snapshot, predizioni |
| **CacheStore** | TTL cache in-memory con lock per ridurre chiamate upstream (Deribit, Farside) |
| **SchedulerManager** | Orchestrator dei 3 APScheduler in-process (alert Telegram, IFI, forecast) |
