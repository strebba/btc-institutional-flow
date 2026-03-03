# Modulo 2 — Gamma Exposure Calculator

## Panoramica

Calcola il Gamma Exposure (GEX) aggregato delle opzioni BTC su Deribit, identifica il regime di mercato (positive/negative gamma) e genera alert operativi.

## Teoria del GEX

I market maker che vendono opzioni devono effettuare delta hedging continuativo per rimanere neutrali. Questo crea flussi meccanici e prevedibili sul sottostante.

**Formula GEX per singola opzione:**

```
GEX = sign × gamma × OI × contract_size × spot² × 0.01

dove:
  sign = +1  per call (MM short call → long gamma → compra in discesa, vende in salita)
  sign = -1  per put  (MM short put  → short gamma → vende in discesa, compra in salita)
  gamma      = sensibilità del delta al prezzo (da Deribit)
  OI         = open interest in contratti
  contract_size = 1.0 BTC (Deribit BTC options)
  spot²      = conversione da % a USD
  × 0.01     = normalizzazione (1% move)
```

**Interpretazione:**
- **GEX > 0** (positive gamma): i MM comprano quando il prezzo scende e vendono quando sale → il mercato si auto-stabilizza attorno allo spot
- **GEX < 0** (negative gamma): i MM vendono quando il prezzo scende e comprano quando sale → i movimenti si amplificano

## Fonte dati — Deribit Public API

Nessuna autenticazione richiesta. Rate limit: ~15 req/s.

```
GET https://www.deribit.com/api/v2/public/get_instruments?currency=BTC&kind=option&expired=false
GET https://www.deribit.com/api/v2/public/ticker?instrument_name=BTC-28MAR25-80000-C
GET https://www.deribit.com/api/v2/public/get_index_price?index_name=btc_usd
```

**Dati estratti per opzione:**
```python
{
    "instrument_name": "BTC-28MAR25-80000-C",
    "strike": 80000.0,
    "option_type": "call",          # "call" | "put"
    "gamma": 0.0000043,             # greeks.gamma
    "open_interest": 1250.0,        # OI in contratti BTC
    "mark_price": 0.023,            # in BTC
}
```

## Metriche calcolate

### GEX totale netto
```python
total_net_gex = sum(gex_by_strike)  # USD
```

### Gamma Flip Price
Strike dove il GEX cumulativo (calcolato dallo strike più basso al più alto) cambia segno. Al di sotto del flip il mercato è in negative gamma, sopra in positive gamma.

```python
cumulative = 0.0
for g in sorted(gex_by_strike, key=lambda x: x.strike):
    prev = cumulative
    cumulative += g.net_gex
    if prev < 0 < cumulative or prev > 0 > cumulative:
        flip_price = g.strike
        break
```

### Put Wall
Strike con il valore GEX più negativo. I MM hanno venduto molte put a quel livello → devono comprare BTC aggressivamente se il prezzo scende fino lì → **supporto meccanico**.

### Call Wall
Strike con il valore GEX più positivo. I MM hanno venduto molte call → devono vendere BTC se il prezzo sale fino lì → **resistenza meccanica**.

### Max Pain
Prezzo che massimizza il valore delle opzioni scadute (per i MM che le hanno vendute), e minimizza le perdite degli acquirenti. Calcolato sommando il payoff intrinseco di tutte le opzioni a ogni possibile prezzo di scadenza.

```python
max_pain = argmax_strike { sum_calls(max(0, strike_test - strike_opt) × OI)
                          + sum_puts(max(0, strike_opt - strike_test) × OI) }
```

## Classificazione regime

```python
threshold = settings["deribit"]["gex_threshold_usd"]  # default: 1_000_000

if total_net_gex > threshold:
    regime = "positive_gamma"
elif total_net_gex < -threshold:
    regime = "negative_gamma"
else:
    regime = "neutral"
```

## Alert generati da `RegimeDetector`

| Alert | Condizione | Significato operativo |
|-------|-----------|----------------------|
| `GAMMA FLIP` | Segno GEX cambiato rispetto allo snapshot precedente | Cambio di regime → alta attenzione nelle 24-48h successive |
| `NEAR PUT_WALL` | Spot entro 2% dal put wall | Supporto meccanico imminente; rimbalzo probabile |
| `NEAR CALL_WALL` | Spot entro 2% dal call wall | Resistenza meccanica imminente; possibile stallo |
| `GEX ESTREMO NEGATIVO` | GEX sotto il 10° percentile storico | Volatilità molto alta attesa; evitare posizioni direzionali grandi |
| `GEX ESTREMO POSITIVO` | GEX sopra il 90° percentile storico | Mercato pinned; volatilità implicita sopravvalutata |

## Script CLI

```bash
python scripts/run_gex.py

# Output esempio:
# INFO  Strumenti BTC attivi: 948
# INFO  Fetch completato: 948 opzioni valide
# INFO  GEX totale: +$41,523,400
# INFO  Regime: POSITIVE_GAMMA
# INFO  Gamma flip: $75,000
# INFO  Put wall:   $60,000 (-12.4% dallo spot)
# INFO  Call wall:  $75,000 (+9.5% dallo spot)
# INFO  Max pain:   $82,000
```

## Risultati con dati reali (marzo 2026)

```
Spot BTC:        $68,473
GEX totale:      +$41.5M  →  POSITIVE_GAMMA
Gamma Flip:      $75,000
Put Wall:        $60,000  (-12.4%)
Call Wall:       $75,000  (+9.5%)
Max Pain:        ~$82,000
Strumenti:       948 opzioni (474 call, 474 put)
OI totale:       432,483 contratti
```

## Utilizzo programmatico

```python
from src.gex.deribit_client import DeribitClient
from src.gex.gex_calculator import GexCalculator
from src.gex.regime_detector import RegimeDetector

client   = DeribitClient()
calc     = GexCalculator()
detector = RegimeDetector()

spot    = client.get_spot_price()
options = client.fetch_all_options("BTC")
snap    = calc.calculate_gex(options, spot)
state   = detector.detect(snap)

print(f"Regime: {state.regime}")
print(f"GEX: ${snap.total_net_gex/1e6:.1f}M")
print(f"Alerts: {state.alerts}")

# Converti in dict per serializzazione/dashboard
snap_dict = calc.gex_to_dict(snap)
```

## Limitazioni

- **Latenza dati Deribit**: il fetch di ~948 opzioni richiede ~2 minuti a causa del rate limit (0.07s/req)
- **Dati intraday**: Deribit aggiorna i greeks continuamente; lo snapshot è un'istantanea puntuale
- **GEX storico**: non disponibile da API pubblica — il sistema accumula snapshot nel DB nel tempo
- **Opzioni settimanali vs mensili**: Deribit offre sia opzioni settimanali che mensili; tutte sono incluse nel calcolo
