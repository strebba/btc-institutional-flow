# Modulo 1 — SEC EDGAR Scraper

## Panoramica

Scarica e parsa i prospetti di emissione (424B2/424B3) delle note strutturate legate a IBIT depositate alla SEC. L'obiettivo è estrarre i livelli di barriera (knock-in, autocall, buffer) e convertirli in prezzi BTC assoluti.

## Fonte dati

**SEC EDGAR EFTS API** — Full-text search engine di EDGAR:

```
GET https://efts.sec.gov/LATEST/search-index?q=%22IBIT%22&dateRange=custom&startdt=2024-01-01&forms=424B2,424B3&from=0&size=10
```

**Header obbligatorio:**
```
User-Agent: ibit-gamma-tracker/1.0 (contact@example.com)
```

### Struttura risposta EFTS

```json
{
  "hits": {
    "hits": [
      {
        "_id": "0001213900-26-003766:ea0272591-01_424b2.htm",
        "_source": {
          "adsh": "0001213900-26-003766",
          "ciks": ["1665650"],
          "display_date_filed": "2026-01-15",
          ...
        }
      }
    ]
  }
}
```

**Costruzione URL documento:**
```python
adsh, doc_filename = _id.split(":", 1)
cik = ciks[-1]                           # emittente, non parent holding
acc_clean = adsh.replace("-", "")        # 0001213900-26-003766 → 000121390026003766
url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{doc_filename}"
```

## Parser

### Prodotti riconosciuti

| `product_type` | Trigger | Emittente tipico |
|----------------|---------|-----------------|
| `autocallable` | Autocall trigger + knock-in barrier | JPMorgan, Morgan Stanley |
| `barrier_note` | Solo knock-in barrier | JPMorgan |
| `buffered_note` | Buffer protection | Goldman Sachs |
| `leveraged_note` | Partecipazione > 100% | Vari |

### Barriere estratte

| `barrier_type` | Meccanismo | Direzione impatto su BTC |
|----------------|-----------|--------------------------|
| `knock_in` | Attivata se prezzo SCENDE sotto soglia | Dealer vende → sell-off accelera |
| `autocall` | Attivata se prezzo SALE sopra soglia | Dealer compra prima → supporto |
| `buffer` | Protezione parziale del capitale in calo | Simile a knock_in |
| `knock_out` | Prodotto chiuso se prezzo sale sopra soglia | Dealer vende → resistenza |

### Regex principali

```python
# Barriera con prezzo assoluto (JPMorgan)
# "Barrier Amount: 55% of the Initial Value, which is $28.138"
_RE_BARRIER_ABS = re.compile(
    r"Barrier\s+Amount[:\s]+(\d+(?:\.\d+)?)\s*%\s*of\s+the\s+Initial\s+Value"
    r"[^$]*\$\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Classificazione barriera: analizza contesto PRIMA del match
context = text[max(0, m.start() - 350):m.start()].lower()
# "autocall" / "call trigger" → autocall
# "knock-in" / "barrier" / "downside" → knock_in
# "buffer" / "protection" → buffer
```

### Calcolo prezzi BTC

```python
# Da prezzo IBIT assoluto
btc_price = ibit_price / ibit_btc_ratio  # ratio ≈ 0.0006

# Da percentuale + livello iniziale
btc_price = initial_level_ibit / ibit_btc_ratio * (barrier_pct / 100)

# Da barriera assoluta espressa in USD (JPMorgan pattern)
initial_level_ibit = barrier_price_ibit / (barrier_pct / 100)
btc_price = initial_level_ibit / ibit_btc_ratio
```

## Database SQLite

```bash
data/structured_notes.db
```

### Operazioni principali

```python
from src.edgar.structured_notes_db import StructuredNotesDB

db = StructuredNotesDB()

# Recupera barriere attive
barriers = db.get_active_barriers()
# → [{"barrier_type": "knock_in", "level_price_btc": 58800, ...}, ...]

# Aggiorna prezzi BTC usando ratio corrente
db.compute_btc_prices(ibit_btc_ratio=0.000612)

# Aggiorna status (triggered/active) in base al prezzo IBIT corrente
db.update_barrier_statuses(current_ibit_price=51.20)
```

## Script CLI

```bash
# Ricerca base (primi 500 filing)
python scripts/run_edgar.py

# Con limite e verbose
python scripts/run_edgar.py --max 20

# Output esempio:
# INFO  Trovati 547 filing EDGAR unici
# INFO  Parsed 8 note strutturate
# INFO  10 barriere attive nel DB
```

## Limitazioni note

1. **Farside bloccato da Cloudflare**: i flussi ETF vengono stimati da yfinance
2. **JPMorgan HTML**: i filing sono PDF convertiti — il testo viene estratto con `soup.get_text(separator=" ")` e normalizzato
3. **Initial level**: non sempre esplicitato — viene derivato dalla formula barriera quando possibile
4. **Nota scaduta**: le note con `maturity_date` passata rimangono nel DB con `status = expired` (non vengono rimosse automaticamente)
5. **CIK parent vs emittente**: EDGAR indicizza il filing sotto il CIK del parent holding (es. JPMorgan Chase & Co.), ma l'emittente reale è JPMorgan Chase Financial Company LLC (`ciks[-1]`)
