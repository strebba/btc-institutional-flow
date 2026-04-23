"""Alert system: Telegram notifier per GEX regime + ETF flow events.

Moduli:
  - telegram_client: wrapper httpx async per Telegram Bot API (parse_mode HTML)
  - alert_db: persistenza dedup/cooldown su SQLite (tabella alert_state)
  - templates: formattazione messaggi HTML per daily recap e flow events
  - gex_alert_monitor: orchestrator che legge dati, applica soglie, invia notifiche
"""
