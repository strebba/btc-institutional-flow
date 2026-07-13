"""Invia una notifica Telegram da riga di comando (CI/CD).

Riutilizza TelegramClient (retry + truncation) invece di curl raw.
Accetta il messaggio come argomenti o da stdin.

Uso:
    python3 scripts/notify_telegram.py "messaggio da inviare"
    echo "messaggio" | python3 scripts/notify_telegram.py
    python3 scripts/notify_telegram.py --msg "messaggio"

Variabili d'ambiente:
    TELEGRAM_BOT_TOKEN    Obbligatorio
    TELEGRAM_CHAT_ID      Obbligatorio
"""
from __future__ import annotations

import asyncio
import os
import sys


def _main_sync() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        print("TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID non configurati — skip notifica", file=sys.stderr)
        sys.exit(0)

    text = ""
    args = sys.argv[1:]

    if args:
        if args[0] == "--msg":
            text = " ".join(args[1:])
        else:
            text = " ".join(args)
    else:
        text = sys.stdin.read().strip()

    if not text:
        print("Nessun messaggio fornito — skip", file=sys.stderr)
        sys.exit(0)

    from src.alerts.telegram_client import TelegramClient

    async def _send() -> bool:
        client = TelegramClient(bot_token=token, chat_id=chat_id, timeout_s=15.0)
        return await client.send_message(text)

    ok = asyncio.run(_send())
    if not ok:
        print("Notifica Telegram fallita", file=sys.stderr)
        sys.exit(1)
    print("Notifica Telegram inviata")
    sys.exit(0)


if __name__ == "__main__":
    _main_sync()
