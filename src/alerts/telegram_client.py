"""Client asincrono per Telegram Bot API.

Wrapper minimale su httpx — POST a /sendMessage con parse_mode HTML.
Non usa python-telegram-bot per evitare dipendenza pesante: servono solo
send_message, stesso pattern usato da PTF-Dashboard/backend/src/api/alert_monitor.py.
"""
from __future__ import annotations

import httpx

from src.config import setup_logging

_log = setup_logging("alerts.telegram")

_API_BASE = "https://api.telegram.org"


class TelegramClient:
    """Client HTTP minimale per Telegram Bot API.

    Args:
        bot_token: token fornito da @BotFather.
        chat_id: ID della chat (numero o @channel) che riceve i messaggi.
        timeout_s: timeout della singola richiesta HTTP (default 10s).
    """

    def __init__(self, bot_token: str, chat_id: str, timeout_s: float = 10.0) -> None:
        if not bot_token or not chat_id:
            raise ValueError("bot_token e chat_id sono obbligatori")
        self._token = bot_token
        self._chat_id = chat_id
        self._timeout = timeout_s

    async def send_message(
        self,
        text: str,
        *,
        disable_web_page_preview: bool = True,
    ) -> bool:
        """Invia un messaggio HTML alla chat configurata.

        Args:
            text: messaggio in HTML (parse_mode="HTML"). Max ~4096 char.
            disable_web_page_preview: evita l'anteprima per link nel messaggio.

        Returns:
            True se Telegram ha accettato (HTTP 2xx), False altrimenti.
            Non solleva eccezioni: i fallimenti di alert non devono bloccare
            il ciclo di scheduling.
        """
        url = f"{_API_BASE}/bot{self._token}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": disable_web_page_preview,
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=payload)
            if not resp.is_success:
                _log.error("Telegram send failed: %s — %s", resp.status_code, resp.text[:200])
                return False
            return True
        except httpx.HTTPError as exc:
            _log.error("Telegram send error: %s", exc)
            return False
