"""Client asincrono per Telegram Bot API.

Wrapper minimale su httpx — POST a /sendMessage con parse_mode HTML.
Non usa python-telegram-bot per evitare dipendenza pesante: servono solo
send_message, stesso pattern usato da PTF-Dashboard/backend/src/api/alert_monitor.py.

Retry: send_message/send_to riprovano fino a 3 volte con exponential backoff
(1s, 2s, 4s) in caso di errori di rete, prima di arrendersi.

Truncation: messaggi oltre TELEGRAM_MAX_LENGTH vengono troncati per evitare
errori 400 Bad Request da Telegram (limite ~4096 char).
"""
from __future__ import annotations

import asyncio

import httpx

from src.config import setup_logging

_log = setup_logging("alerts.telegram")

_API_BASE = "https://api.telegram.org"
_TELEGRAM_MAX_LENGTH = 4000  # Margine di sicurezza sotto il limite Telegram ~4096
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # secondi: 1s, 2s, 4s


def _truncate_html(text: str, max_len: int = _TELEGRAM_MAX_LENGTH) -> str:
    """Tronca il testo HTML se supera max_len, preservando il formato."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "…"


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

    async def _post_with_retry(self, payload: dict) -> bool:
        """POST con retry a exponential backoff (3 tentativi)."""
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(
                        f"{_API_BASE}/bot{self._token}/sendMessage",
                        json=payload,
                    )
                if resp.is_success:
                    return True
                if attempt == _MAX_RETRIES:
                    _log.error(
                        "Telegram send failed after %d attempts: %s — %s",
                        _MAX_RETRIES, resp.status_code, resp.text[:200],
                    )
                    return False
                _log.warning(
                    "Telegram send attempt %d/%d: HTTP %s — retrying…",
                    attempt, _MAX_RETRIES, resp.status_code,
                )
            except httpx.HTTPError as exc:
                if attempt == _MAX_RETRIES:
                    _log.error(
                        "Telegram send failed after %d attempts: %s",
                        _MAX_RETRIES, exc,
                    )
                    return False
                _log.warning(
                    "Telegram send attempt %d/%d: %s — retrying…",
                    attempt, _MAX_RETRIES, exc,
                )
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_RETRY_BASE_DELAY * (2 ** (attempt - 1)))
        return False

    async def send_message(
        self,
        text: str,
        *,
        disable_web_page_preview: bool = True,
    ) -> bool:
        """Invia un messaggio HTML alla chat configurata.

        Args:
            text: messaggio in HTML (parse_mode="HTML"). Troncato se oltre il limite.
            disable_web_page_preview: evita l'anteprima per link nel messaggio.

        Returns:
            True se Telegram ha accettato (HTTP 2xx), False altrimenti.
            Non solleva eccezioni: i fallimenti di alert non devono bloccare
            il ciclo di scheduling.
        """
        payload = {
            "chat_id": self._chat_id,
            "text": _truncate_html(text),
            "parse_mode": "HTML",
            "disable_web_page_preview": disable_web_page_preview,
        }
        return await self._post_with_retry(payload)

    async def send_to(self, chat_id: str, text: str) -> bool:
        """Invia un messaggio HTML a una chat arbitraria (es. risposta a un comando)."""
        payload = {
            "chat_id": chat_id,
            "text": _truncate_html(text),
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        return await self._post_with_retry(payload)

    async def set_webhook(self, webhook_url: str, secret_token: str = "") -> bool:
        """Registra il webhook su Telegram. Ritorna True se ok."""
        url = f"{_API_BASE}/bot{self._token}/setWebhook"
        payload: dict = {"url": webhook_url}
        if secret_token:
            payload["secret_token"] = secret_token
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=payload)
            ok = resp.is_success and resp.json().get("ok", False)
            if ok:
                _log.info("Telegram webhook registrato: %s", webhook_url)
            else:
                _log.warning("Telegram setWebhook fallito: %s", resp.text[:200])
            return ok
        except httpx.HTTPError as exc:
            _log.error("Telegram setWebhook error: %s", exc)
            return False

    async def set_commands(self, commands: list[dict]) -> bool:
        """Registra i comandi del bot visibili nel menu Telegram."""
        url = f"{_API_BASE}/bot{self._token}/setMyCommands"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json={"commands": commands})
            return resp.is_success and resp.json().get("ok", False)
        except httpx.HTTPError as exc:
            _log.error("Telegram setMyCommands error: %s", exc)
            return False
