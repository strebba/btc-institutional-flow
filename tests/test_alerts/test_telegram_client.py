"""Test per TelegramClient — mock httpx, verifica URL/payload/errori."""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import httpx
import pytest

from src.alerts.telegram_client import TelegramClient


def run(coro):
    return asyncio.run(coro)


class TestConstruction:
    def test_raises_on_empty_token(self) -> None:
        with pytest.raises(ValueError):
            TelegramClient(bot_token="", chat_id="123")

    def test_raises_on_empty_chat_id(self) -> None:
        with pytest.raises(ValueError):
            TelegramClient(bot_token="TOKEN", chat_id="")


class TestSendMessage:
    def test_success_returns_true_and_posts_correct_payload(self) -> None:
        client = TelegramClient(bot_token="ABC", chat_id="-100")

        captured: dict = {}

        async def fake_post(self, url, json=None):  # noqa: ANN001
            captured["url"] = url
            captured["json"] = json
            return httpx.Response(200, json={"ok": True})

        with patch.object(httpx.AsyncClient, "post", fake_post):
            ok = run(client.send_message("<b>hi</b>"))

        assert ok is True
        assert captured["url"] == "https://api.telegram.org/botABC/sendMessage"
        assert captured["json"]["chat_id"] == "-100"
        assert captured["json"]["text"] == "<b>hi</b>"
        assert captured["json"]["parse_mode"] == "HTML"
        assert captured["json"]["disable_web_page_preview"] is True

    def test_non_2xx_returns_false(self) -> None:
        client = TelegramClient(bot_token="ABC", chat_id="-100")

        async def fake_post(self, url, json=None):  # noqa: ANN001
            return httpx.Response(401, text="Unauthorized")

        with patch.object(httpx.AsyncClient, "post", fake_post):
            ok = run(client.send_message("x"))

        assert ok is False

    def test_http_error_returns_false_and_does_not_raise(self) -> None:
        client = TelegramClient(bot_token="ABC", chat_id="-100")

        async def fake_post(self, url, json=None):  # noqa: ANN001
            raise httpx.ConnectError("network down")

        with patch.object(httpx.AsyncClient, "post", fake_post):
            ok = run(client.send_message("x"))

        assert ok is False

    def test_disable_preview_flag_propagates(self) -> None:
        client = TelegramClient(bot_token="ABC", chat_id="-100")

        captured: dict = {}

        async def fake_post(self, url, json=None):  # noqa: ANN001
            captured["json"] = json
            return httpx.Response(200, json={"ok": True})

        with patch.object(httpx.AsyncClient, "post", fake_post):
            run(client.send_message("x", disable_web_page_preview=False))

        assert captured["json"]["disable_web_page_preview"] is False
