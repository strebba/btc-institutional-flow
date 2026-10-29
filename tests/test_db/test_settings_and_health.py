"""Override FORECAST_SCHEDULER_ENABLED e backend esposto da /api/health."""
from __future__ import annotations

import pytest


@pytest.mark.parametrize("value,expected", [("false", False), ("0", False), ("true", True)])
def test_forecast_scheduler_env_override(monkeypatch, value, expected):
    from src.config import get_settings

    monkeypatch.setenv("FORECAST_SCHEDULER_ENABLED", value)
    get_settings.cache_clear()
    try:
        assert get_settings()["forecast"]["enabled"] is expected
    finally:
        get_settings.cache_clear()


def test_forecast_scheduler_env_garbage_ignored(monkeypatch):
    from src.config import get_settings

    get_settings.cache_clear()
    default = get_settings().get("forecast", {}).get("enabled", True)
    monkeypatch.setenv("FORECAST_SCHEDULER_ENABLED", "boh")
    get_settings.cache_clear()
    try:
        assert get_settings().get("forecast", {}).get("enabled", True) == default
    finally:
        get_settings.cache_clear()


def test_health_reports_backend_without_url(monkeypatch):
    from src.api.routers.health import health

    monkeypatch.setenv("DATABASE_URL", "")
    import json

    body = json.loads(health().body)
    assert body["data"]["database"] == "sqlite"
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:secret@h/db")
    body = json.loads(health().body)
    assert body["data"]["database"] == "postgres"
    assert "secret" not in json.dumps(body)
