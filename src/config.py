"""Caricamento centralizzato della configurazione e setup logging."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Radice del progetto (due livelli sopra src/config.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE  = PROJECT_ROOT / "config" / "settings.yaml"


@lru_cache(maxsize=1)
def get_settings() -> dict:
    """Carica settings.yaml e sovrascrive con variabili d'ambiente.

    Returns:
        dict: configurazione completa del progetto.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    with CONFIG_FILE.open() as fh:
        cfg: dict = yaml.safe_load(fh)

    # Override via env
    if ua := os.getenv("EDGAR_USER_AGENT"):
        cfg["edgar"]["user_agent"] = ua
    if db := os.getenv("DB_PATH"):
        cfg["database"]["path"] = db

    return cfg


def setup_logging(name: str = "ibit") -> logging.Logger:
    """Configura e restituisce un logger con handler su file e console.

    Args:
        name: nome del logger.

    Returns:
        logging.Logger: logger configurato.
    """
    cfg     = get_settings()
    level   = getattr(logging, cfg["project"]["log_level"], logging.INFO)
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = PROJECT_ROOT / cfg["project"]["log_file"]

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # già configurato

    logger.setLevel(level)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
