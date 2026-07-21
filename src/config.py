"""
config.py
─────────────────────────────────────────────────────────────────────────────
Central configuration for the Football Analytics System.

OOP concept used: **Encapsulation** — all settings live in a single validated
Config object rather than scattered module-level globals.  Pydantic v2
enforces types at runtime so bad config fails fast with a clear message.

Usage:
    from src.config import cfg      # import the singleton
    print(cfg.CACHE_DIR)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings  # pydantic-settings handles env vars

# Load a .env file from the project root (if it exists) before Pydantic reads env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class Config(BaseSettings):
    """
    Project-wide configuration.

    All values can be overridden via environment variables or a .env file.
    Pydantic will validate types and raise clear errors on bad config.
    """

    # ── Paths ─────────────────────────────────────────────────────────────────
    PROJECT_ROOT: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent,
        description="Absolute path to the project root directory.",
    )

    @property
    def DATA_DIR(self) -> Path:
        return self.PROJECT_ROOT / "data"

    @property
    def RAW_DIR(self) -> Path:
        return self.DATA_DIR / "raw"

    @property
    def PROCESSED_DIR(self) -> Path:
        return self.DATA_DIR / "processed"

    @property
    def CACHE_DIR(self) -> Path:
        return self.DATA_DIR / "cache"

    @property
    def REPORTS_DIR(self) -> Path:
        return self.PROJECT_ROOT / "reports"

    @property
    def LOGS_DIR(self) -> Path:
        return self.PROJECT_ROOT / "logs"

    # ── StatsBomb Open Data ───────────────────────────────────────────────────
    STATSBOMB_BASE_URL: str = Field(
        default=(
            "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
        ),
        description="Base URL for StatsBomb open-data JSON files on GitHub.",
    )

    # ── Supported competitions ────────────────────────────────────────────────
    # StatsBomb competition IDs:
    #   2  = Premier League
    #  11  = La Liga
    SUPPORTED_COMPETITIONS: Dict[str, int] = Field(
        default={
            "Premier League": 2,
            "La Liga": 11,
        },
        description="Map of human-readable league name → StatsBomb competition_id.",
    )

    # Seasons available in StatsBomb open data for our target leagues
    # (season_id : season_name)  — add new seasons here to expand coverage
    SUPPORTED_SEASONS: Dict[int, str] = Field(
        default={
            # La Liga seasons (richest open-data set)
            1: "2017/2018",
            2: "2018/2019",
            4: "2020/2021",
            26: "2019/2020",
            27: "2015/2016",
            37: "2016/2017",
            42: "2019/2020",
            90: "2021/2022",
        },
        description="Map of season_id → human-readable season string.",
    )

    # ── HTTP / caching ────────────────────────────────────────────────────────
    REQUEST_TIMEOUT_SECONDS: int = Field(
        default=30,
        ge=5,
        description="Timeout in seconds for each HTTP request to StatsBomb.",
    )
    REQUEST_MAX_RETRIES: int = Field(
        default=3,
        ge=1,
        description="Number of retry attempts on transient HTTP errors.",
    )
    REQUEST_RETRY_BACKOFF: float = Field(
        default=1.5,
        description="Exponential backoff multiplier between retries.",
    )
    CACHE_ENABLED: bool = Field(
        default=True,
        description="If True, downloaded JSON is saved to disk to avoid re-downloading.",
    )

    # ── Analytics defaults ────────────────────────────────────────────────────
    MIN_MINUTES_FOR_ANALYSIS: int = Field(
        default=450,
        ge=90,
        description="Minimum minutes played before a player is included in rankings.",
    )
    PER_90_DENOMINATOR: float = Field(
        default=90.0,
        description="Used when calculating per-90-minute metrics.",
    )

    # ── Visualisation ─────────────────────────────────────────────────────────
    FIGURE_DPI: int = Field(default=150, ge=72)
    FIGURE_SAVE_FORMAT: str = Field(default="png")
    DARK_THEME: bool = Field(
        default=True,
        description="Use dark background for all figures.",
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = Field(default="INFO")
    LOG_ROTATION: str = Field(default="10 MB")

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}")
        return v.upper()

    def ensure_directories(self) -> None:
        """Create all required directories if they do not already exist."""
        dirs = [
            self.DATA_DIR,
            self.RAW_DIR,
            self.PROCESSED_DIR,
            self.CACHE_DIR,
            self.REPORTS_DIR,
            self.LOGS_DIR,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    model_config = {"env_prefix": "FA_", "extra": "ignore"}


# ── Singleton instance ────────────────────────────────────────────────────────
# Import `cfg` anywhere in the project — one source of truth.
cfg = Config()
cfg.ensure_directories()
