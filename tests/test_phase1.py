"""
tests/test_phase1.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for Phase 1 components: Config and StatsBombLoader.

Run with:  pytest tests/test_phase1.py -v
─────────────────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import cfg
from src.utils import per_90, safe_divide


# ── Config tests ──────────────────────────────────────────────────────────────
class TestConfig:
    def test_project_root_exists(self):
        assert cfg.PROJECT_ROOT.exists(), "Project root directory should exist"

    def test_cache_dir_created(self):
        assert cfg.CACHE_DIR.exists(), "Cache directory should be created by ensure_directories()"

    def test_supported_competitions_contains_target_leagues(self):
        assert "Premier League" in cfg.SUPPORTED_COMPETITIONS
        assert "La Liga" in cfg.SUPPORTED_COMPETITIONS

    def test_competition_ids_are_correct(self):
        # These are the canonical StatsBomb IDs — if they change, the project breaks
        assert cfg.SUPPORTED_COMPETITIONS["Premier League"] == 2
        assert cfg.SUPPORTED_COMPETITIONS["La Liga"] == 11

    def test_log_level_is_valid(self):
        valid = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
        assert cfg.LOG_LEVEL in valid


# ── Utils tests ───────────────────────────────────────────────────────────────
class TestUtils:
    def test_safe_divide_normal(self):
        assert safe_divide(10, 2) == 5.0

    def test_safe_divide_by_zero_returns_default(self):
        assert safe_divide(10, 0) == 0.0

    def test_safe_divide_custom_default(self):
        assert safe_divide(5, 0, default=-1.0) == -1.0

    def test_per_90_calculation(self):
        # 9 goals in 810 minutes → 1.0 per 90
        assert per_90(9, 810) == pytest.approx(1.0)

    def test_per_90_zero_minutes(self):
        assert per_90(5, 0) == 0.0


# ── StatsBombLoader unit tests (offline — mock the HTTP layer) ────────────────
class TestStatsBombLoaderOffline:
    """
    These tests exercise the loader's logic WITHOUT making real HTTP calls.
    We patch _fetch to return static fixture data.
    """

    def test_cache_path_is_deterministic(self):
        from src.loaders import StatsBombLoader
        loader = StatsBombLoader(use_cache=False)
        url = f"{cfg.STATSBOMB_BASE_URL}/competitions.json"
        p1 = loader._cache_path(url)
        p2 = loader._cache_path(url)
        assert p1 == p2, "Cache path must be deterministic for the same URL"

    def test_cache_path_uses_underscores_not_slashes(self):
        from src.loaders import StatsBombLoader
        loader = StatsBombLoader(use_cache=False)
        url = f"{cfg.STATSBOMB_BASE_URL}/matches/11/90.json"
        path = loader._cache_path(url)
        assert "/" not in path.name, "Cache filename should not contain slashes"

    def test_estimate_minutes_all_90_if_no_subs(self):
        from src.loaders import StatsBombLoader
        # Simulate events with no substitutions
        data = pd.DataFrame({
            "type_name": ["Pass", "Shot", "Pass"],
            "player_id": [101, 102, 101],
            "minute": [10, 55, 80],
        })
        result = StatsBombLoader._estimate_minutes(data)
        assert result.loc[result["player_id"] == 101, "minutes_played"].values[0] == 90.0
        assert result.loc[result["player_id"] == 102, "minutes_played"].values[0] == 90.0

    def test_lineups_returns_correct_columns(self, monkeypatch):
        from src.loaders import StatsBombLoader

        fixture = [
            {
                "team_id": 1,
                "team_name": "Barcelona",
                "players": [
                    {
                        "player_id": 5503,
                        "player_name": "Lionel Messi",
                        "jersey_number": 10,
                        "country": {"name": "Argentina"},
                        "positions": [{"position": {"name": "Right Wing"}}],
                        "cards": [],
                    }
                ],
            }
        ]

        loader = StatsBombLoader(use_cache=False)
        monkeypatch.setattr(loader, "_get", lambda endpoint: fixture)

        df = loader.lineups(12345)
        assert "player_name" in df.columns
        assert "team_name" in df.columns
        assert df.iloc[0]["player_name"] == "Lionel Messi"
        assert df.iloc[0]["positions"] == ["Right Wing"]
