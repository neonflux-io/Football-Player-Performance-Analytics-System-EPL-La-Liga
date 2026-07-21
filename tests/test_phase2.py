"""
tests/test_phase2.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for Phase 2 domain models: SeasonRecord, Player, Outfielder,
Goalkeeper, Team, League.

Run with:  pytest tests/test_phase2.py -v
─────────────────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import (
    Goalkeeper,
    League,
    Outfielder,
    Player,
    SeasonRecord,
    Team,
    create_player,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────
def make_season(
    season_id: int = 90,
    season_name: str = "2021/22",
    goals: int = 10,
    assists: int = 5,
    xg: float = 8.5,
    passes: int = 800,
    passes_completed: int = 700,
    minutes_played: float = 2700.0,
    dribbles_attempted: int = 40,
    dribbles_completed: int = 28,
    pressures: int = 150,
    carries: int = 200,
) -> SeasonRecord:
    return SeasonRecord(
        competition_id=11,
        season_id=season_id,
        season_name=season_name,
        team_name="FC Barcelona",
        minutes_played=minutes_played,
        goals=goals,
        assists=assists,
        xg=xg,
        passes=passes,
        passes_completed=passes_completed,
        dribbles_attempted=dribbles_attempted,
        dribbles_completed=dribbles_completed,
        pressures=pressures,
        carries=carries,
    )


def make_outfielder(
    player_id: int = 5503,
    name: str = "Lionel Messi",
    position: str = "Right Wing",
) -> Outfielder:
    p = Outfielder(player_id, name, position, "Argentina", 34)
    p.add_season(make_season())
    return p


def make_goalkeeper() -> Goalkeeper:
    gk = Goalkeeper(9999, "Marc-André ter Stegen", "Goalkeeper", "Germany", 29)
    rec = SeasonRecord(
        competition_id=11,
        season_id=90,
        season_name="2021/22",
        team_name="FC Barcelona",
        minutes_played=2700.0,
        saves=90,
        goals_conceded=30,
        clean_sheets=10,
        passes=300,
        passes_completed=270,
    )
    gk.add_season(rec)
    return gk


# ── SeasonRecord tests ────────────────────────────────────────────────────────
class TestSeasonRecord:
    def test_xg_overperformance_positive(self):
        r = make_season(goals=10, xg=8.0)
        assert r.xg_overperformance == pytest.approx(2.0)

    def test_xg_overperformance_negative(self):
        r = make_season(goals=3, xg=6.0)
        assert r.xg_overperformance == pytest.approx(-3.0)

    def test_pass_completion_rate(self):
        r = make_season(passes=100, passes_completed=80)
        assert r.pass_completion_rate == pytest.approx(80.0)

    def test_pass_completion_zero_passes(self):
        r = SeasonRecord(
            competition_id=11, season_id=1, season_name="x", team_name="y",
            passes=0, passes_completed=0,
        )
        assert r.pass_completion_rate == 0.0

    def test_shot_conversion_rate(self):
        r = SeasonRecord(
            competition_id=11, season_id=1, season_name="x", team_name="y",
            goals=5, shots=20,
        )
        assert r.shot_conversion_rate == pytest.approx(25.0)

    def test_dribble_success_rate(self):
        r = make_season(dribbles_attempted=40, dribbles_completed=28)
        assert r.dribble_success_rate == pytest.approx(70.0)

    def test_to_dict_contains_all_keys(self):
        r = make_season()
        d = r.to_dict()
        required = ["goals", "assists", "xg", "passes", "minutes_played"]
        for key in required:
            assert key in d, f"Key '{key}' missing from to_dict()"

    def test_save_percentage(self):
        r = SeasonRecord(
            competition_id=11, season_id=1, season_name="x", team_name="y",
            saves=80, goals_conceded=20,
        )
        assert r.save_percentage == pytest.approx(80.0)


# ── Outfielder tests ──────────────────────────────────────────────────────────
class TestOutfielder:
    def test_is_instance_of_player(self):
        p = make_outfielder()
        assert isinstance(p, Player)

    def test_per_90_goals(self):
        # 10 goals in 2700 min → 10/30 = 0.333.../90
        p = make_outfielder()
        assert p.per_90("goals") == pytest.approx(10 / 30, rel=1e-3)

    def test_per_90_unknown_season_returns_zero(self):
        p = make_outfielder()
        assert p.per_90("goals", season_id=999) == 0.0

    def test_primary_metrics_has_all_keys(self):
        p = make_outfielder()
        m = p.primary_metrics
        expected_keys = {"Goals/90", "Assists/90", "xG/90", "xA/90",
                         "Key Passes/90", "Dribbles/90", "Pressures/90", "Carries/90"}
        assert set(m.keys()) == expected_keys

    def test_career_totals_sums_across_seasons(self):
        p = Outfielder(1, "Test Player", "Forward")
        p.add_season(make_season(season_id=1, goals=10, minutes_played=900.0))
        p.add_season(make_season(season_id=2, goals=15, minutes_played=1800.0))
        totals = p.career_totals()
        assert totals["goals"] == 25
        assert totals["minutes_played"] == 2700.0

    def test_career_stats_df_shape(self):
        p = make_outfielder()
        p.add_season(make_season(season_id=2, season_name="2020/21"))
        df = p.career_stats_df()
        assert len(df) == 2
        assert "goals_p90" in df.columns

    def test_summary_is_string(self):
        p = make_outfielder()
        s = p.summary()
        assert isinstance(s, str)
        assert "Messi" in s

    def test_add_same_season_overwrites(self):
        p = make_outfielder()
        new_rec = make_season(season_id=90, goals=20)
        p.add_season(new_rec)
        assert p.get_season(90).goals == 20

    def test_equality_by_player_id(self):
        p1 = Outfielder(5503, "Messi", "Forward")
        p2 = Outfielder(5503, "Messi (updated)", "Right Wing")
        assert p1 == p2

    def test_hashable(self):
        p = make_outfielder()
        player_set = {p}
        assert p in player_set


# ── Goalkeeper tests ──────────────────────────────────────────────────────────
class TestGoalkeeper:
    def test_is_instance_of_player(self):
        gk = make_goalkeeper()
        assert isinstance(gk, Player)

    def test_primary_metrics_has_gk_keys(self):
        gk = make_goalkeeper()
        m = gk.primary_metrics
        assert "Saves/90" in m
        assert "Save %" in m
        assert "Clean Sheets" in m

    def test_save_percentage_calculation(self):
        gk = make_goalkeeper()
        m = gk.primary_metrics
        # 90 saves, 30 conceded → 90/120 = 75%
        assert m["Save %"] == pytest.approx(75.0)

    def test_summary_mentions_gk(self):
        gk = make_goalkeeper()
        s = gk.summary()
        assert "GK" in s


# ── Factory tests ─────────────────────────────────────────────────────────────
class TestCreatePlayer:
    def test_goalkeeper_position_returns_goalkeeper(self):
        p = create_player(1, "Test GK", "Goalkeeper")
        assert isinstance(p, Goalkeeper)

    def test_forward_returns_outfielder(self):
        p = create_player(2, "Test FW", "Center Forward")
        assert isinstance(p, Outfielder)

    def test_midfielder_returns_outfielder(self):
        p = create_player(3, "Test MF", "Central Midfield")
        assert isinstance(p, Outfielder)

    def test_keeper_in_name_returns_goalkeeper(self):
        p = create_player(4, "Test", "goal keeper")
        assert isinstance(p, Goalkeeper)


# ── Team tests ────────────────────────────────────────────────────────────────
class TestTeam:
    def test_add_and_retrieve_player(self):
        team = Team(1, "FC Barcelona", "La Liga", "2021/22")
        p = make_outfielder()
        team.add_player(p)
        assert team.get_player(5503) is p

    def test_squad_size(self):
        team = Team(1, "FC Barcelona", "La Liga", "2021/22")
        team.add_player(make_outfielder())
        team.add_player(make_goalkeeper())
        assert team.squad_size == 2

    def test_remove_player(self):
        team = Team(1, "FC Barcelona", "La Liga", "2021/22")
        team.add_player(make_outfielder())
        removed = team.remove_player(5503)
        assert removed is True
        assert team.squad_size == 0

    def test_remove_nonexistent_player_returns_false(self):
        team = Team(1, "FC Barcelona", "La Liga", "2021/22")
        assert team.remove_player(99999) is False

    def test_get_outfielders_filters_correctly(self):
        team = Team(1, "FC Barcelona", "La Liga", "2021/22")
        team.add_player(make_outfielder())
        team.add_player(make_goalkeeper())
        outfielders = team.get_outfielders()
        assert len(outfielders) == 1
        assert all(isinstance(p, Outfielder) for p in outfielders)

    def test_squad_stats_df_has_player_rows(self):
        team = Team(1, "FC Barcelona", "La Liga", "2021/22")
        team.add_player(make_outfielder())
        df = team.squad_stats_df(season_id=90)
        assert len(df) >= 1
        assert "goals" in df.columns

    def test_len_dunder(self):
        team = Team(1, "FC Barcelona", "La Liga", "2021/22")
        team.add_player(make_outfielder())
        assert len(team) == 1


# ── League tests ──────────────────────────────────────────────────────────────
class TestLeague:
    def _make_league_with_team(self) -> tuple:
        league = League(11, "La Liga", "Spain")
        team = Team(1, "FC Barcelona", "La Liga", "2021/22")
        team.add_player(make_outfielder())
        team.add_player(make_goalkeeper())
        league.add_team(team, season_id=90)
        return league, team

    def test_add_and_get_team(self):
        league, team = self._make_league_with_team()
        retrieved = league.get_team("FC Barcelona", season_id=90)
        assert retrieved is team

    def test_all_players_returns_both_subtypes(self):
        league, _ = self._make_league_with_team()
        players = league.all_players(season_id=90)
        assert len(players) == 2

    def test_find_player_case_insensitive(self):
        league, _ = self._make_league_with_team()
        p = league.find_player("messi")
        assert p is not None
        assert "Messi" in p.name

    def test_find_player_not_found_returns_none(self):
        league, _ = self._make_league_with_team()
        p = league.find_player("Cristiano Ronaldo")
        assert p is None

    def test_season_ids_property(self):
        league, _ = self._make_league_with_team()
        assert 90 in league.season_ids

    def test_build_from_stats_df(self):
        """Test the bridge method using a synthetic DataFrame."""
        df = pd.DataFrame([
            {
                "player_id": 1001,
                "player_name": "Test Forward",
                "team_name": "Team A",
                "position": "Center Forward",
                "goals": 15,
                "assists": 8,
                "xg": 12.5,
                "xa": 6.0,
                "passes": 500,
                "passes_completed": 420,
                "key_passes": 30,
                "shots": 60,
                "carries": 150,
                "dribbles_attempted": 45,
                "dribbles_completed": 30,
                "pressures": 180,
                "minutes_played": 2700.0,
            },
            {
                "player_id": 1002,
                "player_name": "Test Midfielder",
                "team_name": "Team A",
                "position": "Central Midfield",
                "goals": 5,
                "assists": 12,
                "xg": 4.0,
                "xa": 9.0,
                "passes": 1500,
                "passes_completed": 1350,
                "key_passes": 55,
                "shots": 20,
                "carries": 300,
                "dribbles_attempted": 35,
                "dribbles_completed": 22,
                "pressures": 400,
                "minutes_played": 3150.0,
            },
        ])

        league = League.build_from_stats_df(
            stats_df=df,
            competition_id=11,
            competition_name="La Liga",
            season_id=90,
            season_name="2021/2022",
            country="Spain",
        )

        assert len(league.all_players()) == 2
        player = league.find_player("Test Forward")
        assert player is not None
        assert player.get_season(90).goals == 15

    def test_repr_contains_key_info(self):
        league, _ = self._make_league_with_team()
        r = repr(league)
        assert "La Liga" in r
        assert "11" in r
