"""
tests/test_phase3.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for Phase 3: PerformanceAnalyzer.

Uses a synthetic League fixture with 3 teams and 10 players so tests run
offline without any HTTP calls.

Run with:  pytest tests/test_phase3.py -v
─────────────────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analyzer import ComparisonResult, PerformanceAnalyzer, PercentileProfile, TrajectoryResult
from src.models import (
    Goalkeeper,
    League,
    Outfielder,
    SeasonRecord,
    Team,
    create_player,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixture: build a realistic mini-league
# ─────────────────────────────────────────────────────────────────────────────
def _make_record(
    season_id: int,
    season_name: str,
    team: str,
    goals: int,
    assists: int,
    xg: float,
    xa: float = 0.0,
    passes: int = 800,
    passes_completed: int = 680,
    key_passes: int = 30,
    shots: int = 40,
    carries: int = 200,
    dribbles_attempted: int = 30,
    dribbles_completed: int = 20,
    pressures: int = 150,
    minutes: float = 2700.0,
) -> SeasonRecord:
    return SeasonRecord(
        competition_id=11,
        season_id=season_id,
        season_name=season_name,
        team_name=team,
        minutes_played=minutes,
        goals=goals,
        assists=assists,
        xg=xg,
        xa=xa,
        passes=passes,
        passes_completed=passes_completed,
        key_passes=key_passes,
        shots=shots,
        carries=carries,
        dribbles_attempted=dribbles_attempted,
        dribbles_completed=dribbles_completed,
        pressures=pressures,
    )


@pytest.fixture(scope="module")
def mini_league() -> League:
    """
    Synthetic La Liga with 3 teams and 10 players across 2 seasons.

    Players (season 90 = 2021/22):
      Barcelona: Messi (FW), Xavi (MF), Puyol (CB), ter Stegen (GK)
      Madrid:    Ronaldo (FW), Modric (MF), Ramos (CB)
      Atletico:  Griezmann (FW), Koke (MF), Oblak (GK)
    """
    league = League(11, "La Liga", "Spain")

    # ── Barcelona ─────────────────────────────────────────────────────────────
    barca = Team(1, "FC Barcelona", "La Liga", "2021/22")

    messi = Outfielder(5503, "Lionel Messi", "Right Wing", "Argentina", 34)
    messi.add_season(_make_record(90, "2021/22", "FC Barcelona", goals=20, assists=12, xg=17.5, xa=8.0, minutes=3000.0))
    messi.add_season(_make_record(27, "2020/21", "FC Barcelona", goals=16, assists=9,  xg=13.0, xa=7.0, minutes=2700.0))

    xavi = Outfielder(5213, "Xavi Hernandez", "Central Midfield", "Spain", 41)
    xavi.add_season(_make_record(90, "2021/22", "FC Barcelona", goals=3, assists=10, xg=2.5, xa=7.0, passes=1800, passes_completed=1680, pressures=400))

    puyol = Outfielder(3109, "Carles Puyol", "Center Back", "Spain", 43)
    puyol.add_season(_make_record(90, "2021/22", "FC Barcelona", goals=2, assists=1, xg=1.5, passes=1200, passes_completed=1100, pressures=300))

    ter_stegen = Goalkeeper(5213 + 1, "Marc-André ter Stegen", "Goalkeeper", "Germany", 29)
    ter_stegen.add_season(SeasonRecord(
        competition_id=11, season_id=90, season_name="2021/22",
        team_name="FC Barcelona", minutes_played=2700.0,
        saves=95, goals_conceded=28, clean_sheets=12,
        passes=400, passes_completed=360,
    ))

    for p in [messi, xavi, puyol, ter_stegen]:
        barca.add_player(p)

    # ── Real Madrid ───────────────────────────────────────────────────────────
    madrid = Team(2, "Real Madrid", "La Liga", "2021/22")

    ronaldo = Outfielder(20055, "Cristiano Ronaldo", "Center Forward", "Portugal", 37)
    ronaldo.add_season(_make_record(90, "2021/22", "Real Madrid", goals=18, assists=6, xg=16.0, xa=3.0, shots=90, minutes=2900.0))

    modric = Outfielder(3501, "Luka Modric", "Central Midfield", "Croatia", 36)
    modric.add_season(_make_record(90, "2021/22", "Real Madrid", goals=5, assists=11, xg=4.0, xa=8.5, passes=1600, passes_completed=1488, key_passes=55))

    ramos = Outfielder(3082, "Sergio Ramos", "Center Back", "Spain", 35)
    ramos.add_season(_make_record(90, "2021/22", "Real Madrid", goals=4, assists=2, xg=3.0, passes=1300, passes_completed=1200, pressures=350, minutes=2500.0))

    for p in [ronaldo, modric, ramos]:
        madrid.add_player(p)

    # ── Atletico Madrid ───────────────────────────────────────────────────────
    atletico = Team(3, "Atletico Madrid", "La Liga", "2021/22")

    griezmann = Outfielder(10012, "Antoine Griezmann", "Center Forward", "France", 31)
    griezmann.add_season(_make_record(90, "2021/22", "Atletico Madrid", goals=12, assists=8, xg=11.0, xa=6.0, shots=60, minutes=2600.0))

    koke = Outfielder(5890, "Koke", "Central Midfield", "Spain", 30)
    koke.add_season(_make_record(90, "2021/22", "Atletico Madrid", goals=2, assists=7, xg=1.5, xa=5.5, passes=1700, passes_completed=1564, pressures=380))

    oblak = Goalkeeper(8000, "Jan Oblak", "Goalkeeper", "Slovenia", 29)
    oblak.add_season(SeasonRecord(
        competition_id=11, season_id=90, season_name="2021/22",
        team_name="Atletico Madrid", minutes_played=3060.0,
        saves=110, goals_conceded=22, clean_sheets=18,
        passes=350, passes_completed=310,
    ))

    for p in [griezmann, koke, oblak]:
        atletico.add_player(p)

    league.add_team(barca, season_id=90)
    league.add_team(madrid, season_id=90)
    league.add_team(atletico, season_id=90)

    # Also add season 27 teams (for trajectory tests with 2 seasons)
    barca_prev = Team(1, "FC Barcelona", "La Liga", "2020/21")
    barca_prev.add_player(messi)
    league.add_team(barca_prev, season_id=27)

    return league


@pytest.fixture(scope="module")
def analyzer(mini_league: League) -> PerformanceAnalyzer:
    return PerformanceAnalyzer(mini_league)


# ─────────────────────────────────────────────────────────────────────────────
# TrajectoryResult tests
# ─────────────────────────────────────────────────────────────────────────────
class TestCareerTrajectory:
    def test_returns_trajectory_result(self, analyzer):
        result = analyzer.career_trajectory("Messi")
        assert isinstance(result, TrajectoryResult)

    def test_df_has_one_row_per_season(self, analyzer):
        result = analyzer.career_trajectory("Messi")
        # Messi has 2 seasons loaded
        assert len(result.df) == 2

    def test_df_contains_per90_columns(self, analyzer):
        result = analyzer.career_trajectory("Messi")
        assert "goals_p90" in result.df.columns
        assert "xg_p90" in result.df.columns

    def test_peak_season_identified(self, analyzer):
        result = analyzer.career_trajectory("Messi")
        # Messi scored more in 2021/22 (20 goals) than 2020/21 (16)
        assert result.peak_season_name == "2021/22"

    def test_player_not_found_raises(self, analyzer):
        with pytest.raises(ValueError, match="not found"):
            analyzer.career_trajectory("Zlatan Ibrahimovic")

    def test_trajectory_contains_player_metadata(self, analyzer):
        result = analyzer.career_trajectory("Messi")
        assert result.player_name == "Lionel Messi"
        assert result.player_id == 5503

    def test_peak_value_is_positive(self, analyzer):
        result = analyzer.career_trajectory("Messi", primary_metric="Goals/90")
        assert result.peak_value > 0

    def test_trend_direction_is_valid_string(self, analyzer):
        result = analyzer.career_trajectory("Messi")
        assert result.trend_direction in {"improving", "declining", "stable"}


# ─────────────────────────────────────────────────────────────────────────────
# PercentileProfile tests
# ─────────────────────────────────────────────────────────────────────────────
class TestPercentileProfile:
    def test_returns_percentile_profile(self, analyzer):
        profile = analyzer.player_percentile_profile("Messi", season_id=90, min_minutes=100)
        assert isinstance(profile, PercentileProfile)

    def test_position_group_is_outfield(self, analyzer):
        profile = analyzer.player_percentile_profile("Messi", season_id=90, min_minutes=100)
        assert profile.position_group == "outfield"

    def test_goalkeeper_gets_gk_group(self, analyzer):
        profile = analyzer.player_percentile_profile("ter Stegen", season_id=90, min_minutes=100)
        assert profile.position_group == "goalkeeper"

    def test_percentiles_are_0_to_100(self, analyzer):
        profile = analyzer.player_percentile_profile("Messi", season_id=90, min_minutes=100)
        for label, pct in profile.percentiles.items():
            assert 0.0 <= pct <= 100.0, f"{label}: percentile {pct} out of range"

    def test_messi_goals_percentile_is_high(self, analyzer):
        profile = analyzer.player_percentile_profile("Messi", season_id=90, min_minutes=100)
        # Messi is top scorer — should be high percentile
        assert profile.percentiles.get("Goals/90", 0) >= 50

    def test_raw_values_match_per90(self, analyzer):
        profile = analyzer.player_percentile_profile("Messi", season_id=90, min_minutes=100)
        # Goals/90 for Messi: 20 goals / (3000/90) = 20/33.3 = 0.6
        expected = round(20 / (3000 / 90), 3)
        assert profile.raw_values.get("Goals/90", 0) == pytest.approx(expected, rel=1e-2)

    def test_player_not_found_raises(self, analyzer):
        with pytest.raises(ValueError, match="not found"):
            analyzer.player_percentile_profile("Nobody")

    def test_peer_count_is_positive(self, analyzer):
        profile = analyzer.player_percentile_profile("Messi", season_id=90, min_minutes=100)
        assert profile.n_peers > 0


# ─────────────────────────────────────────────────────────────────────────────
# ComparisonResult tests
# ─────────────────────────────────────────────────────────────────────────────
class TestComparePlayers:
    def test_returns_comparison_result(self, analyzer):
        result = analyzer.compare_players("Messi", "Ronaldo", season_id=90, min_minutes=100)
        assert isinstance(result, ComparisonResult)

    def test_player_names_populated(self, analyzer):
        result = analyzer.compare_players("Messi", "Ronaldo", season_id=90, min_minutes=100)
        assert "Messi" in result.player_a_name
        assert "Ronaldo" in result.player_b_name

    def test_metrics_dicts_are_non_empty(self, analyzer):
        result = analyzer.compare_players("Messi", "Ronaldo", season_id=90, min_minutes=100)
        assert len(result.metrics_a) > 0
        assert len(result.metrics_b) > 0

    def test_edge_has_entry_for_each_metric(self, analyzer):
        result = analyzer.compare_players("Messi", "Ronaldo", season_id=90, min_minutes=100)
        assert set(result.edge.keys()) == set(result.metrics_a.keys())

    def test_edge_values_are_valid(self, analyzer):
        result = analyzer.compare_players("Messi", "Ronaldo", season_id=90, min_minutes=100)
        valid_edges = {result.player_a_name, result.player_b_name, "draw"}
        for metric, winner in result.edge.items():
            assert winner in valid_edges, f"Edge '{winner}' not valid for {metric}"

    def test_comparison_df_has_correct_shape(self, analyzer):
        result = analyzer.compare_players("Messi", "Ronaldo", season_id=90, min_minutes=100)
        # Should have one row per metric
        assert len(result.comparison_df) == len(result.metrics_a)
        assert "Metric" in result.comparison_df.columns

    def test_messi_wins_assists(self, analyzer):
        result = analyzer.compare_players("Messi", "Ronaldo", season_id=90, min_minutes=100)
        # Messi: 12 assists, Ronaldo: 6 assists
        assert result.edge.get("Assists/90") == "Lionel Messi"

    def test_ronaldo_wins_shots(self, analyzer):
        result = analyzer.compare_players("Messi", "Ronaldo", season_id=90, min_minutes=100)
        # Ronaldo: 90 shots, Messi: 40 shots
        assert result.edge.get("Shots/90") == "Cristiano Ronaldo"

    def test_player_not_found_raises(self, analyzer):
        with pytest.raises(ValueError, match="not found"):
            analyzer.compare_players("Messi", "Zinedine Zidane", season_id=90)


# ─────────────────────────────────────────────────────────────────────────────
# League Rankings tests
# ─────────────────────────────────────────────────────────────────────────────
class TestLeagueRankings:
    def test_returns_dataframe(self, analyzer):
        df = analyzer.league_rankings(metric="Goals/90", season_id=90, min_minutes=100)
        assert isinstance(df, pd.DataFrame)

    def test_first_row_is_top_scorer(self, analyzer):
        df = analyzer.league_rankings(metric="Goals/90", season_id=90, min_minutes=100)
        # Messi has most goals (20 in 3000 min ≈ 0.60/90)
        assert "Messi" in df.iloc[0]["player_name"]

    def test_respects_top_n(self, analyzer):
        df = analyzer.league_rankings(metric="Goals/90", season_id=90, min_minutes=100, top_n=3)
        assert len(df) <= 3

    def test_invalid_metric_raises(self, analyzer):
        with pytest.raises(ValueError, match="not found"):
            analyzer.league_rankings(metric="Magic/90", season_id=90, min_minutes=100)

    def test_ranks_start_at_one(self, analyzer):
        df = analyzer.league_rankings(metric="Goals/90", season_id=90, min_minutes=100)
        assert df.index[0] == 1

    def test_gk_rankings_excluded_from_outfield(self, analyzer):
        df = analyzer.league_rankings(
            metric="Goals/90", season_id=90, position_group="outfield", min_minutes=100
        )
        names = df["player_name"].tolist()
        assert "ter Stegen" not in names
        assert "Oblak" not in names


# ─────────────────────────────────────────────────────────────────────────────
# Positional Benchmark tests
# ─────────────────────────────────────────────────────────────────────────────
class TestPositionalBenchmark:
    def test_returns_dataframe(self, analyzer):
        df = analyzer.positional_benchmark("Messi", season_id=90, min_minutes=100)
        assert isinstance(df, pd.DataFrame)

    def test_target_player_is_marked(self, analyzer):
        df = analyzer.positional_benchmark("Messi", season_id=90, min_minutes=100)
        assert df.iloc[0]["is_target"] is True or df.iloc[0]["is_target"] == True

    def test_player_not_found_raises(self, analyzer):
        with pytest.raises(ValueError, match="not found"):
            analyzer.positional_benchmark("Unknown Player")


# ─────────────────────────────────────────────────────────────────────────────
# Team Analysis tests
# ─────────────────────────────────────────────────────────────────────────────
class TestAnalyseTeam:
    def test_returns_team_report(self, analyzer):
        from src.analyzer import TeamReport
        report = analyzer.analyse_team("FC Barcelona", season_id=90)
        assert isinstance(report, TeamReport)

    def test_correct_total_goals(self, analyzer):
        report = analyzer.analyse_team("FC Barcelona", season_id=90)
        # Messi 20 + Xavi 3 + Puyol 2 = 25
        assert report.total_goals == 25

    def test_top_scorer_is_messi(self, analyzer):
        report = analyzer.analyse_team("FC Barcelona", season_id=90)
        assert "Messi" in report.top_scorer_name

    def test_goal_contribution_df_not_empty(self, analyzer):
        report = analyzer.analyse_team("FC Barcelona", season_id=90)
        assert not report.goal_contribution_df.empty
        assert "contribution_share_%" in report.goal_contribution_df.columns

    def test_team_not_found_raises(self, analyzer):
        with pytest.raises(ValueError, match="not found"):
            analyzer.analyse_team("Juventus FC", season_id=90)


# ─────────────────────────────────────────────────────────────────────────────
# Player Report tests
# ─────────────────────────────────────────────────────────────────────────────
class TestPlayerReport:
    def test_returns_dict(self, analyzer):
        report = analyzer.player_report("Messi", season_id=90, min_minutes=100)
        assert isinstance(report, dict)

    def test_report_has_required_keys(self, analyzer):
        report = analyzer.player_report("Messi", season_id=90, min_minutes=100)
        required = {"player", "summary", "career_totals", "trajectory", "percentile_profile", "primary_metrics"}
        assert required.issubset(report.keys())

    def test_player_section_has_correct_name(self, analyzer):
        report = analyzer.player_report("Messi", season_id=90, min_minutes=100)
        assert report["player"]["name"] == "Lionel Messi"

    def test_summary_is_string(self, analyzer):
        report = analyzer.player_report("Messi", season_id=90, min_minutes=100)
        assert isinstance(report["summary"], str)


# ─────────────────────────────────────────────────────────────────────────────
# Similarity Search tests
# ─────────────────────────────────────────────────────────────────────────────
class TestFindSimilarPlayers:
    def test_returns_dataframe(self, analyzer):
        df = analyzer.find_similar_players("Messi", season_id=90, min_minutes=100)
        assert isinstance(df, pd.DataFrame)

    def test_target_not_in_results(self, analyzer):
        df = analyzer.find_similar_players("Messi", season_id=90, min_minutes=100)
        assert "Lionel Messi" not in df["player_name"].values

    def test_top_n_respected(self, analyzer):
        df = analyzer.find_similar_players("Messi", season_id=90, min_minutes=100, top_n=3)
        assert len(df) <= 3

    def test_similarity_score_between_0_and_1(self, analyzer):
        df = analyzer.find_similar_players("Messi", season_id=90, min_minutes=100)
        if not df.empty:
            assert (df["similarity_score"] >= 0).all()
            assert (df["similarity_score"] <= 1).all()


# ─────────────────────────────────────────────────────────────────────────────
# Cache behaviour tests
# ─────────────────────────────────────────────────────────────────────────────
class TestCacheBehaviour:
    def test_cache_populated_after_first_call(self, mini_league):
        fresh_analyzer = PerformanceAnalyzer(mini_league)
        assert len(fresh_analyzer._cache) == 0
        fresh_analyzer.player_percentile_profile("Messi", season_id=90, min_minutes=100)
        assert len(fresh_analyzer._cache) > 0

    def test_clear_cache_empties_cache(self, mini_league):
        fresh_analyzer = PerformanceAnalyzer(mini_league)
        fresh_analyzer.player_percentile_profile("Messi", season_id=90, min_minutes=100)
        fresh_analyzer.clear_cache()
        assert len(fresh_analyzer._cache) == 0

    def test_repr_contains_league_name(self, analyzer):
        r = repr(analyzer)
        assert "La Liga" in r
