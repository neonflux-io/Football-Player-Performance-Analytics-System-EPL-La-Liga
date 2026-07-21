"""
tests/test_phase6.py
─────────────────────────────────────────────────────────────────────────────
Phase 6 tests — Dashboard state management and component logic.

We cannot run Streamlit's rendering in pytest (it needs a browser session),
but we CAN test:
  • AppState.load() integration with real StatsBomb loader (mocked HTTP)
  • AppState property accessors
  • Component helper functions (pure Python logic)
  • The data pipeline: loader → League → analyzers in one call

Run with:  pytest tests/test_phase6.py -v
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Mock st.session_state BEFORE importing anything that uses streamlit ───────
# We replace st.session_state with a plain dict so AppState works in pytest.
import types
mock_st = types.ModuleType("streamlit")
_session: dict = {}

mock_st.session_state = _session
mock_st.cache_resource = lambda f: f          # no-op decorator
mock_st.set_page_config = lambda **kw: None

# Stub all Streamlit UI functions used in app.py / components.py
for fn_name in [
    "markdown", "write", "info", "warning", "error", "success",
    "dataframe", "metric", "divider", "stop", "image", "bar_chart",
    "spinner", "button", "selectbox", "slider", "checkbox", "radio",
    "columns", "tabs", "sidebar", "expander", "header", "subheader",
    "text_input", "number_input", "multiselect",
]:
    setattr(mock_st, fn_name, MagicMock())

# Make mock_st.columns return list of MagicMocks for col1, col2, col3 pattern
mock_st.columns = lambda n: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]
mock_st.tabs = lambda labels: [MagicMock() for _ in labels]

# Make sidebar usable
mock_st.sidebar = MagicMock()
mock_st.sidebar.header = MagicMock()
mock_st.sidebar.selectbox = MagicMock(return_value="La Liga (richest open data)")
mock_st.sidebar.checkbox = MagicMock(return_value=True)
mock_st.sidebar.button = MagicMock(return_value=False)
mock_st.sidebar.radio = MagicMock(return_value="🏠 Overview")
mock_st.sidebar.divider = MagicMock()
mock_st.sidebar.markdown = MagicMock()
mock_st.sidebar.spinner = MagicMock()
mock_st.sidebar.success = MagicMock()
mock_st.sidebar.error = MagicMock()

sys.modules["streamlit"] = mock_st

# Now safe to import dashboard modules
from dashboards.state import AppState, _KEY_LEAGUE, _KEY_ANALYZER, _KEY_ADVANCED
from dashboards.components import _percentile_colour, _progress_bar_html
from src.models import League, Outfielder, Goalkeeper, SeasonRecord, Team


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
def _make_mini_league() -> League:
    """Build the same mini-league used in Phase 3 tests."""
    league = League(11, "La Liga", "Spain")
    barca  = Team(1, "FC Barcelona", "La Liga", "2021/22")

    messi = Outfielder(5503, "Lionel Messi", "Right Wing", "Argentina", 34)
    messi.add_season(SeasonRecord(
        competition_id=11, season_id=90, season_name="2021/22",
        team_name="FC Barcelona", minutes_played=3000.0,
        goals=20, assists=12, xg=17.5, xa=8.0,
        passes=800, passes_completed=700, key_passes=40, shots=60,
        carries=200, dribbles_attempted=50, dribbles_completed=35,
        pressures=180,
    ))

    ronaldo = Outfielder(20055, "Cristiano Ronaldo", "Center Forward", "Portugal", 37)
    ronaldo.add_season(SeasonRecord(
        competition_id=11, season_id=90, season_name="2021/22",
        team_name="FC Barcelona", minutes_played=2900.0,
        goals=18, assists=6, xg=16.0, shots=90,
        passes=500, passes_completed=420, key_passes=20,
        carries=150, dribbles_attempted=30, dribbles_completed=20,
        pressures=120,
    ))

    gk = Goalkeeper(9999, "Marc ter Stegen", "Goalkeeper", "Germany", 29)
    gk.add_season(SeasonRecord(
        competition_id=11, season_id=90, season_name="2021/22",
        team_name="FC Barcelona", minutes_played=2700.0,
        saves=95, goals_conceded=28, clean_sheets=12,
        passes=400, passes_completed=360,
    ))

    for p in [messi, ronaldo, gk]:
        barca.add_player(p)
    league.add_team(barca, season_id=90)
    return league


@pytest.fixture(autouse=True)
def clear_session():
    """Reset mock session_state before each test."""
    _session.clear()
    yield
    _session.clear()


@pytest.fixture
def loaded_state(clear_session) -> AppState:
    """AppState with a pre-loaded mini-league injected directly."""
    from src.analyzer import PerformanceAnalyzer
    from src.advanced_analysis import AdvancedAnalyzer

    league   = _make_mini_league()
    analyzer = PerformanceAnalyzer(league)
    advanced = AdvancedAnalyzer(league, n_clusters=2)

    _session[_KEY_LEAGUE]    = league
    _session[_KEY_ANALYZER]  = analyzer
    _session[_KEY_ADVANCED]  = advanced
    _session["fa_competition_id"] = 11
    _session["fa_season_id"]      = 90

    return AppState()


# ─────────────────────────────────────────────────────────────────────────────
# AppState tests
# ─────────────────────────────────────────────────────────────────────────────
class TestAppStateProperties:
    def test_is_loaded_false_when_empty(self, clear_session):
        state = AppState()
        assert state.is_loaded is False

    def test_is_loaded_true_when_league_in_session(self, loaded_state):
        assert loaded_state.is_loaded is True

    def test_league_property_returns_league(self, loaded_state):
        assert isinstance(loaded_state.league, League)

    def test_analyzer_property_returns_analyzer(self, loaded_state):
        from src.analyzer import PerformanceAnalyzer
        assert isinstance(loaded_state.analyzer, PerformanceAnalyzer)

    def test_advanced_property_returns_advanced(self, loaded_state):
        from src.advanced_analysis import AdvancedAnalyzer
        assert isinstance(loaded_state.advanced, AdvancedAnalyzer)

    def test_player_names_returns_sorted_list(self, loaded_state):
        names = loaded_state.player_names()
        assert isinstance(names, list)
        assert len(names) == 3
        assert names == sorted(names)

    def test_team_names_returns_list(self, loaded_state):
        teams = loaded_state.team_names()
        assert "FC Barcelona" in teams

    def test_competition_id_stored(self, loaded_state):
        assert loaded_state.competition_id == 11

    def test_season_id_stored(self, loaded_state):
        assert loaded_state.season_id == 90

    def test_clear_removes_league(self, loaded_state):
        loaded_state.clear()
        assert loaded_state.is_loaded is False

    def test_clear_removes_analyzer(self, loaded_state):
        loaded_state.clear()
        assert loaded_state.analyzer is None

    def test_is_loading_defaults_false(self, clear_session):
        state = AppState()
        assert state.is_loading is False

    def test_load_error_defaults_none(self, clear_session):
        state = AppState()
        assert state.load_error is None


# ─────────────────────────────────────────────────────────────────────────────
# AppState.load() — mocked HTTP, real pipeline
# ─────────────────────────────────────────────────────────────────────────────
class TestAppStateLoad:
    def _make_stats_df(self) -> pd.DataFrame:
        """Minimal player stats DataFrame matching what StatsBombLoader returns."""
        return pd.DataFrame([
            {"player_id": 5503, "player_name": "Lionel Messi",
             "team_name": "FC Barcelona", "position": "Right Wing",
             "goals": 20, "assists": 12, "xg": 17.5, "xa": 8.0,
             "passes": 800, "passes_completed": 700, "key_passes": 40,
             "shots": 60, "carries": 200, "dribbles_attempted": 50,
             "dribbles_completed": 35, "pressures": 180, "minutes_played": 3000.0},
            {"player_id": 20055, "player_name": "Cristiano Ronaldo",
             "team_name": "Real Madrid", "position": "Center Forward",
             "goals": 18, "assists": 6, "xg": 16.0, "xa": 3.0,
             "passes": 500, "passes_completed": 420, "key_passes": 20,
             "shots": 90, "carries": 150, "dribbles_attempted": 30,
             "dribbles_completed": 20, "pressures": 120, "minutes_played": 2900.0},
        ])

    def _make_comps_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "competition_id": 11,
            "competition_name": "La Liga",
            "season_id": 90,
            "season_name": "2021/2022",
            "country_name": "Spain",
        }])

    def test_load_success_sets_is_loaded(self, clear_session):
        state = AppState()
        stats_df = self._make_stats_df()
        comps_df = self._make_comps_df()

        with patch("dashboards.state.StatsBombLoader") as MockLoader:
            instance = MockLoader.return_value
            instance.competitions.return_value  = comps_df
            instance.player_season_stats.return_value = stats_df

            result = state.load(competition_id=11, season_id=90, max_matches=5)

        assert result is True
        assert state.is_loaded is True

    def test_load_success_builds_league(self, clear_session):
        state = AppState()
        stats_df = self._make_stats_df()
        comps_df = self._make_comps_df()

        with patch("dashboards.state.StatsBombLoader") as MockLoader:
            instance = MockLoader.return_value
            instance.competitions.return_value  = comps_df
            instance.player_season_stats.return_value = stats_df
            state.load(11, 90, max_matches=5)

        assert isinstance(state.league, League)
        assert len(state.league.all_players()) == 2

    def test_load_success_builds_analyzer(self, clear_session):
        from src.analyzer import PerformanceAnalyzer
        state = AppState()
        stats_df = self._make_stats_df()
        comps_df = self._make_comps_df()

        with patch("dashboards.state.StatsBombLoader") as MockLoader:
            instance = MockLoader.return_value
            instance.competitions.return_value  = comps_df
            instance.player_season_stats.return_value = stats_df
            state.load(11, 90, max_matches=5)

        assert isinstance(state.analyzer, PerformanceAnalyzer)

    def test_load_skip_if_already_loaded_same_season(self, loaded_state):
        """load() should be a no-op if same competition + season already loaded."""
        with patch("dashboards.state.StatsBombLoader") as MockLoader:
            result = loaded_state.load(competition_id=11, season_id=90)
        MockLoader.assert_not_called()
        assert result is True

    def test_load_failure_sets_error(self, clear_session):
        state = AppState()
        with patch("dashboards.state.StatsBombLoader") as MockLoader:
            instance = MockLoader.return_value
            instance.competitions.side_effect = ConnectionError("GitHub unreachable")
            result = state.load(11, 90)

        assert result is False
        assert state.load_error is not None
        assert "GitHub" in state.load_error

    def test_load_empty_stats_df_sets_error(self, clear_session):
        state = AppState()
        comps_df = self._make_comps_df()
        with patch("dashboards.state.StatsBombLoader") as MockLoader:
            instance = MockLoader.return_value
            instance.competitions.return_value  = comps_df
            instance.player_season_stats.return_value = pd.DataFrame()  # empty
            result = state.load(11, 90)

        assert result is False
        assert state.load_error is not None

    def test_loading_flag_is_false_after_load(self, clear_session):
        state = AppState()
        comps_df = self._make_comps_df()
        with patch("dashboards.state.StatsBombLoader") as MockLoader:
            instance = MockLoader.return_value
            instance.competitions.return_value  = comps_df
            instance.player_season_stats.return_value = self._make_stats_df()
            state.load(11, 90)
        assert state.is_loading is False

    def test_loading_flag_false_after_failure(self, clear_session):
        state = AppState()
        with patch("dashboards.state.StatsBombLoader") as MockLoader:
            instance = MockLoader.return_value
            instance.competitions.side_effect = RuntimeError("boom")
            state.load(11, 90)
        assert state.is_loading is False


# ─────────────────────────────────────────────────────────────────────────────
# Component helper function tests (pure Python — no Streamlit needed)
# ─────────────────────────────────────────────────────────────────────────────
class TestComponentHelpers:
    def test_percentile_colour_high(self):
        assert _percentile_colour(90) == "#39d353"

    def test_percentile_colour_mid(self):
        assert _percentile_colour(65) == "#f0a500"

    def test_percentile_colour_low(self):
        assert _percentile_colour(30) == "#f78166"

    def test_percentile_colour_boundary_80(self):
        assert _percentile_colour(80) == "#39d353"

    def test_percentile_colour_boundary_50(self):
        assert _percentile_colour(50) == "#f0a500"

    def test_progress_bar_html_contains_width(self):
        html = _progress_bar_html(75, "#39d353")
        assert "75%" in html
        assert "#39d353" in html

    def test_progress_bar_html_min_width_for_zero(self):
        html = _progress_bar_html(0, "#f78166")
        # Even 0 percentile should render as 2% minimum width
        assert "2%" in html

    def test_progress_bar_html_returns_string(self):
        result = _progress_bar_html(50, "#aaa")
        assert isinstance(result, str)
        assert "<div" in result


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end pipeline test (loader → league → analyzer, real objects)
# ─────────────────────────────────────────────────────────────────────────────
class TestEndToEndPipeline:
    """
    Validates the complete data pipeline that the dashboard's load() method runs.
    Uses the mini-league fixture to avoid real network calls.
    """

    def test_analyzer_can_rank_players(self, loaded_state):
        df = loaded_state.analyzer.league_rankings(
            "Goals/90", season_id=90, min_minutes=100, top_n=3
        )
        assert not df.empty
        assert df.iloc[0]["player_name"] in ["Lionel Messi", "Cristiano Ronaldo"]

    def test_analyzer_can_profile_player(self, loaded_state):
        profile = loaded_state.analyzer.player_percentile_profile(
            "Messi", season_id=90, min_minutes=100
        )
        assert profile.player_name == "Lionel Messi"
        assert len(profile.percentiles) > 0

    def test_analyzer_can_compare_players(self, loaded_state):
        result = loaded_state.analyzer.compare_players(
            "Messi", "Ronaldo", season_id=90, min_minutes=100
        )
        assert "Messi" in result.player_a_name
        assert "Ronaldo" in result.player_b_name

    def test_analyzer_can_analyse_team(self, loaded_state):
        report = loaded_state.analyzer.analyse_team("FC Barcelona", season_id=90)
        assert report.total_goals > 0
        assert "Messi" in report.top_scorer_name

    def test_factory_renders_radar_from_state(self, loaded_state):
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src.visualizations import VisualisationFactory

        profile = loaded_state.analyzer.player_percentile_profile(
            "Messi", season_id=90, min_minutes=100
        )
        factory = VisualisationFactory()
        fig = factory.radar(profile)
        assert fig is not None
        plt.close("all")

    def test_player_names_sorted(self, loaded_state):
        names = loaded_state.player_names()
        assert names == sorted(names)

    def test_team_names_includes_barcelona(self, loaded_state):
        teams = loaded_state.team_names()
        assert any("Barcelona" in t for t in teams)
