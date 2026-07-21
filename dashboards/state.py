"""
dashboards/state.py
─────────────────────────────────────────────────────────────────────────────
Session state manager for the Streamlit dashboard.

DESIGN PATTERN — Singleton via st.session_state:
─────────────────────────────────────────────────
Streamlit re-runs the entire script on every user interaction (button click,
slider move, selectbox change).  Without caching, that would mean re-loading
hundreds of JSON files and rebuilding the League object on every keystroke.

AppState solves this by storing expensive objects (League, analyzers) in
st.session_state — Streamlit's key-value store that persists across reruns
within the same browser session.

The class acts as a FAÇADE over st.session_state: all pages import AppState
and call clean named methods rather than reading/writing raw keys.

Usage (in any page file):
    from dashboards.state import AppState
    state = AppState()
    if not state.is_loaded:
        state.load(competition_id=11, season_id=90)
    analyzer = state.analyzer
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from src.advanced_analysis import AdvancedAnalyzer
from src.analyzer import PerformanceAnalyzer
from src.config import cfg
from src.loaders import StatsBombLoader
from src.models import League
from src.utils import get_logger

log = get_logger()

# Keys used in st.session_state — defined as constants to avoid typos
_KEY_LEAGUE        = "fa_league"
_KEY_ANALYZER      = "fa_analyzer"
_KEY_ADVANCED      = "fa_advanced"
_KEY_LOADER        = "fa_loader"
_KEY_COMPETITION   = "fa_competition_id"
_KEY_SEASON        = "fa_season_id"
_KEY_LOADING       = "fa_loading"
_KEY_LOAD_ERROR    = "fa_load_error"


class AppState:
    """
    Façade over st.session_state for the Football Analytics dashboard.

    Provides typed access to the League, PerformanceAnalyzer, and
    AdvancedAnalyzer objects, managing their lifecycle (build once,
    cache in session, invalidate on competition/season change).
    """

    # ── Read-only properties ──────────────────────────────────────────────────
    @property
    def league(self) -> Optional[League]:
        """The loaded League object, or None if not yet loaded."""
        return st.session_state.get(_KEY_LEAGUE)

    @property
    def analyzer(self) -> Optional[PerformanceAnalyzer]:
        """The PerformanceAnalyzer, or None if not loaded."""
        return st.session_state.get(_KEY_ANALYZER)

    @property
    def advanced(self) -> Optional[AdvancedAnalyzer]:
        """The AdvancedAnalyzer (Phase 4), or None if not loaded."""
        return st.session_state.get(_KEY_ADVANCED)

    @property
    def loader(self) -> Optional[StatsBombLoader]:
        """The StatsBombLoader, built once per session."""
        return st.session_state.get(_KEY_LOADER)

    @property
    def is_loaded(self) -> bool:
        """True if a League has been successfully loaded."""
        return self.league is not None

    @property
    def is_loading(self) -> bool:
        """True while data is being fetched/processed."""
        return st.session_state.get(_KEY_LOADING, False)

    @property
    def load_error(self) -> Optional[str]:
        """Last load error message, or None."""
        return st.session_state.get(_KEY_LOAD_ERROR)

    @property
    def competition_id(self) -> Optional[int]:
        return st.session_state.get(_KEY_COMPETITION)

    @property
    def season_id(self) -> Optional[int]:
        return st.session_state.get(_KEY_SEASON)

    # ── Core load method ──────────────────────────────────────────────────────
    def load(
        self,
        competition_id: int,
        season_id: int,
        max_matches: Optional[int] = None,
    ) -> bool:
        """
        Load StatsBomb data and build all analytical objects.

        This is the expensive operation — called once when the user selects
        a competition/season from the sidebar.  All results are stored in
        st.session_state for the rest of the session.

        Args:
            competition_id: StatsBomb competition ID (2=EPL, 11=La Liga).
            season_id:      StatsBomb season ID.
            max_matches:    Cap for development/demo mode (None = full season).

        Returns:
            True if load succeeded, False on error.
        """
        # Skip if already loaded with the same competition + season
        if (
            self.is_loaded
            and self.competition_id == competition_id
            and self.season_id == season_id
        ):
            return True

        st.session_state[_KEY_LOADING]   = True
        st.session_state[_KEY_LOAD_ERROR] = None

        try:
            # ── 1. Build loader (reuse if already exists in session) ──────────
            if self.loader is None:
                st.session_state[_KEY_LOADER] = StatsBombLoader(use_cache=True)
            loader = st.session_state[_KEY_LOADER]

            # ── 2. Fetch competition metadata ─────────────────────────────────
            comps_df = loader.competitions()
            comp_row = comps_df[
                (comps_df["competition_id"] == competition_id)
                & (comps_df["season_id"] == season_id)
            ]
            if comp_row.empty:
                raise ValueError(
                    f"No data found for competition_id={competition_id}, "
                    f"season_id={season_id}. Check StatsBomb open data."
                )
            competition_name = str(comp_row["competition_name"].iloc[0])
            season_name      = str(comp_row["season_name"].iloc[0])
            country          = str(comp_row.get("country_name", "Unknown").iloc[0]
                                   if "country_name" in comp_row.columns else "Unknown")

            # ── 3. Load player stats ───────────────────────────────────────────
            stats_df = loader.player_season_stats(
                competition_id=competition_id,
                season_id=season_id,
                max_matches=max_matches,
            )
            if stats_df.empty:
                raise ValueError("No player stats returned. The season may have no open data.")

            # ── 4. Build domain objects ────────────────────────────────────────
            league = League.build_from_stats_df(
                stats_df=stats_df,
                competition_id=competition_id,
                competition_name=competition_name,
                season_id=season_id,
                season_name=season_name,
                country=country,
            )

            # ── 5. Build analyzers ─────────────────────────────────────────────
            analyzer = PerformanceAnalyzer(league)
            advanced = AdvancedAnalyzer(league, n_clusters=4)

            # ── 6. Store in session state ──────────────────────────────────────
            st.session_state[_KEY_LEAGUE]       = league
            st.session_state[_KEY_ANALYZER]     = analyzer
            st.session_state[_KEY_ADVANCED]     = advanced
            st.session_state[_KEY_COMPETITION]  = competition_id
            st.session_state[_KEY_SEASON]       = season_id

            log.success(
                f"Dashboard loaded: {competition_name} {season_name} | "
                f"{len(league.all_players())} players"
            )
            return True

        except Exception as exc:
            st.session_state[_KEY_LOAD_ERROR] = str(exc)
            log.error(f"Dashboard load failed: {exc}")
            return False

        finally:
            st.session_state[_KEY_LOADING] = False

    def clear(self) -> None:
        """Reset all session state (force a full reload on next interaction)."""
        for key in [_KEY_LEAGUE, _KEY_ANALYZER, _KEY_ADVANCED,
                    _KEY_COMPETITION, _KEY_SEASON, _KEY_LOAD_ERROR]:
            st.session_state.pop(key, None)
        log.info("AppState cleared.")

    def player_names(self) -> list:
        """Sorted list of all player names in the loaded league."""
        if not self.is_loaded:
            return []
        return sorted(p.name for p in self.league.all_players())

    def team_names(self) -> list:
        """Sorted list of all team names in the loaded season."""
        if not self.is_loaded or not self.season_id:
            return []
        return sorted(t.name for t in self.league.get_teams(self.season_id))
