"""
analyzer.py
─────────────────────────────────────────────────────────────────────────────
PerformanceAnalyzer — the primary analytics engine for the Football Analytics
System.

Responsibilities
────────────────
  1. Career trajectory analysis   – how a player's per-90 metrics evolve
                                    across seasons
  2. Player comparison            – head-to-head stats + percentile rankings
  3. Percentile engine            – rank any player vs. the full league
  4. Positional benchmarking      – compare a player to position peers
  5. Team-level analysis          – squad depth, goal contribution share
  6. Report generation            – structured summary dicts for the dashboard

OOP concepts demonstrated
──────────────────────────
  • Single Responsibility  – Analyzer only analyses; it does NOT load data or
                             render charts (those are in loaders.py / visualizations.py)
  • Dependency Injection   – League object is passed into __init__, not built
                             here; makes the class easily testable with mocks
  • Encapsulation          – computed percentile tables are lazily built and
                             stored in a private _cache dict
  • Open / Closed          – adding a new metric group = one dict entry,
                             zero changes to method logic
  • Abstraction            – compare_players() works on any two Player subclasses
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.config import cfg
from src.models import Goalkeeper, League, Outfielder, Player, SeasonRecord
from src.utils import get_logger, per_90, safe_divide, timer

log = get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses  (lightweight, typed containers for analyzer output)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrajectoryResult:
    """
    Output of career_trajectory().

    Contains the season-by-season DataFrame plus derived trend signals.
    """

    player_name: str
    player_id: int
    # One row per season; columns include per-90 metrics and raw totals
    df: pd.DataFrame
    # Peak season metadata
    peak_season_name: str = ""
    peak_metric: str = ""
    peak_value: float = 0.0
    # Simple linear trend slope for the primary metric (positive = improving)
    trend_slope: float = 0.0
    trend_direction: str = "stable"  # "improving" | "declining" | "stable"


@dataclass
class ComparisonResult:
    """
    Output of compare_players().

    Contains side-by-side metrics + percentile ranks for both players.
    """

    player_a_name: str
    player_b_name: str
    # Raw per-90 metrics for each player: {metric_label: value}
    metrics_a: Dict[str, float] = field(default_factory=dict)
    metrics_b: Dict[str, float] = field(default_factory=dict)
    # Percentile rank within position peers: {metric_label: percentile 0-100}
    percentiles_a: Dict[str, float] = field(default_factory=dict)
    percentiles_b: Dict[str, float] = field(default_factory=dict)
    # Flat comparison DataFrame (for tables in the dashboard)
    comparison_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Who "wins" each metric
    edge: Dict[str, str] = field(default_factory=dict)   # {metric: "A" | "B" | "draw"}


@dataclass
class PercentileProfile:
    """
    Output of player_percentile_profile().

    Every key metric expressed as a 0–100 percentile vs. position peers.
    """

    player_name: str
    player_id: int
    position_group: str
    season_id: Optional[int]
    min_minutes: int
    # {metric_label: percentile}
    percentiles: Dict[str, float] = field(default_factory=dict)
    # {metric_label: raw per-90 value}
    raw_values: Dict[str, float] = field(default_factory=dict)
    n_peers: int = 0


@dataclass
class TeamReport:
    """Output of analyse_team()."""

    team_name: str
    season_name: str
    squad_size: int
    total_goals: int
    total_assists: int
    top_scorer_name: str
    top_scorer_goals: int
    avg_pass_completion: float
    goal_contribution_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    squad_stats_df: pd.DataFrame = field(default_factory=pd.DataFrame)


# ─────────────────────────────────────────────────────────────────────────────
# PerformanceAnalyzer
# ─────────────────────────────────────────────────────────────────────────────
class PerformanceAnalyzer:
    """
    Main analytics engine — consumes League/Player domain objects and
    produces structured analytical results.

    Usage
    ─────
        league = League.build_from_stats_df(...)
        analyzer = PerformanceAnalyzer(league)

        trajectory = analyzer.career_trajectory("Lionel Messi")
        comparison = analyzer.compare_players("Messi", "Ronaldo")
        profile    = analyzer.player_percentile_profile("Messi")

    The analyzer never loads data or renders charts.  It only computes.
    """

    # ── Metric group definitions ───────────────────────────────────────────────
    # OPEN/CLOSED PRINCIPLE: add a new group here without touching any method.
    # Keys become column headers; values are (stat_attr, label) tuples.
    OUTFIELD_METRICS: ClassVar[Dict[str, str]] = {
        # internal attr name → human-readable label
        "goals":              "Goals/90",
        "assists":            "Assists/90",
        "xg":                 "xG/90",
        "xa":                 "xA/90",
        "key_passes":         "Key Passes/90",
        "dribbles_completed": "Dribbles/90",
        "pressures":          "Pressures/90",
        "carries":            "Carries/90",
        "shots":              "Shots/90",
        "passes":             "Passes/90",
    }

    GK_METRICS: ClassVar[Dict[str, str]] = {
        "saves":          "Saves/90",
        "goals_conceded": "Goals Conceded/90",
        "passes":         "Passes/90",
    }

    # Minimum minutes threshold for percentile rankings
    DEFAULT_MIN_MINUTES: int = cfg.MIN_MINUTES_FOR_ANALYSIS

    def __init__(self, league: League) -> None:
        """
        Initialise the analyzer with a populated League.

        DEPENDENCY INJECTION: We accept an already-built League rather than
        building one internally.  This makes the class trivially testable —
        pass in a League built from a tiny fixture DataFrame in tests.

        Args:
            league: A fully-populated League instance (from Phase 2).
        """
        self._league: League = league

        # ENCAPSULATION: lazy cache for expensive computed tables
        # Keys: "outfield_p90_{season_id}" or "gk_p90_{season_id}"
        self._cache: Dict[str, pd.DataFrame] = {}

        log.info(
            f"PerformanceAnalyzer initialised | league='{league.name}' | "
            f"seasons={league.season_ids}"
        )

    # ── Private: percentile table builder ────────────────────────────────────
    def _build_percentile_table(
        self,
        season_id: Optional[int],
        position_group: str,
        min_minutes: int,
    ) -> pd.DataFrame:
        """
        Build (or retrieve from cache) a DataFrame of per-90 metrics for all
        qualifying players in a given position group.

        This table is the denominator for all percentile calculations.

        Args:
            season_id:      Filter to one season, or None for career totals.
            position_group: "outfield" or "goalkeeper".
            min_minutes:    Exclude players below this threshold.

        Returns:
            pd.DataFrame with one row per qualifying player.
            Columns: player_id, player_name + one column per metric.
        """
        cache_key = f"{position_group}_{season_id}_{min_minutes}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        metric_map = (
            self.OUTFIELD_METRICS if position_group == "outfield" else self.GK_METRICS
        )

        players = self._league.all_players(season_id=season_id)

        rows: List[Dict[str, Any]] = []
        for player in players:
            # Filter by position type
            is_gk = isinstance(player, Goalkeeper)
            if position_group == "goalkeeper" and not is_gk:
                continue
            if position_group == "outfield" and is_gk:
                continue

            # Filter by minutes
            if season_id is not None:
                rec = player.get_season(season_id)
                if rec is None or rec.minutes_played < min_minutes:
                    continue
                total_minutes = rec.minutes_played
            else:
                totals = player.career_totals()
                total_minutes = totals.get("minutes_played", 0)
                if total_minutes < min_minutes:
                    continue

            row: Dict[str, Any] = {
                "player_id": player.player_id,
                "player_name": player.name,
                "position": player.position,
                "minutes_played": total_minutes,
            }
            for attr, label in metric_map.items():
                row[label] = player.per_90(attr, season_id=season_id)

            rows.append(row)

        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        self._cache[cache_key] = df
        log.debug(
            f"Percentile table built | group={position_group} | "
            f"season={season_id} | n={len(df)}"
        )
        return df

    def _get_percentile(
        self,
        value: float,
        series: pd.Series,
    ) -> float:
        """
        Calculate the percentile rank of *value* within *series*.

        Uses scipy.stats.percentileofscore with 'rank' method (ties → average).

        Args:
            value:  The player's metric value.
            series: All values for that metric across peer players.

        Returns:
            Percentile 0.0–100.0 (e.g., 87.5 = top 12.5%).
        """
        if series.empty or series.isna().all():
            return 0.0
        return round(
            float(stats.percentileofscore(series.dropna(), value, kind="rank")), 1
        )

    # ── 1. Career Trajectory ─────────────────────────────────────────────────
    @timer
    def career_trajectory(
        self,
        player_name: str,
        primary_metric: str = "Goals/90",
        season_id: Optional[int] = None,
    ) -> TrajectoryResult:
        """
        Analyse how a player's key metrics have evolved season by season.

        Produces a season-indexed DataFrame plus:
          • Peak season identification (highest value of primary_metric)
          • Linear trend slope (scipy linregress over season index)
          • Trend direction label: "improving" | "declining" | "stable"

        Args:
            player_name:    Partial or full player name (case-insensitive).
            primary_metric: The metric used to determine peak/trend.
                            Must be a label from OUTFIELD_METRICS values.
            season_id:      Unused here (trajectory always uses all seasons),
                            kept for API consistency.

        Returns:
            TrajectoryResult with .df and trend metadata.

        Raises:
            ValueError: If player not found or has fewer than 1 season.
        """
        player = self._league.find_player(player_name)
        if player is None:
            raise ValueError(
                f"Player '{player_name}' not found in league '{self._league.name}'. "
                f"Try a partial name match."
            )

        if not player.seasons:
            raise ValueError(f"Player '{player.name}' has no season data loaded.")

        df = player.career_stats_df()

        if df.empty:
            raise ValueError(f"No stats available for '{player.name}'.")

        # ── Determine peak season ─────────────────────────────────────────────
        # Map human label → column name in career_stats_df()
        label_to_col = {
            "Goals/90": "goals_p90",
            "Assists/90": "assists_p90",
            "xG/90": "xg_p90",
            "xA/90": "xa_p90",
            "Key Passes/90": "key_passes_p90",
            "Pressures/90": "pressures_p90",
        }
        col = label_to_col.get(primary_metric, "goals_p90")

        peak_idx = df[col].idxmax() if col in df.columns else 0
        peak_row = df.loc[peak_idx]
        peak_value = float(peak_row.get(col, 0.0))
        peak_season = str(peak_row.get("season_name", "Unknown"))

        # ── Linear trend ──────────────────────────────────────────────────────
        trend_slope = 0.0
        trend_direction = "stable"
        if col in df.columns and len(df) >= 2:
            x = np.arange(len(df), dtype=float)
            y = df[col].fillna(0).values.astype(float)
            slope, _, r_value, p_value, _ = stats.linregress(x, y)
            trend_slope = round(float(slope), 4)

            # Only label as improving/declining if trend is meaningful
            # (p < 0.1 and |slope| > 0.01 per season)
            if p_value < 0.1 and abs(trend_slope) > 0.01:
                trend_direction = "improving" if trend_slope > 0 else "declining"

        result = TrajectoryResult(
            player_name=player.name,
            player_id=player.player_id,
            df=df,
            peak_season_name=peak_season,
            peak_metric=primary_metric,
            peak_value=round(peak_value, 3),
            trend_slope=trend_slope,
            trend_direction=trend_direction,
        )
        log.info(
            f"Trajectory: {player.name} | peak={peak_season} ({primary_metric}={peak_value:.3f}) "
            f"| trend={trend_direction} (slope={trend_slope:+.4f})"
        )
        return result

    # ── 2. Percentile Profile ────────────────────────────────────────────────
    @timer
    def player_percentile_profile(
        self,
        player_name: str,
        season_id: Optional[int] = None,
        min_minutes: int = DEFAULT_MIN_MINUTES,
    ) -> PercentileProfile:
        """
        Express every key metric as a percentile rank vs. position peers.

        Example: a player in the 91st percentile for Goals/90 scores more
        per 90 minutes than 91% of qualifying players in their position group.

        Args:
            player_name: Partial or full player name.
            season_id:   Restrict comparison to one season, or None for career.
            min_minutes: Minimum minutes for a peer to be included in rankings.

        Returns:
            PercentileProfile with .percentiles and .raw_values dicts.

        Raises:
            ValueError: If player not found.
        """
        player = self._league.find_player(player_name)
        if player is None:
            raise ValueError(f"Player '{player_name}' not found.")

        position_group = "goalkeeper" if isinstance(player, Goalkeeper) else "outfield"
        metric_map = (
            self.OUTFIELD_METRICS if position_group == "outfield" else self.GK_METRICS
        )

        # Build the peer table (cached after first call)
        peer_df = self._build_percentile_table(season_id, position_group, min_minutes)

        percentiles: Dict[str, float] = {}
        raw_values: Dict[str, float] = {}

        for attr, label in metric_map.items():
            raw_val = player.per_90(attr, season_id=season_id)
            raw_values[label] = round(raw_val, 3)

            if not peer_df.empty and label in peer_df.columns:
                pct = self._get_percentile(raw_val, peer_df[label])
            else:
                pct = 0.0
            percentiles[label] = pct

        profile = PercentileProfile(
            player_name=player.name,
            player_id=player.player_id,
            position_group=position_group,
            season_id=season_id,
            min_minutes=min_minutes,
            percentiles=percentiles,
            raw_values=raw_values,
            n_peers=len(peer_df),
        )
        log.info(
            f"Percentile profile: {player.name} | group={position_group} | "
            f"peers={len(peer_df)} | season={season_id}"
        )
        return profile

    # ── 3. Player Comparison ─────────────────────────────────────────────────
    @timer
    def compare_players(
        self,
        player_a_name: str,
        player_b_name: str,
        season_id: Optional[int] = None,
        min_minutes: int = DEFAULT_MIN_MINUTES,
    ) -> ComparisonResult:
        """
        Head-to-head comparison of two players with per-90 metrics and
        percentile ranks.

        Both players are benchmarked against the same position peer group
        (determined by player_a's position group).

        Args:
            player_a_name: First player (partial name match).
            player_b_name: Second player (partial name match).
            season_id:     Restrict to one season, or None for career.
            min_minutes:   Min minutes for peer ranking denominator.

        Returns:
            ComparisonResult with side-by-side metrics, percentiles, edge.

        Raises:
            ValueError: If either player is not found.
        """
        player_a = self._league.find_player(player_a_name)
        player_b = self._league.find_player(player_b_name)

        if player_a is None:
            raise ValueError(f"Player '{player_a_name}' not found.")
        if player_b is None:
            raise ValueError(f"Player '{player_b_name}' not found.")

        # Use player_a's position group as the benchmark group
        pos_group = "goalkeeper" if isinstance(player_a, Goalkeeper) else "outfield"
        metric_map = (
            self.OUTFIELD_METRICS if pos_group == "outfield" else self.GK_METRICS
        )

        peer_df = self._build_percentile_table(season_id, pos_group, min_minutes)

        metrics_a: Dict[str, float] = {}
        metrics_b: Dict[str, float] = {}
        percentiles_a: Dict[str, float] = {}
        percentiles_b: Dict[str, float] = {}
        edge: Dict[str, str] = {}

        for attr, label in metric_map.items():
            val_a = round(player_a.per_90(attr, season_id=season_id), 3)
            val_b = round(player_b.per_90(attr, season_id=season_id), 3)

            metrics_a[label] = val_a
            metrics_b[label] = val_b

            if not peer_df.empty and label in peer_df.columns:
                percentiles_a[label] = self._get_percentile(val_a, peer_df[label])
                percentiles_b[label] = self._get_percentile(val_b, peer_df[label])
            else:
                percentiles_a[label] = 0.0
                percentiles_b[label] = 0.0

            # Determine edge (who wins this metric)
            if abs(val_a - val_b) < 1e-6:
                edge[label] = "draw"
            elif val_a > val_b:
                edge[label] = player_a.name
            else:
                edge[label] = player_b.name

        # Build flat comparison DataFrame
        rows = []
        for label in metric_map.values():
            rows.append(
                {
                    "Metric": label,
                    f"{player_a.name} (raw)": metrics_a.get(label, 0.0),
                    f"{player_b.name} (raw)": metrics_b.get(label, 0.0),
                    f"{player_a.name} (pct)": percentiles_a.get(label, 0.0),
                    f"{player_b.name} (pct)": percentiles_b.get(label, 0.0),
                    "Edge": edge.get(label, "draw"),
                }
            )

        comparison_df = pd.DataFrame(rows)

        result = ComparisonResult(
            player_a_name=player_a.name,
            player_b_name=player_b.name,
            metrics_a=metrics_a,
            metrics_b=metrics_b,
            percentiles_a=percentiles_a,
            percentiles_b=percentiles_b,
            comparison_df=comparison_df,
            edge=edge,
        )
        log.info(
            f"Comparison: {player_a.name} vs {player_b.name} | "
            f"season={season_id} | peers={len(peer_df)}"
        )
        return result

    # ── 4. League Rankings ───────────────────────────────────────────────────
    @timer
    def league_rankings(
        self,
        metric: str = "Goals/90",
        season_id: Optional[int] = None,
        position_group: str = "outfield",
        min_minutes: int = DEFAULT_MIN_MINUTES,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Return the top-N players in the league for a given metric.

        Args:
            metric:         Column label (e.g. "Goals/90", "xG/90").
            season_id:      Filter to one season, or None for career.
            position_group: "outfield" or "goalkeeper".
            min_minutes:    Minimum minutes threshold.
            top_n:          Number of rows to return.

        Returns:
            pd.DataFrame sorted descending by *metric*, with rank column.
        """
        df = self._build_percentile_table(season_id, position_group, min_minutes)

        if df.empty:
            log.warning("League rankings: no qualifying players found.")
            return pd.DataFrame()

        if metric not in df.columns:
            available = [c for c in df.columns if c not in ("player_id", "player_name", "position")]
            raise ValueError(
                f"Metric '{metric}' not found. Available: {available}"
            )

        ranked = (
            df[["player_name", "position", "minutes_played", metric]]
            .sort_values(metric, ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        ranked.index += 1  # 1-based rank
        ranked.index.name = "Rank"

        log.info(f"League rankings | metric={metric} | top {top_n}")
        return ranked

    # ── 5. Positional Benchmarking ───────────────────────────────────────────
    @timer
    def positional_benchmark(
        self,
        player_name: str,
        position_filter: Optional[str] = None,
        season_id: Optional[int] = None,
        min_minutes: int = DEFAULT_MIN_MINUTES,
    ) -> pd.DataFrame:
        """
        Compare a player against all players in their position group.

        Optionally narrow the comparison to a specific position label
        (e.g. "Center Forward" only, rather than all outfielders).

        Args:
            player_name:     Target player.
            position_filter: Optional exact position string to filter peers.
                             If None, uses the full outfield/GK group.
            season_id:       Season filter.
            min_minutes:     Minimum minutes for peers.

        Returns:
            pd.DataFrame with the target player highlighted (is_target=True)
            and all peers, sorted by Goals/90 desc.
        """
        player = self._league.find_player(player_name)
        if player is None:
            raise ValueError(f"Player '{player_name}' not found.")

        pos_group = "goalkeeper" if isinstance(player, Goalkeeper) else "outfield"
        peer_df = self._build_percentile_table(season_id, pos_group, min_minutes).copy()

        if peer_df.empty:
            return pd.DataFrame()

        # Optionally narrow by specific position
        if position_filter and "position" in peer_df.columns:
            peer_df = peer_df[
                peer_df["position"].str.lower() == position_filter.lower()
            ]

        # Mark the target player
        peer_df["is_target"] = peer_df["player_id"] == player.player_id

        # Sort: target first, then by goals/90 desc
        sort_col = "Goals/90" if "Goals/90" in peer_df.columns else peer_df.columns[-2]
        peer_df = peer_df.sort_values(
            ["is_target", sort_col], ascending=[False, False]
        ).reset_index(drop=True)

        log.info(
            f"Positional benchmark: {player.name} vs {len(peer_df)-1} peers | "
            f"position_filter={position_filter}"
        )
        return peer_df

    # ── 6. Team Analysis ─────────────────────────────────────────────────────
    @timer
    def analyse_team(
        self,
        team_name: str,
        season_id: int,
    ) -> TeamReport:
        """
        Generate an overview report for a team in a specific season.

        Covers:
          • Total goals / assists
          • Top scorer
          • Average pass completion
          • Goal contribution share per player (for pie chart)

        Args:
            team_name: Exact team name as stored in the League.
            season_id: Season to analyse.

        Returns:
            TeamReport dataclass instance.

        Raises:
            ValueError: If team or season not found.
        """
        team = self._league.get_team(team_name, season_id)
        if team is None:
            available = [t.name for t in self._league.get_teams(season_id)]
            raise ValueError(
                f"Team '{team_name}' not found for season {season_id}. "
                f"Available: {available}"
            )

        squad_df = team.squad_stats_df(season_id=season_id)

        if squad_df.empty:
            raise ValueError(f"No stats for '{team_name}' in season {season_id}.")

        total_goals = int(squad_df["goals"].sum()) if "goals" in squad_df.columns else 0
        total_assists = int(squad_df["assists"].sum()) if "assists" in squad_df.columns else 0

        # Top scorer
        if "goals" in squad_df.columns and not squad_df.empty:
            top_idx = squad_df["goals"].idxmax()
            top_scorer_name = str(squad_df.loc[top_idx, "player_name"])
            top_scorer_goals = int(squad_df.loc[top_idx, "goals"])
        else:
            top_scorer_name, top_scorer_goals = "Unknown", 0

        # Average pass completion
        avg_pass_completion = 0.0
        if "pass_completion_rate" in squad_df.columns:
            avg_pass_completion = round(
                float(squad_df["pass_completion_rate"].mean()), 1
            )

        # Goal contribution share (goals + assists per player)
        goal_contribution_df = pd.DataFrame()
        if "goals" in squad_df.columns and "assists" in squad_df.columns:
            gc_df = squad_df[["player_name", "goals", "assists"]].copy()
            gc_df["goal_contributions"] = gc_df["goals"] + gc_df["assists"]
            total_gc = gc_df["goal_contributions"].sum()
            gc_df["contribution_share_%"] = (
                safe_divide(gc_df["goal_contributions"], total_gc) * 100
            ).round(1)
            goal_contribution_df = gc_df.sort_values(
                "goal_contributions", ascending=False
            ).reset_index(drop=True)

        season_name = ""
        if team.players:
            record = team.players[0].get_season(season_id)
            if record:
                season_name = record.season_name

        report = TeamReport(
            team_name=team_name,
            season_name=season_name,
            squad_size=team.squad_size,
            total_goals=total_goals,
            total_assists=total_assists,
            top_scorer_name=top_scorer_name,
            top_scorer_goals=top_scorer_goals,
            avg_pass_completion=avg_pass_completion,
            goal_contribution_df=goal_contribution_df,
            squad_stats_df=squad_df,
        )

        log.info(
            f"Team report: {team_name} | season={season_id} | "
            f"goals={total_goals} assists={total_assists}"
        )
        return report

    # ── 7. Player Report ─────────────────────────────────────────────────────
    @timer
    def player_report(
        self,
        player_name: str,
        season_id: Optional[int] = None,
        min_minutes: int = DEFAULT_MIN_MINUTES,
    ) -> Dict[str, Any]:
        """
        Generate a complete structured report for a single player.

        Combines:
          • Identity info
          • Career trajectory
          • Percentile profile
          • Summary metrics table

        This dict is the primary input for the Streamlit dashboard (Phase 6).

        Args:
            player_name: Partial or full player name.
            season_id:   Optional season filter.
            min_minutes: Min minutes for percentile ranking.

        Returns:
            Dict with keys: player, trajectory, percentile_profile, summary.
        """
        player = self._league.find_player(player_name)
        if player is None:
            raise ValueError(f"Player '{player_name}' not found.")

        trajectory = self.career_trajectory(player_name)
        profile = self.player_percentile_profile(player_name, season_id, min_minutes)

        report = {
            "player": {
                "id": player.player_id,
                "name": player.name,
                "position": player.position,
                "nationality": player.nationality,
                "age": player.age,
                "seasons_loaded": len(player.seasons),
                "type": "Goalkeeper" if isinstance(player, Goalkeeper) else "Outfielder",
            },
            "summary": player.summary(),
            "career_totals": player.career_totals(),
            "trajectory": trajectory,
            "percentile_profile": profile,
            "primary_metrics": player.primary_metrics,
        }

        log.info(f"Player report generated for: {player.name}")
        return report

    # ── 8. Similarity Search ─────────────────────────────────────────────────
    @timer
    def find_similar_players(
        self,
        player_name: str,
        season_id: Optional[int] = None,
        min_minutes: int = DEFAULT_MIN_MINUTES,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        Find the *top_n* most statistically similar players in the league.

        Uses Euclidean distance on standardised per-90 metrics (z-scores).
        Players closer in the metric space → more similar playing style.

        Args:
            player_name: Target player name.
            season_id:   Season filter.
            min_minutes: Min minutes for peers.
            top_n:       Number of similar players to return.

        Returns:
            pd.DataFrame with columns: player_name, position, similarity_score,
            and the key per-90 metrics. Sorted by similarity (ascending distance).
        """
        player = self._league.find_player(player_name)
        if player is None:
            raise ValueError(f"Player '{player_name}' not found.")

        pos_group = "goalkeeper" if isinstance(player, Goalkeeper) else "outfield"
        peer_df = self._build_percentile_table(season_id, pos_group, min_minutes).copy()

        if peer_df.empty or len(peer_df) < 2:
            log.warning("Not enough peers for similarity search.")
            return pd.DataFrame()

        metric_map = (
            self.OUTFIELD_METRICS if pos_group == "outfield" else self.GK_METRICS
        )
        metric_cols = list(metric_map.values())
        available_cols = [c for c in metric_cols if c in peer_df.columns]

        if not available_cols:
            return pd.DataFrame()

        # Standardise: subtract mean, divide by std (z-score normalisation)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled = scaler.fit_transform(peer_df[available_cols].fillna(0))
        scaled_df = pd.DataFrame(scaled, columns=available_cols, index=peer_df.index)

        # Find target player's row
        target_mask = peer_df["player_id"] == player.player_id
        if not target_mask.any():
            log.warning(f"Target player '{player.name}' not in peer table (check min_minutes).")
            return pd.DataFrame()

        target_vec = scaled_df[target_mask].values[0]

        # Euclidean distance from target to every other player
        distances = np.sqrt(((scaled_df.values - target_vec) ** 2).sum(axis=1))
        peer_df = peer_df.copy()
        peer_df["distance"] = distances
        peer_df["similarity_score"] = (
            1 / (1 + peer_df["distance"])
        ).round(3)

        # Exclude the target player itself, return top_n
        similar = (
            peer_df[peer_df["player_id"] != player.player_id]
            .sort_values("distance")
            .head(top_n)[["player_name", "position", "similarity_score"] + available_cols]
            .reset_index(drop=True)
        )
        similar.index += 1

        log.info(
            f"Similarity search: {player.name} | top {top_n} | season={season_id}"
        )
        return similar

    # ── Convenience properties ────────────────────────────────────────────────
    @property
    def league(self) -> League:
        """Read-only access to the underlying League object."""
        return self._league

    def clear_cache(self) -> None:
        """Invalidate all cached percentile tables (call after adding new data)."""
        self._cache.clear()
        log.info("Analyzer cache cleared.")

    def __repr__(self) -> str:
        return (
            f"PerformanceAnalyzer("
            f"league='{self._league.name}', "
            f"seasons={self._league.season_ids}, "
            f"cache_entries={len(self._cache)})"
        )
