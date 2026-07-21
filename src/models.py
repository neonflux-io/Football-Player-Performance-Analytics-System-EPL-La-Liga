"""
models.py
─────────────────────────────────────────────────────────────────────────────
Core domain models for the Football Analytics System.

Class hierarchy
───────────────
    Player  (abstract base)
    ├── Outfielder   (inherits Player, adds attacking/pressing metrics)
    └── Goalkeeper   (inherits Player, adds save/claim metrics)

    Team             (composes Player objects)
    League           (composes Team objects, manages seasons)

OOP concepts demonstrated
──────────────────────────
• Encapsulation  – stats live in _raw_stats dict; only accessible through
                   controlled properties/methods, never modified directly.
• Abstraction    – callers use .per_90("goals") without knowing the math.
• Inheritance    – Goalkeeper/Outfielder extend Player; each overrides
                   summary() and primary_metrics property.
• Polymorphism   – PerformanceAnalyzer can call player.summary() regardless
                   of whether the player is a GK or outfielder.
• Composition    – Team holds a dict of Player objects (has-a, not is-a).
                   League holds a dict of Team objects.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import pandas as pd

from src.config import cfg
from src.utils import get_logger, per_90, safe_divide

log = get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Helper dataclass: SeasonRecord
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SeasonRecord:
    """
    Holds all raw statistics for a single player in a single season.

    Using a dataclass (Python 3.7+) gives us:
      • Free __init__, __repr__, __eq__
      • Type hints enforced at static-analysis time
      • Lightweight alternative to a full class for pure data storage

    All counting stats default to 0 so records can be partially populated.
    """

    # Identity
    competition_id: int
    season_id: int
    season_name: str
    team_name: str

    # ── Volume stats (counting) ───────────────────────────────────────────────
    minutes_played: float = 0.0
    matches_played: int = 0

    # Attacking
    goals: int = 0
    assists: int = 0
    shots: int = 0
    shots_on_target: int = 0

    # Expected metrics
    xg: float = 0.0          # Expected goals
    xa: float = 0.0          # Expected assists
    xg_overperformance: float = 0.0  # goals - xG (calculated on init)

    # Passing
    passes: int = 0
    passes_completed: int = 0
    key_passes: int = 0       # passes leading directly to a shot

    # Ball carrying
    carries: int = 0
    progressive_carries: int = 0

    # Dribbles
    dribbles_attempted: int = 0
    dribbles_completed: int = 0

    # Defensive
    pressures: int = 0
    tackles: int = 0
    interceptions: int = 0
    clearances: int = 0

    # Goalkeeper-specific (populated for GKs only)
    saves: int = 0
    goals_conceded: int = 0
    clean_sheets: int = 0

    def __post_init__(self) -> None:
        """Derived fields calculated after __init__ runs."""
        self.xg_overperformance = round(self.goals - self.xg, 2)

    # ── Computed properties ───────────────────────────────────────────────────
    @property
    def pass_completion_rate(self) -> float:
        """Percentage of passes that reached their target."""
        return round(safe_divide(self.passes_completed, self.passes) * 100, 1)

    @property
    def shot_conversion_rate(self) -> float:
        """Percentage of shots that resulted in a goal."""
        return round(safe_divide(self.goals, self.shots) * 100, 1)

    @property
    def dribble_success_rate(self) -> float:
        """Percentage of dribble attempts completed."""
        return round(
            safe_divide(self.dribbles_completed, self.dribbles_attempted) * 100, 1
        )

    @property
    def save_percentage(self) -> float:
        """Goalkeeper save % = saves / (saves + goals_conceded)."""
        return round(
            safe_divide(self.saves, self.saves + self.goals_conceded) * 100, 1
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to plain dict (used for DataFrame construction)."""
        return {
            "competition_id": self.competition_id,
            "season_id": self.season_id,
            "season_name": self.season_name,
            "team_name": self.team_name,
            "minutes_played": self.minutes_played,
            "matches_played": self.matches_played,
            "goals": self.goals,
            "assists": self.assists,
            "shots": self.shots,
            "shots_on_target": self.shots_on_target,
            "xg": self.xg,
            "xa": self.xa,
            "xg_overperformance": self.xg_overperformance,
            "passes": self.passes,
            "passes_completed": self.passes_completed,
            "pass_completion_rate": self.pass_completion_rate,
            "key_passes": self.key_passes,
            "carries": self.carries,
            "progressive_carries": self.progressive_carries,
            "dribbles_attempted": self.dribbles_attempted,
            "dribbles_completed": self.dribbles_completed,
            "dribble_success_rate": self.dribble_success_rate,
            "pressures": self.pressures,
            "tackles": self.tackles,
            "interceptions": self.interceptions,
            "clearances": self.clearances,
            "saves": self.saves,
            "goals_conceded": self.goals_conceded,
            "clean_sheets": self.clean_sheets,
            "save_percentage": self.save_percentage,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base class: Player
# ─────────────────────────────────────────────────────────────────────────────
class Player(ABC):
    """
    Abstract base class representing a football player.

    WHY ABSTRACT?
    ─────────────
    A raw "Player" is too vague to be useful — a goalkeeper's primary metrics
    (saves, clean sheets) are completely different from a striker's (goals, xG).
    By making Player abstract we FORCE every concrete subclass to implement:
      • primary_metrics  (property) – the KPIs most relevant to the position
      • summary()        (method)   – a human-readable one-liner stat block

    This enables POLYMORPHISM: PerformanceAnalyzer can call player.summary()
    on any collection of mixed player types and get meaningful output.

    Attributes (set in __init__)
    ────────────────────────────
    player_id    : Unique StatsBomb player ID
    name         : Full player name
    position     : General position label (e.g. "Forward", "Midfielder")
    nationality  : Country name
    age          : Player age (can be None if not in data)
    _seasons     : Dict[season_id → SeasonRecord]  (PRIVATE — use methods)
    """

    # Class-level constant: stat columns that exist for all positions
    BASE_STAT_COLUMNS: ClassVar[List[str]] = [
        "goals", "assists", "xg", "xa", "passes", "key_passes",
        "dribbles_completed", "pressures", "carries",
    ]

    def __init__(
        self,
        player_id: int,
        name: str,
        position: str,
        nationality: str = "Unknown",
        age: Optional[int] = None,
    ) -> None:
        # Public identity attributes
        self.player_id: int = player_id
        self.name: str = name
        self.position: str = position
        self.nationality: str = nationality
        self.age: Optional[int] = age

        # ENCAPSULATION: season records stored privately
        # External code uses .add_season(), .get_season(), .career_stats()
        self._seasons: Dict[int, SeasonRecord] = {}

    # ── Season management ─────────────────────────────────────────────────────
    def add_season(self, record: SeasonRecord) -> None:
        """
        Add or replace a SeasonRecord for this player.

        Args:
            record: A fully-populated SeasonRecord dataclass instance.
        """
        self._seasons[record.season_id] = record
        log.debug(f"[{self.name}] Added season {record.season_name} ({record.team_name})")

    def get_season(self, season_id: int) -> Optional[SeasonRecord]:
        """
        Return the SeasonRecord for a given season_id, or None.

        Args:
            season_id: StatsBomb season ID integer.
        """
        return self._seasons.get(season_id)

    @property
    def seasons(self) -> List[SeasonRecord]:
        """Return all SeasonRecords, sorted by season_id ascending."""
        return sorted(self._seasons.values(), key=lambda r: r.season_id)

    @property
    def season_ids(self) -> List[int]:
        """Return all season IDs this player has data for."""
        return sorted(self._seasons.keys())

    def has_season(self, season_id: int) -> bool:
        """Return True if this player has data for the given season."""
        return season_id in self._seasons

    # ── Per-90 interface (ABSTRACTION) ────────────────────────────────────────
    def per_90(self, stat: str, season_id: Optional[int] = None) -> float:
        """
        Return *stat* normalised to per-90-minutes for a given season.

        If season_id is None, uses the CAREER totals across all seasons.
        Returns 0.0 if the stat doesn't exist or minutes_played is 0.

        Args:
            stat:      Attribute name on SeasonRecord (e.g. "goals", "xg").
            season_id: Season to look up, or None for career aggregate.

        Returns:
            Float: per-90-minute rate.

        Example:
            player.per_90("goals")           # career goals per 90
            player.per_90("xg", season_id=90)  # xG per 90 in 2021/22
        """
        if season_id is not None:
            record = self._seasons.get(season_id)
            if record is None:
                return 0.0
            raw = getattr(record, stat, 0.0)
            return per_90(float(raw), record.minutes_played)

        # Career aggregate
        total_stat = sum(getattr(r, stat, 0.0) for r in self._seasons.values())
        total_minutes = sum(r.minutes_played for r in self._seasons.values())
        return per_90(float(total_stat), total_minutes)

    # ── Career aggregates ─────────────────────────────────────────────────────
    def career_totals(self) -> Dict[str, float]:
        """
        Sum all counting stats across all loaded seasons.

        Returns:
            Dict mapping stat name → career total.
        """
        if not self._seasons:
            return {}

        records = list(self._seasons.values())
        stat_keys = [
            "goals", "assists", "shots", "xg", "xa",
            "passes", "passes_completed", "key_passes",
            "dribbles_attempted", "dribbles_completed",
            "pressures", "carries", "minutes_played", "matches_played",
            "saves", "goals_conceded", "clean_sheets",
        ]
        return {k: sum(getattr(r, k, 0) for r in records) for k in stat_keys}

    def career_stats_df(self) -> pd.DataFrame:
        """
        Return a DataFrame with one row per season (the career trajectory table).

        Each row is a SeasonRecord serialised to dict, plus per-90 columns.
        This is the primary input for the PerformanceAnalyzer in Phase 3.

        Returns:
            pd.DataFrame sorted by season_id, with all SeasonRecord fields
            plus per-90 versions of key stats.
        """
        if not self._seasons:
            return pd.DataFrame()

        rows = []
        for record in self.seasons:
            row = record.to_dict()
            row["player_id"] = self.player_id
            row["player_name"] = self.name
            row["position"] = self.position

            # Add per-90 columns for the most important metrics
            p90_stats = ["goals", "assists", "xg", "xa", "key_passes",
                         "pressures", "dribbles_completed"]
            for stat in p90_stats:
                row[f"{stat}_p90"] = round(
                    per_90(float(getattr(record, stat, 0)), record.minutes_played), 3
                )
            rows.append(row)

        return pd.DataFrame(rows).sort_values("season_id").reset_index(drop=True)

    # ── Abstract interface (POLYMORPHISM contract) ────────────────────────────
    @property
    @abstractmethod
    def primary_metrics(self) -> Dict[str, float]:
        """
        Return the KPIs most relevant to this player's position.

        Subclasses MUST implement this.  This is the polymorphic hook that
        allows the PerformanceAnalyzer to build radar charts without knowing
        whether it's dealing with a GK, midfielder, or striker.

        Returns:
            Dict of {metric_label: value} using career per-90 rates.
        """
        ...

    @abstractmethod
    def summary(self) -> str:
        """
        Return a one-line stat summary string for this player.

        Used in reports and the dashboard sidebar.  Each position subclass
        formats its own most relevant metrics.

        Returns:
            Human-readable string, e.g.
            "Lionel Messi | FW | 0.82 goals/90 | 0.43 xG/90"
        """
        ...

    # ── Dunder methods ────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        n_seasons = len(self._seasons)
        return (
            f"{self.__class__.__name__}("
            f"id={self.player_id}, name='{self.name}', "
            f"pos='{self.position}', seasons={n_seasons})"
        )

    def __eq__(self, other: object) -> bool:
        """Two Player objects are equal if they share the same player_id."""
        if not isinstance(other, Player):
            return NotImplemented
        return self.player_id == other.player_id

    def __hash__(self) -> int:
        """Allow Player objects to be used in sets and as dict keys."""
        return hash(self.player_id)


# ─────────────────────────────────────────────────────────────────────────────
# Concrete subclass: Outfielder
# ─────────────────────────────────────────────────────────────────────────────
class Outfielder(Player):
    """
    Concrete Player subclass for all non-goalkeeper positions.

    INHERITANCE: inherits all Player infrastructure (season management,
    per_90(), career_stats_df(), etc.) and only adds/overrides what differs.

    Positions covered: Forward, Midfielder, Defender, Wing-Back.

    The primary_metrics property returns the 8 metrics used on the radar
    chart in Phase 5 — tailored to outfield play.
    """

    # Metrics displayed on the radar chart (Phase 5)
    RADAR_METRICS: ClassVar[List[str]] = [
        "goals_p90",
        "assists_p90",
        "xg_p90",
        "xa_p90",
        "key_passes_p90",
        "dribbles_completed_p90",
        "pressures_p90",
        "carries_p90",
    ]

    @property
    def primary_metrics(self) -> Dict[str, float]:
        """
        Return 8 career per-90 metrics for the outfield radar chart.

        Returns:
            Dict with metric name → career per-90 value.
        """
        return {
            "Goals/90": round(self.per_90("goals"), 3),
            "Assists/90": round(self.per_90("assists"), 3),
            "xG/90": round(self.per_90("xg"), 3),
            "xA/90": round(self.per_90("xa"), 3),
            "Key Passes/90": round(self.per_90("key_passes"), 3),
            "Dribbles/90": round(self.per_90("dribbles_completed"), 3),
            "Pressures/90": round(self.per_90("pressures"), 3),
            "Carries/90": round(self.per_90("carries"), 3),
        }

    def summary(self) -> str:
        """
        One-line summary emphasising attacking output.

        Example:
            "Lionel Messi | FW | G:0.82 A:0.43 xG:0.71 KP:2.1 /90"
        """
        m = self.primary_metrics
        totals = self.career_totals()
        return (
            f"{self.name} | {self.position} | "
            f"G:{m['Goals/90']:.2f} A:{m['Assists/90']:.2f} "
            f"xG:{m['xG/90']:.2f} xA:{m['xA/90']:.2f} "
            f"KP:{m['Key Passes/90']:.2f} /90 "
            f"| {int(totals.get('minutes_played', 0))} mins"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Concrete subclass: Goalkeeper
# ─────────────────────────────────────────────────────────────────────────────
class Goalkeeper(Player):
    """
    Concrete Player subclass for goalkeepers.

    INHERITANCE + POLYMORPHISM: same base class, completely different
    primary_metrics and summary — the analyzer treats GKs identically
    at the interface level but gets GK-specific data back.
    """

    RADAR_METRICS: ClassVar[List[str]] = [
        "saves_p90",
        "save_percentage",
        "clean_sheets",
        "goals_conceded_p90",
        "passes_p90",
        "passes_completed_p90",
    ]

    @property
    def primary_metrics(self) -> Dict[str, float]:
        """Return GK-specific career metrics."""
        totals = self.career_totals()
        total_minutes = totals.get("minutes_played", 0)
        saves = totals.get("saves", 0)
        conceded = totals.get("goals_conceded", 0)

        return {
            "Saves/90": round(per_90(saves, total_minutes), 3),
            "Save %": round(
                safe_divide(saves, saves + conceded) * 100, 1
            ),
            "Clean Sheets": int(totals.get("clean_sheets", 0)),
            "Goals Conceded/90": round(per_90(conceded, total_minutes), 3),
            "Passes/90": round(self.per_90("passes"), 3),
            "Pass Completion %": round(
                safe_divide(
                    totals.get("passes_completed", 0),
                    totals.get("passes", 1),
                ) * 100,
                1,
            ),
        }

    def summary(self) -> str:
        """
        One-line summary emphasising save metrics.

        Example:
            "Thibaut Courtois | GK | Saves:4.2/90 | Save%:74.3 | CS:12"
        """
        m = self.primary_metrics
        totals = self.career_totals()
        return (
            f"{self.name} | GK | "
            f"Saves:{m['Saves/90']:.2f}/90 "
            f"Save%:{m['Save %']:.1f} "
            f"CS:{m['Clean Sheets']} "
            f"| {int(totals.get('minutes_played', 0))} mins"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY FUNCTION: create the right Player subclass automatically
# ─────────────────────────────────────────────────────────────────────────────
def create_player(
    player_id: int,
    name: str,
    position: str,
    nationality: str = "Unknown",
    age: Optional[int] = None,
) -> Player:
    """
    Factory function: return a Goalkeeper or Outfielder based on *position*.

    WHY A FACTORY?
    ─────────────
    Callers don't need to know the class hierarchy — they just say "give me a
    player for this position" and get back the right concrete subclass.
    This decouples the rest of the system from the inheritance hierarchy.

    Args:
        player_id:   StatsBomb player ID.
        name:        Full name.
        position:    Position string from StatsBomb data.
        nationality: Country name.
        age:         Age in years.

    Returns:
        Goalkeeper instance if position is a GK variant, else Outfielder.
    """
    gk_positions = {"Goalkeeper", "GK", "Goal Keeper", "keeper"}
    if position in gk_positions or "keeper" in position.lower():
        return Goalkeeper(player_id, name, position, nationality, age)
    return Outfielder(player_id, name, position, nationality, age)


# ─────────────────────────────────────────────────────────────────────────────
# Team class
# ─────────────────────────────────────────────────────────────────────────────
class Team:
    """
    Represents a football club in a given season.

    COMPOSITION: A Team HAS players (it does not inherit from Player).
    Players are stored in a dict keyed by player_id for O(1) lookup.

    Responsibilities:
      • Store and retrieve Player objects for the squad
      • Provide squad-level aggregate statistics
      • Serialise to DataFrame for bulk analysis
    """

    def __init__(self, team_id: int, name: str, competition: str, season: str) -> None:
        """
        Initialise the team.

        Args:
            team_id:     StatsBomb team ID.
            name:        Official team name (e.g. "FC Barcelona").
            competition: League name (e.g. "La Liga").
            season:      Season string (e.g. "2021/2022").
        """
        self.team_id: int = team_id
        self.name: str = name
        self.competition: str = competition
        self.season: str = season

        # ENCAPSULATION: internal squad dict, not exposed directly
        self._squad: Dict[int, Player] = {}

    # ── Squad management ──────────────────────────────────────────────────────
    def add_player(self, player: Player) -> None:
        """
        Add a Player to the squad.

        Args:
            player: Any Player subclass instance (Outfielder or Goalkeeper).
        """
        self._squad[player.player_id] = player
        log.debug(f"[{self.name}] Added player: {player.name}")

    def get_player(self, player_id: int) -> Optional[Player]:
        """
        Return a Player by ID, or None if not in the squad.

        Args:
            player_id: StatsBomb player ID.
        """
        return self._squad.get(player_id)

    def remove_player(self, player_id: int) -> bool:
        """
        Remove a player from the squad by ID.

        Args:
            player_id: StatsBomb player ID.

        Returns:
            True if player was found and removed, False otherwise.
        """
        if player_id in self._squad:
            del self._squad[player_id]
            return True
        return False

    @property
    def players(self) -> List[Player]:
        """Return all players sorted by name."""
        return sorted(self._squad.values(), key=lambda p: p.name)

    @property
    def squad_size(self) -> int:
        """Number of players in the squad."""
        return len(self._squad)

    def get_outfielders(self) -> List[Outfielder]:
        """Return only outfield players."""
        return [p for p in self._squad.values() if isinstance(p, Outfielder)]

    def get_goalkeepers(self) -> List[Goalkeeper]:
        """Return only goalkeepers."""
        return [p for p in self._squad.values() if isinstance(p, Goalkeeper)]

    # ── Squad statistics ──────────────────────────────────────────────────────
    def squad_stats_df(self, season_id: Optional[int] = None) -> pd.DataFrame:
        """
        Build a DataFrame with one row per player, containing their stats.

        Optionally filter to a specific season.

        Args:
            season_id: If provided, only include stats for that season.
                       If None, use career totals.

        Returns:
            pd.DataFrame sorted by goals descending.
        """
        rows = []
        for player in self.players:
            if season_id is not None:
                record = player.get_season(season_id)
                if record is None:
                    continue
                row = record.to_dict()
                row["player_id"] = player.player_id
                row["player_name"] = player.name
                row["position"] = player.position
                row["nationality"] = player.nationality
            else:
                row = player.career_totals()
                row["player_id"] = player.player_id
                row["player_name"] = player.name
                row["position"] = player.position
                row["nationality"] = player.nationality
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # Sort by goals then minutes (most productive players first)
        sort_col = "goals" if "goals" in df.columns else df.columns[0]
        return df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    def top_scorers(self, n: int = 5) -> pd.DataFrame:
        """
        Return the top *n* goal scorers in the squad.

        Args:
            n: Number of players to return.

        Returns:
            pd.DataFrame with player_name, goals, xg columns.
        """
        df = self.squad_stats_df()
        if df.empty or "goals" not in df.columns:
            return pd.DataFrame()
        cols = [c for c in ["player_name", "goals", "xg", "assists", "minutes_played"] if c in df.columns]
        return df[cols].head(n)

    def __repr__(self) -> str:
        return (
            f"Team(id={self.team_id}, name='{self.name}', "
            f"competition='{self.competition}', season='{self.season}', "
            f"squad_size={self.squad_size})"
        )

    def __len__(self) -> int:
        return self.squad_size


# ─────────────────────────────────────────────────────────────────────────────
# League class
# ─────────────────────────────────────────────────────────────────────────────
class League:
    """
    Manages multiple teams across multiple seasons for a single competition.

    COMPOSITION: League HAS teams (one per season) and manages the full
    data lifecycle from raw StatsBomb data → Player objects → Team objects.

    This is the top-level domain object; the PerformanceAnalyzer works
    with League instances.

    Key responsibility: `build_from_stats_df()` is the bridge between the
    StatsBombLoader (data layer) and the domain model layer.
    """

    def __init__(
        self,
        competition_id: int,
        name: str,
        country: str = "Unknown",
    ) -> None:
        """
        Initialise the league.

        Args:
            competition_id: StatsBomb competition ID.
            name:           Human-readable competition name.
            country:        Country where the league is played.
        """
        self.competition_id: int = competition_id
        self.name: str = name
        self.country: str = country

        # Dict[season_id → Dict[team_name → Team]]
        # Two-level dict: seasons → teams within that season
        self._seasons: Dict[int, Dict[str, Team]] = {}

    # ── Season / team management ──────────────────────────────────────────────
    def add_team(self, team: Team, season_id: int) -> None:
        """
        Register a Team in the league for a specific season.

        Args:
            team:      A populated Team object.
            season_id: The season this team belongs to.
        """
        if season_id not in self._seasons:
            self._seasons[season_id] = {}
        self._seasons[season_id][team.name] = team
        log.debug(f"[{self.name}] Registered team '{team.name}' for season {season_id}")

    def get_team(self, team_name: str, season_id: int) -> Optional[Team]:
        """
        Return a Team by name and season, or None.

        Args:
            team_name: Exact team name as stored.
            season_id: Season ID.
        """
        return self._seasons.get(season_id, {}).get(team_name)

    def get_teams(self, season_id: int) -> List[Team]:
        """Return all teams for a given season, sorted by name."""
        return sorted(self._seasons.get(season_id, {}).values(), key=lambda t: t.name)

    @property
    def season_ids(self) -> List[int]:
        """Return all loaded season IDs, sorted."""
        return sorted(self._seasons.keys())

    def all_players(self, season_id: Optional[int] = None) -> List[Player]:
        """
        Return every player across all teams (optionally for one season).

        Args:
            season_id: If provided, filter to a single season.

        Returns:
            List of Player objects (all subclasses).
        """
        seasons_to_check = (
            [season_id] if season_id is not None else self.season_ids
        )
        seen_ids: set = set()
        players: List[Player] = []

        for sid in seasons_to_check:
            for team in self.get_teams(sid):
                for player in team.players:
                    if player.player_id not in seen_ids:
                        players.append(player)
                        seen_ids.add(player.player_id)

        return players

    def find_player(self, name: str) -> Optional[Player]:
        """
        Find a player by name (case-insensitive partial match).

        Args:
            name: Player name or partial name to search.

        Returns:
            First matching Player, or None if not found.
        """
        name_lower = name.lower()
        for player in self.all_players():
            if name_lower in player.name.lower():
                return player
        return None

    def find_player_by_id(self, player_id: int) -> Optional[Player]:
        """Find a player by their StatsBomb player_id."""
        for player in self.all_players():
            if player.player_id == player_id:
                return player
        return None

    # ── The bridge method: raw data → domain objects ──────────────────────────
    @classmethod
    def build_from_stats_df(
        cls,
        stats_df: pd.DataFrame,
        competition_id: int,
        competition_name: str,
        season_id: int,
        season_name: str,
        country: str = "Unknown",
    ) -> "League":
        """
        Build a fully-populated League from a player stats DataFrame.

        This is the critical BRIDGE between the data layer (StatsBombLoader)
        and the domain layer (Player, Team, League).

        The stats_df is produced by StatsBombLoader.player_season_stats().
        Each row is one player; columns are the aggregated stats.

        Args:
            stats_df:         DataFrame from StatsBombLoader.player_season_stats().
            competition_id:   StatsBomb competition ID.
            competition_name: Human-readable league name.
            season_id:        StatsBomb season ID.
            season_name:      Human-readable season string (e.g. "2021/2022").
            country:          Country of the competition.

        Returns:
            League instance with all teams and players populated.
        """
        log.info(
            f"Building League '{competition_name}' season {season_name} "
            f"from {len(stats_df)} player rows…"
        )

        league = cls(competition_id, competition_name, country)
        team_registry: Dict[str, Team] = {}

        required_cols = {"player_id", "player_name", "team_name"}
        missing = required_cols - set(stats_df.columns)
        if missing:
            raise ValueError(f"stats_df is missing required columns: {missing}")

        for _, row in stats_df.iterrows():
            team_name = str(row.get("team_name", "Unknown"))
            player_id = int(row["player_id"])
            player_name = str(row["player_name"])

            # Infer position (StatsBomb sometimes gives us this; otherwise Unknown)
            position = str(row.get("position", "Unknown"))

            # ── Get-or-create Team ────────────────────────────────────────────
            if team_name not in team_registry:
                # Use a hash of team_name as a surrogate team_id
                team_id = abs(hash(team_name)) % 100_000
                team = Team(
                    team_id=team_id,
                    name=team_name,
                    competition=competition_name,
                    season=season_name,
                )
                team_registry[team_name] = team
            else:
                team = team_registry[team_name]

            # ── Create Player (factory selects subclass) ──────────────────────
            existing_player = team.get_player(player_id)
            if existing_player is None:
                player = create_player(
                    player_id=player_id,
                    name=player_name,
                    position=position,
                )
                team.add_player(player)
            else:
                player = existing_player

            # ── Build SeasonRecord from DataFrame row ─────────────────────────
            def _int(col: str) -> int:
                return int(row.get(col, 0) or 0)

            def _float(col: str) -> float:
                return float(row.get(col, 0.0) or 0.0)

            record = SeasonRecord(
                competition_id=competition_id,
                season_id=season_id,
                season_name=season_name,
                team_name=team_name,
                minutes_played=_float("minutes_played"),
                goals=_int("goals"),
                assists=_int("assists"),
                shots=_int("shots"),
                xg=_float("xg"),
                xa=_float("xa"),
                passes=_int("passes"),
                passes_completed=_int("passes_completed"),
                key_passes=_int("key_passes"),
                carries=_int("carries"),
                dribbles_attempted=_int("dribbles_attempted"),
                dribbles_completed=_int("dribbles_completed"),
                pressures=_int("pressures"),
            )
            player.add_season(record)

        # Register all teams in the league
        for team in team_registry.values():
            league.add_team(team, season_id)

        n_teams = len(team_registry)
        n_players = sum(t.squad_size for t in team_registry.values())
        log.success(
            f"League built: '{competition_name}' {season_name} | "
            f"{n_teams} teams | {n_players} players"
        )
        return league

    # ── League-wide stats ─────────────────────────────────────────────────────
    def league_stats_df(self, season_id: Optional[int] = None) -> pd.DataFrame:
        """
        Return a DataFrame with one row per player across all teams.

        Useful for percentile rankings and league-wide comparisons.

        Args:
            season_id: Optional filter to a single season.

        Returns:
            pd.DataFrame concatenating all team.squad_stats_df() results.
        """
        frames: List[pd.DataFrame] = []
        seasons_to_check = (
            [season_id] if season_id is not None else self.season_ids
        )
        for sid in seasons_to_check:
            for team in self.get_teams(sid):
                df = team.squad_stats_df(season_id=sid)
                if not df.empty:
                    df["team_name"] = team.name
                    frames.append(df)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        return combined.sort_values("goals", ascending=False).reset_index(drop=True)

    def top_scorers(self, season_id: Optional[int] = None, n: int = 10) -> pd.DataFrame:
        """
        Return the top *n* goal scorers across the entire league.

        Args:
            season_id: Filter to one season, or None for career totals.
            n:         Number of players to return.
        """
        df = self.league_stats_df(season_id)
        if df.empty or "goals" not in df.columns:
            return pd.DataFrame()
        cols = [c for c in ["player_name", "team_name", "goals", "xg", "assists"] if c in df.columns]
        return df[cols].drop_duplicates("player_name").head(n)

    def __repr__(self) -> str:
        n_seasons = len(self._seasons)
        n_teams = sum(len(teams) for teams in self._seasons.values())
        return (
            f"League(id={self.competition_id}, name='{self.name}', "
            f"seasons={n_seasons}, teams={n_teams})"
        )
