"""
loaders.py
─────────────────────────────────────────────────────────────────────────────
StatsBombLoader – responsible for fetching and preprocessing StatsBomb open
data.  This is a **pure data-access layer**; it knows nothing about football
analytics logic (that belongs in analyzer.py).

OOP concepts demonstrated:
  • **Abstraction**     – public methods hide the complexity of HTTP + caching
  • **Encapsulation**   – private helpers (_get, _cache_path) are internal only
  • **Single Responsibility** – this class ONLY loads data; it does NOT analyse

Architecture note:
  StatsBomb distributes free open data as JSON files on GitHub.  The structure
  is hierarchical:

      competitions.json
      └── matches/{competition_id}/{season_id}.json
              └── events/{match_id}.json
              └── lineups/{match_id}.json

  We fetch each level lazily and cache to disk so repeated runs are fast.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from src.config import cfg
from src.utils import get_logger, retry, timer

log = get_logger()


class StatsBombLoader:
    """
    Fetches StatsBomb open-data JSON and returns clean pandas DataFrames.

    All network calls go through _get() which implements:
      1. Disk-cache lookup  → return immediately if fresh data exists
      2. HTTP fetch with exponential-backoff retry
      3. Cache write        → persist for future runs

    Public API
    ──────────
    competitions()              → DataFrame of all available competitions
    matches(competition_id, season_id) → DataFrame of matches in a season
    events(match_id)            → DataFrame of all events in a match
    lineups(match_id)           → DataFrame of lineups for a match
    season_events(competition_id, season_id) → all events for a full season
    player_season_stats(competition_id, season_id) → aggregated player stats
    """

    # Base URL for StatsBomb open-data GitHub repository
    _BASE_URL: str = cfg.STATSBOMB_BASE_URL

    def __init__(self, use_cache: bool = True) -> None:
        """
        Initialise the loader.

        Args:
            use_cache: If True (default), JSON responses are saved to disk and
                       re-used on subsequent calls.  Set False to always fetch
                       fresh data (e.g. in CI pipelines).
        """
        self._use_cache: bool = use_cache and cfg.CACHE_ENABLED
        self._cache_dir: Path = cfg.CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._session: requests.Session = self._build_session()
        log.info(f"StatsBombLoader initialised | cache={'on' if self._use_cache else 'off'}")

    # ── Session setup ─────────────────────────────────────────────────────────
    @staticmethod
    def _build_session() -> requests.Session:
        """Create a requests.Session with sensible headers."""
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "FootballAnalytics/1.0 (open-data research project)",
                "Accept": "application/json",
            }
        )
        return session

    # ── Cache helpers ─────────────────────────────────────────────────────────
    def _cache_path(self, url: str) -> Path:
        """
        Convert a URL to a local cache file path.

        We replace '/' with '__' in the URL path portion so the filename
        stays flat inside the cache directory.

        Example:
            https://raw.github.../data/matches/2/44.json
            → {CACHE_DIR}/matches__2__44.json
        """
        # Strip the base URL prefix and clean up the path segment
        relative = url.replace(self._BASE_URL, "").lstrip("/")
        safe_name = relative.replace("/", "__")
        return self._cache_dir / safe_name

    def _load_from_cache(self, url: str) -> Optional[Any]:
        """Return cached JSON data or None if cache miss."""
        path = self._cache_path(url)
        if path.exists():
            log.debug(f"Cache HIT  → {path.name}")
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        log.debug(f"Cache MISS → {path.name}")
        return None

    def _write_to_cache(self, url: str, data: Any) -> None:
        """Persist JSON data to the cache directory."""
        path = self._cache_path(url)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh)
        log.debug(f"Cached     → {path.name}")

    # ── Core HTTP layer ───────────────────────────────────────────────────────
    @retry(
        max_attempts=cfg.REQUEST_MAX_RETRIES,
        backoff=cfg.REQUEST_RETRY_BACKOFF,
        exceptions=(requests.HTTPError, requests.ConnectionError, requests.Timeout),
    )
    def _fetch(self, url: str) -> Any:
        """
        Fetch JSON from *url* with timeout and HTTP error checking.

        The @retry decorator (from utils.py) will automatically retry this
        method up to REQUEST_MAX_RETRIES times on network errors.
        """
        response = self._session.get(url, timeout=cfg.REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()

    def _get(self, endpoint: str) -> Any:
        """
        Main data-retrieval method: cache → network → cache.

        Args:
            endpoint: Path relative to _BASE_URL  (e.g. "matches/2/44.json")

        Returns:
            Parsed JSON (dict or list).
        """
        url = f"{self._BASE_URL}/{endpoint}"

        if self._use_cache:
            cached = self._load_from_cache(url)
            if cached is not None:
                return cached

        log.info(f"Fetching   → {url}")
        data = self._fetch(url)

        if self._use_cache:
            self._write_to_cache(url, data)

        return data

    # ── Public data methods ───────────────────────────────────────────────────
    @timer
    def competitions(self) -> pd.DataFrame:
        """
        Return a DataFrame of all competitions available in StatsBomb open data.

        Columns include: competition_id, competition_name, season_id,
        season_name, competition_gender, country_name.

        Returns:
            pd.DataFrame with one row per (competition, season) pair.
        """
        data = self._get("competitions.json")
        df = pd.DataFrame(data)
        log.success(f"Loaded {len(df)} competition-season pairs")
        return df

    @timer
    def matches(self, competition_id: int, season_id: int) -> pd.DataFrame:
        """
        Return a DataFrame of all matches for a competition season.

        Args:
            competition_id: StatsBomb competition ID (e.g. 11 for La Liga).
            season_id:      StatsBomb season ID (e.g. 90 for 2021/22).

        Returns:
            pd.DataFrame with one row per match.
        """
        endpoint = f"matches/{competition_id}/{season_id}.json"
        data = self._get(endpoint)
        df = pd.json_normalize(data)  # flatten nested dicts (teams, scores, etc.)

        # ── Standardise column names ──────────────────────────────────────────
        df.rename(
            columns={
                "home_team.home_team_name": "home_team",
                "away_team.away_team_name": "away_team",
                "home_score": "home_goals",
                "away_score": "away_goals",
                "competition.competition_name": "competition",
                "season.season_name": "season",
            },
            inplace=True,
        )
        log.success(
            f"Loaded {len(df)} matches | competition={competition_id}, season={season_id}"
        )
        return df

    @timer
    def events(self, match_id: int) -> pd.DataFrame:
        """
        Return a DataFrame of all on-pitch events for a single match.

        StatsBomb event data is extremely rich: ~70+ columns covering shots,
        passes, dribbles, pressures, ball receipts, etc.

        Args:
            match_id: StatsBomb match ID (from the matches DataFrame).

        Returns:
            pd.DataFrame with one row per event (~2,000–4,000 rows per match).
        """
        endpoint = f"events/{match_id}.json"
        data = self._get(endpoint)
        df = pd.json_normalize(data, sep="_")
        log.debug(f"Loaded {len(df)} events for match_id={match_id}")
        return df

    @timer
    def lineups(self, match_id: int) -> pd.DataFrame:
        """
        Return a DataFrame containing lineup info for both teams in a match.

        Args:
            match_id: StatsBomb match ID.

        Returns:
            pd.DataFrame with columns: team_id, team_name, player_id,
            player_name, jersey_number, country, positions, cards.
        """
        endpoint = f"lineups/{match_id}.json"
        data = self._get(endpoint)

        rows: List[Dict[str, Any]] = []
        for team in data:
            for player in team.get("players", []):
                rows.append(
                    {
                        "team_id": team["team_id"],
                        "team_name": team["team_name"],
                        "player_id": player["player_id"],
                        "player_name": player["player_name"],
                        "jersey_number": player.get("jersey_number"),
                        "country": player.get("country", {}).get("name"),
                        "positions": [
                            p["position"]["name"]
                            for p in player.get("positions", [])
                        ],
                        "cards": player.get("cards", []),
                    }
                )

        df = pd.DataFrame(rows)
        log.debug(f"Loaded lineups for match_id={match_id}: {len(df)} players")
        return df

    # ── Higher-level aggregation methods ──────────────────────────────────────
    @timer
    def season_events(
        self,
        competition_id: int,
        season_id: int,
        max_matches: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load and concatenate events for every match in a season.

        This is a potentially expensive call (hundreds of HTTP requests if not
        cached).  A progress bar is shown via tqdm.

        Args:
            competition_id: StatsBomb competition ID.
            season_id:      StatsBomb season ID.
            max_matches:    Optional cap on the number of matches to load
                            (useful for quick tests / development).

        Returns:
            pd.DataFrame with all events, plus 'match_id' column added.
        """
        from tqdm import tqdm  # local import to keep startup fast

        matches_df = self.matches(competition_id, season_id)
        match_ids = matches_df["match_id"].tolist()

        if max_matches is not None:
            match_ids = match_ids[:max_matches]
            log.info(f"Capped to first {max_matches} matches")

        all_events: List[pd.DataFrame] = []
        log.info(f"Loading events for {len(match_ids)} matches…")

        for mid in tqdm(match_ids, desc="Loading matches", unit="match"):
            try:
                ev = self.events(mid)
                ev["match_id"] = mid
                all_events.append(ev)
            except Exception as exc:
                log.warning(f"Skipping match_id={mid}: {exc}")

        if not all_events:
            log.error("No events loaded — returning empty DataFrame")
            return pd.DataFrame()

        combined = pd.concat(all_events, ignore_index=True)
        log.success(
            f"Season events loaded | {len(combined):,} events across {len(all_events)} matches"
        )
        return combined

    @timer
    def player_season_stats(
        self,
        competition_id: int,
        season_id: int,
        max_matches: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Aggregate event-level data into player-level season statistics.

        This is the primary data source for the Player class in Phase 2.
        Metrics extracted per player per season:
          - Minutes played (approximated from starting + substitution events)
          - Goals, assists
          - Shots (total, on target)
          - xG (expected goals from shot events)
          - xA (expected assists from pass events leading to shots)
          - Key passes
          - Passes (total, completed)
          - Progressive carries
          - Dribbles (attempted, completed)
          - Pressures

        Args:
            competition_id: StatsBomb competition ID.
            season_id:      StatsBomb season ID.
            max_matches:    Cap for dev/testing.

        Returns:
            pd.DataFrame indexed by player, with aggregated stats columns.
        """
        events_df = self.season_events(competition_id, season_id, max_matches)

        if events_df.empty:
            return pd.DataFrame()

        log.info("Aggregating player-level statistics…")

        # Helper: safely get a column (returns Series of NaN if missing)
        def col(name: str) -> pd.Series:
            return events_df.get(name, pd.Series(dtype=float))

        # ── Goals ─────────────────────────────────────────────────────────────
        shots = events_df[events_df["type_name"] == "Shot"].copy()
        goals = shots[shots.get("shot_outcome_name", pd.Series()) == "Goal"]

        goals_per_player = (
            goals.groupby(["player_id", "player_name", "team_name"])
            .size()
            .reset_index(name="goals")
        )

        # ── Shots ─────────────────────────────────────────────────────────────
        shots_per_player = (
            shots.groupby(["player_id", "player_name", "team_name"])
            .size()
            .reset_index(name="shots")
        )

        # ── xG ────────────────────────────────────────────────────────────────
        if "shot_statsbomb_xg" in shots.columns:
            xg_per_player = (
                shots.groupby(["player_id", "player_name", "team_name"])["shot_statsbomb_xg"]
                .sum()
                .reset_index(name="xg")
            )
        else:
            xg_per_player = goals_per_player[["player_id", "player_name", "team_name"]].copy()
            xg_per_player["xg"] = 0.0

        # ── Passes ────────────────────────────────────────────────────────────
        passes = events_df[events_df["type_name"] == "Pass"].copy()

        passes_total = (
            passes.groupby(["player_id", "player_name", "team_name"])
            .size()
            .reset_index(name="passes")
        )

        if "pass_outcome_name" in passes.columns:
            completed_passes = passes[passes["pass_outcome_name"].isna()]
        else:
            completed_passes = passes.copy()

        passes_completed = (
            completed_passes.groupby(["player_id", "player_name", "team_name"])
            .size()
            .reset_index(name="passes_completed")
        )

        # ── Key passes & xA ───────────────────────────────────────────────────
        if "pass_goal_assist" in passes.columns:
            assists_df = passes[passes["pass_goal_assist"] == True]
            assists_per_player = (
                assists_df.groupby(["player_id", "player_name", "team_name"])
                .size()
                .reset_index(name="assists")
            )
        else:
            assists_per_player = passes_total[["player_id", "player_name", "team_name"]].copy()
            assists_per_player["assists"] = 0

        if "pass_shot_assist" in passes.columns:
            key_passes_df = passes[passes["pass_shot_assist"] == True]
            key_passes_per_player = (
                key_passes_df.groupby(["player_id", "player_name", "team_name"])
                .size()
                .reset_index(name="key_passes")
            )
        else:
            key_passes_per_player = passes_total[["player_id", "player_name", "team_name"]].copy()
            key_passes_per_player["key_passes"] = 0

        if "pass_xa" in passes.columns:
            xa_per_player = (
                passes.groupby(["player_id", "player_name", "team_name"])["pass_xa"]
                .sum()
                .reset_index(name="xa")
            )
        else:
            xa_per_player = passes_total[["player_id", "player_name", "team_name"]].copy()
            xa_per_player["xa"] = 0.0

        # ── Dribbles ──────────────────────────────────────────────────────────
        dribbles = events_df[events_df["type_name"] == "Dribble"].copy()

        dribbles_attempted = (
            dribbles.groupby(["player_id", "player_name", "team_name"])
            .size()
            .reset_index(name="dribbles_attempted")
        )

        if "dribble_outcome_name" in dribbles.columns:
            dribbles_completed_df = dribbles[dribbles["dribble_outcome_name"] == "Complete"]
            dribbles_completed = (
                dribbles_completed_df.groupby(["player_id", "player_name", "team_name"])
                .size()
                .reset_index(name="dribbles_completed")
            )
        else:
            dribbles_completed = dribbles_attempted.copy()
            dribbles_completed["dribbles_completed"] = 0

        # ── Pressures ─────────────────────────────────────────────────────────
        pressures = events_df[events_df["type_name"] == "Pressure"]
        pressures_per_player = (
            pressures.groupby(["player_id", "player_name", "team_name"])
            .size()
            .reset_index(name="pressures")
        )

        # ── Carries (progressive) ─────────────────────────────────────────────
        carries = events_df[events_df["type_name"] == "Carry"].copy()
        carries_per_player = (
            carries.groupby(["player_id", "player_name", "team_name"])
            .size()
            .reset_index(name="carries")
        )

        # ── Minutes played (via substitution events) ──────────────────────────
        minutes_df = self._estimate_minutes(events_df)

        # ── Merge all stat DataFrames on player_id ────────────────────────────
        base_cols = ["player_id", "player_name", "team_name"]

        dfs_to_merge = [
            goals_per_player,
            shots_per_player,
            xg_per_player,
            assists_per_player,
            key_passes_per_player,
            xa_per_player,
            passes_total,
            passes_completed,
            dribbles_attempted,
            dribbles_completed,
            pressures_per_player,
            carries_per_player,
        ]

        # Start with all unique players
        all_players = (
            events_df[events_df["player_id"].notna()]
            .drop_duplicates(subset=["player_id"])[base_cols]
            .reset_index(drop=True)
        )

        merged = all_players.copy()
        for df in dfs_to_merge:
            if not df.empty and "player_id" in df.columns:
                merged = merged.merge(df, on=base_cols, how="left")

        # Merge minutes
        if not minutes_df.empty:
            merged = merged.merge(minutes_df, on="player_id", how="left")
        else:
            merged["minutes_played"] = 0

        # Fill NaN stat columns with 0
        stat_cols = [
            c for c in merged.columns if c not in base_cols + ["minutes_played"]
        ]
        merged[stat_cols] = merged[stat_cols].fillna(0)
        merged["minutes_played"] = merged["minutes_played"].fillna(0)

        # Add competition metadata
        merged["competition_id"] = competition_id
        merged["season_id"] = season_id

        log.success(f"Player stats aggregated | {len(merged)} players")
        return merged

    # ── Private helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _estimate_minutes(events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate minutes played per player using substitution events.

        Logic:
          • A player who starts and is NOT substituted off played ≈ 90 min
          • A player substituted OFF played until the substitution minute
          • A player substituted ON played from that minute to 90 (or 120)

        This is an approximation; StatsBomb does not give exact per-player
        on-pitch minutes in the free tier.

        Returns:
            pd.DataFrame with columns [player_id, minutes_played].
        """
        if "type_name" not in events_df.columns:
            return pd.DataFrame(columns=["player_id", "minutes_played"])

        subs = events_df[events_df["type_name"] == "Substitution"].copy()

        # Players who were substituted OFF
        subbed_off: Dict[int, float] = {}
        subbed_on: Dict[int, float] = {}

        if not subs.empty and "player_id" in subs.columns:
            for _, row in subs.iterrows():
                minute = row.get("minute", 90) or 90
                pid = row.get("player_id")
                if pd.notna(pid):
                    subbed_off[int(pid)] = float(minute)

                # The player coming ON
                on_id = row.get("substitution_replacement_id")
                if pd.notna(on_id):
                    subbed_on[int(on_id)] = float(minute)

        # All players who appeared
        player_ids = events_df[events_df["player_id"].notna()]["player_id"].unique()
        rows = []
        for pid in player_ids:
            pid_int = int(pid)
            if pid_int in subbed_on:
                # Came on as substitute
                minutes = 90.0 - subbed_on[pid_int]
            elif pid_int in subbed_off:
                # Substituted off
                minutes = subbed_off[pid_int]
            else:
                # Started and finished
                minutes = 90.0
            rows.append({"player_id": pid_int, "minutes_played": max(0.0, minutes)})

        return pd.DataFrame(rows)

    # ── Convenience class method ──────────────────────────────────────────────
    @classmethod
    def available_competitions(cls) -> pd.DataFrame:
        """
        Quick utility: return EPL & La Liga rows from competitions().

        Returns:
            pd.DataFrame filtered to the leagues defined in cfg.SUPPORTED_COMPETITIONS.
        """
        loader = cls()
        comps = loader.competitions()
        target_names = list(cfg.SUPPORTED_COMPETITIONS.keys())
        filtered = comps[comps["competition_name"].isin(target_names)]
        return filtered.sort_values(["competition_name", "season_name"])
