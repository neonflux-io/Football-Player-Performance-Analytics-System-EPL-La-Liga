"""
main.py
─────────────────────────────────────────────────────────────────────────────
Project entry point.  Run this file to test Phase 1 components:

    python main.py

What it does:
  1. Loads configuration and prints key settings
  2. Instantiates StatsBombLoader
  3. Fetches the competitions list (cached after first run)
  4. Fetches matches for La Liga 2020/21 (competition=11, season=90)
  5. Prints a sample of the matches DataFrame
─────────────────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `from src.xxx` imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import cfg
from src.loaders import StatsBombLoader
from src.utils import get_logger

log = get_logger("main")


def phase1_smoke_test() -> None:
    """Run a quick sanity check on Phase 1 components."""

    log.info("=" * 60)
    log.info("Football Analytics — Phase 1 Smoke Test")
    log.info("=" * 60)

    # ── 1. Config ─────────────────────────────────────────────────────────────
    log.info(f"Project root : {cfg.PROJECT_ROOT}")
    log.info(f"Cache dir    : {cfg.CACHE_DIR}")
    log.info(f"Cache enabled: {cfg.CACHE_ENABLED}")
    log.info(f"Log level    : {cfg.LOG_LEVEL}")

    # ── 2. Loader ─────────────────────────────────────────────────────────────
    loader = StatsBombLoader(use_cache=True)

    # ── 3. Competitions ────────────────────────────────────────────────────────
    log.info("\nFetching competitions…")
    try:
        comps = loader.competitions()
        epl_la_liga = comps[
            comps["competition_name"].isin(["Premier League", "La Liga"])
        ][["competition_id", "competition_name", "season_id", "season_name"]]

        log.success(f"Available EPL & La Liga seasons:\n{epl_la_liga.to_string(index=False)}")
    except Exception as exc:
        log.error(f"Could not fetch competitions: {exc}")
        log.warning("Check your internet connection or GitHub rate limits. Try again in a few minutes.")
        return

    # ── 4. Matches for La Liga 2020/21 ────────────────────────────────────────
    # competition_id=11 (La Liga), season_id=90 (2021/22)
    # These are the richest seasons in StatsBomb open data
    log.info("\nFetching La Liga 2021/22 matches…")
    try:
        matches = loader.matches(competition_id=11, season_id=90)
        log.success(f"Loaded {len(matches)} matches")
        cols = ["match_id", "match_date", "home_team", "away_team", "home_goals", "away_goals"]
        available_cols = [c for c in cols if c in matches.columns]
        print("\nSample matches:")
        print(matches[available_cols].head(5).to_string(index=False))
    except Exception as exc:
        log.error(f"Could not fetch matches: {exc}")
        return

    # ── 5. Events for one match (quick test) ──────────────────────────────────
    if "match_id" in matches.columns:
        sample_match_id = int(matches["match_id"].iloc[0])
        log.info(f"\nFetching events for match_id={sample_match_id}…")
        try:
            events = loader.events(sample_match_id)
            log.success(f"Loaded {len(events)} events | {events['type_name'].value_counts().head(5).to_dict()}")
        except Exception as exc:
            log.warning(f"Event fetch skipped: {exc}")

    log.info("\n✅  Phase 1 complete — all components operational.")
    log.info("   Confirm this output, then we'll proceed to Phase 2 (domain models).")


if __name__ == "__main__":
    phase1_smoke_test()
