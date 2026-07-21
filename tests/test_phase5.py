"""
tests/test_phase5.py
─────────────────────────────────────────────────────────────────────────────
Phase 5 visualisation tests.

All charts render headlessly (Agg backend) — we validate:
  • Return type is matplotlib.figure.Figure
  • Figure has at least one axes
  • No exception is raised on edge-case inputs (empty data, 1 season, etc.)
  • save() produces a file on disk
  • to_bytes() returns non-empty bytes

Run with:  pytest tests/test_phase5.py -v
─────────────────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analyzer import ComparisonResult, PercentileProfile, TrajectoryResult
from src.advanced_analysis import ClusterResult, InjuryImpactResult
from src.visualizations import (
    BaseChart,
    CareerPlotter,
    ClusterPlotter,
    InjuryPlotter,
    PitchMapper,
    RadarChart,
    ReportFigure,
    VisualisationFactory,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_profile() -> PercentileProfile:
    return PercentileProfile(
        player_name="Lionel Messi",
        player_id=5503,
        position_group="outfield",
        season_id=90,
        min_minutes=450,
        percentiles={
            "Goals/90":       91.0,
            "Assists/90":     88.0,
            "xG/90":          89.0,
            "xA/90":          85.0,
            "Key Passes/90":  87.0,
            "Dribbles/90":    95.0,
            "Pressures/90":   62.0,
            "Carries/90":     78.0,
        },
        raw_values={
            "Goals/90":       0.60,
            "Assists/90":     0.36,
            "xG/90":          0.53,
            "xA/90":          0.24,
            "Key Passes/90":  1.20,
            "Dribbles/90":    1.05,
            "Pressures/90":   4.50,
            "Carries/90":     6.00,
        },
        n_peers=45,
    )


@pytest.fixture
def sample_trajectory() -> TrajectoryResult:
    df = pd.DataFrame({
        "season_name":       ["2019/20", "2020/21", "2021/22"],
        "season_id":         [27, 42, 90],
        "goals_p90":         [0.52, 0.59, 0.60],
        "xg_p90":            [0.44, 0.50, 0.53],
        "assists_p90":       [0.28, 0.34, 0.36],
        "key_passes_p90":    [1.10, 1.18, 1.20],
        "pressures_p90":     [4.20, 4.35, 4.50],
        "xg_overperformance":[0.08, 0.09, 0.07],
        "goals":             [14, 16, 20],
        "minutes_played":    [2430, 2700, 3000],
    })
    return TrajectoryResult(
        player_name="Lionel Messi",
        player_id=5503,
        df=df,
        peak_season_name="2021/22",
        peak_metric="Goals/90",
        peak_value=0.60,
        trend_slope=0.04,
        trend_direction="improving",
    )


@pytest.fixture
def sample_comparison(sample_profile) -> ComparisonResult:
    metrics = list(sample_profile.percentiles.keys())
    metrics_b = {m: v * 0.88 for m, v in sample_profile.raw_values.items()}
    pct_b = {m: max(p - 10, 0) for m, p in sample_profile.percentiles.items()}
    edge = {m: "Lionel Messi" if sample_profile.raw_values[m] > metrics_b[m] else "draw"
            for m in metrics}

    rows = [{"Metric": m,
             "Lionel Messi (raw)": sample_profile.raw_values[m],
             "Cristiano Ronaldo (raw)": metrics_b[m],
             "Lionel Messi (pct)": sample_profile.percentiles[m],
             "Cristiano Ronaldo (pct)": pct_b[m],
             "Edge": edge[m]} for m in metrics]

    return ComparisonResult(
        player_a_name="Lionel Messi",
        player_b_name="Cristiano Ronaldo",
        metrics_a=sample_profile.raw_values,
        metrics_b=metrics_b,
        percentiles_a=sample_profile.percentiles,
        percentiles_b=pct_b,
        comparison_df=pd.DataFrame(rows),
        edge=edge,
    )


@pytest.fixture
def sample_cluster_result() -> ClusterResult:
    np.random.seed(42)
    n = 20
    df = pd.DataFrame({
        "player_id":      range(n),
        "player_name":    [f"Player {i}" for i in range(n)],
        "position":       ["Forward"] * 10 + ["Midfielder"] * 10,
        "minutes_played": np.random.randint(1000, 3000, n),
        "Goals/90":       np.random.uniform(0, 1.2, n),
        "Assists/90":     np.random.uniform(0, 0.8, n),
        "xG/90":          np.random.uniform(0, 1.0, n),
        "xA/90":          np.random.uniform(0, 0.6, n),
        "Key Passes/90":  np.random.uniform(0.5, 3.0, n),
        "Dribbles/90":    np.random.uniform(0, 2.0, n),
        "Pressures/90":   np.random.uniform(2, 8, n),
        "Carries/90":     np.random.uniform(3, 10, n),
        "cluster_id":     [0]*5 + [1]*5 + [2]*5 + [3]*5,
        "archetype":      ["Clinical Finisher"]*5 + ["Creative Playmaker"]*5 +
                          ["High-Press Engine"]*5 + ["Progressive Ball-Carrier"]*5,
    })
    profiles = pd.DataFrame({
        "cluster_id": [0, 1, 2, 3],
        "archetype":  ["Clinical Finisher", "Creative Playmaker",
                       "High-Press Engine", "Progressive Ball-Carrier"],
        "n_players":  [5, 5, 5, 5],
        "Goals/90":   [0.90, 0.30, 0.20, 0.15],
        "Assists/90": [0.20, 0.60, 0.25, 0.30],
        "Pressures/90": [3.0, 4.5, 7.0, 5.0],
        "Carries/90": [5.0, 6.0, 5.5, 8.5],
    })
    return ClusterResult(
        cluster_df=df,
        profiles=profiles,
        labels={0: "Clinical Finisher", 1: "Creative Playmaker",
                2: "High-Press Engine", 3: "Progressive Ball-Carrier"},
        n_clusters=4,
        silhouette=0.34,
        inertia=42.1,
    )


@pytest.fixture
def sample_injury_result() -> InjuryImpactResult:
    return InjuryImpactResult(
        player_name="Lionel Messi",
        player_id=5503,
        affected_season_name="2020/21",
        affected_season_minutes=1350.0,
        pre_metrics={"goals": 0.52, "assists": 0.28, "xg": 0.44,
                     "xa": 0.22, "key_passes": 1.10, "dribbles_completed": 1.00, "pressures": 4.20},
        post_metrics={"goals": 0.60, "assists": 0.36, "xg": 0.53,
                      "xa": 0.24, "key_passes": 1.20, "dribbles_completed": 1.05, "pressures": 4.50},
        delta={"goals": 0.08, "assists": 0.08, "xg": 0.09,
               "xa": 0.02, "key_passes": 0.10, "dribbles_completed": 0.05, "pressures": 0.30},
        recovery_score=0.72,
        interpretation="Messi shows strong recovery post-absence: 7/7 metrics improved.",
        trajectory_df=pd.DataFrame(),
    )


@pytest.fixture
def sample_events_df() -> pd.DataFrame:
    """Synthetic events DataFrame with pass, shot, and pressure events."""
    np.random.seed(0)
    n = 50
    rows = []
    for i in range(n):
        event_type = ["Pass", "Shot", "Pressure"][i % 3]
        lx = float(np.random.uniform(10, 110))
        ly = float(np.random.uniform(5, 75))
        row = {
            "type_name": event_type,
            "player_id": 5503,
            "player_name": "Lionel Messi",
            "minute": np.random.randint(1, 90),
            "location": [lx, ly],
        }
        if event_type == "Pass":
            row["pass_end_location"] = [lx + np.random.uniform(5, 20), ly + np.random.uniform(-5, 5)]
            row["pass_outcome_name"] = None if i % 5 != 0 else "Incomplete"
        if event_type == "Shot":
            row["shot_statsbomb_xg"] = float(np.random.uniform(0.03, 0.35))
            row["shot_outcome_name"] = "Goal" if i % 7 == 0 else "Saved"
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def factory() -> VisualisationFactory:
    return VisualisationFactory()


# ─────────────────────────────────────────────────────────────────────────────
# RadarChart tests
# ─────────────────────────────────────────────────────────────────────────────
class TestRadarChart:
    def test_plot_returns_figure(self, sample_profile):
        chart = RadarChart()
        fig = chart.plot(sample_profile)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_has_axes(self, sample_profile):
        chart = RadarChart()
        fig = chart.plot(sample_profile)
        assert len(fig.get_axes()) > 0
        plt.close("all")

    def test_comparison_returns_figure(self, sample_comparison):
        chart = RadarChart()
        fig = chart.plot_comparison(sample_comparison)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_empty_profile_does_not_crash(self):
        chart = RadarChart()
        empty_profile = PercentileProfile(
            player_name="Ghost", player_id=0,
            position_group="outfield", season_id=None,
            min_minutes=450, percentiles={}, raw_values={}, n_peers=0,
        )
        fig = chart.plot(empty_profile)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_custom_title_is_accepted(self, sample_profile):
        chart = RadarChart()
        fig = chart.plot(sample_profile, title="Custom Title")
        assert isinstance(fig, plt.Figure)
        plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# CareerPlotter tests
# ─────────────────────────────────────────────────────────────────────────────
class TestCareerPlotter:
    def test_plot_returns_figure(self, sample_trajectory):
        plotter = CareerPlotter()
        fig = plotter.plot(sample_trajectory)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_single_season_does_not_crash(self):
        """One data point — trend line should gracefully degrade."""
        df = pd.DataFrame({
            "season_name": ["2021/22"],
            "season_id": [90],
            "goals_p90": [0.60],
            "xg_p90": [0.53],
            "assists_p90": [0.36],
            "key_passes_p90": [1.20],
        })
        traj = TrajectoryResult(
            player_name="Test", player_id=1, df=df,
            peak_season_name="2021/22", peak_metric="Goals/90",
            peak_value=0.60, trend_slope=0.0, trend_direction="stable",
        )
        plotter = CareerPlotter()
        fig = plotter.plot(traj)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_missing_metrics_column_does_not_crash(self, sample_trajectory):
        """If a metric column is absent, it should be skipped gracefully."""
        traj = TrajectoryResult(
            player_name="Test", player_id=1,
            df=sample_trajectory.df.drop(columns=["assists_p90"]),
            peak_season_name="2021/22", peak_metric="Goals/90",
            peak_value=0.60, trend_slope=0.0, trend_direction="stable",
        )
        plotter = CareerPlotter()
        fig = plotter.plot(traj)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_improving_trend_shown_in_title(self, sample_trajectory):
        """Trend direction should be embedded in the figure."""
        plotter = CareerPlotter()
        fig = plotter.plot(sample_trajectory)
        # Check the suptitle contains the trend word
        suptitle = fig._suptitle.get_text() if fig._suptitle else ""
        assert "Improving" in suptitle or "improving" in suptitle
        plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# PitchMapper tests
# ─────────────────────────────────────────────────────────────────────────────
class TestPitchMapper:
    def test_pass_map_returns_figure(self, sample_events_df):
        mapper = PitchMapper()
        fig = mapper.pass_map(sample_events_df, "Lionel Messi")
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_shot_map_returns_figure(self, sample_events_df):
        mapper = PitchMapper()
        fig = mapper.shot_map(sample_events_df, "Lionel Messi")
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_pressure_map_returns_figure(self, sample_events_df):
        mapper = PitchMapper()
        fig = mapper.pressure_map(sample_events_df, "Lionel Messi")
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_empty_df_returns_blank_pitch(self):
        mapper = PitchMapper()
        fig = mapper.pass_map(pd.DataFrame(), "Nobody")
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_dispatcher_pass_map(self, sample_events_df):
        mapper = PitchMapper()
        fig = mapper.plot(sample_events_df, "Test", chart_type="pass_map")
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_dispatcher_invalid_type_raises(self, sample_events_df):
        mapper = PitchMapper()
        with pytest.raises(ValueError, match="Unknown chart_type"):
            mapper.plot(sample_events_df, chart_type="magic_map")

    def test_shot_map_with_no_shots(self):
        mapper = PitchMapper()
        df_no_shots = pd.DataFrame({"type_name": ["Pass", "Pass"], "location": [[10, 20], [30, 40]]})
        fig = mapper.shot_map(df_no_shots, "Test")
        assert isinstance(fig, plt.Figure)
        plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# ClusterPlotter tests
# ─────────────────────────────────────────────────────────────────────────────
class TestClusterPlotter:
    def test_plot_returns_figure(self, sample_cluster_result):
        plotter = ClusterPlotter()
        fig = plotter.plot(sample_cluster_result)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_highlight_player_does_not_crash(self, sample_cluster_result):
        plotter = ClusterPlotter()
        fig = plotter.plot(sample_cluster_result, highlight_player="Player 3")
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_highlight_nonexistent_player_does_not_crash(self, sample_cluster_result):
        plotter = ClusterPlotter()
        fig = plotter.plot(sample_cluster_result, highlight_player="Zidane")
        assert isinstance(fig, plt.Figure)
        plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# InjuryPlotter tests
# ─────────────────────────────────────────────────────────────────────────────
class TestInjuryPlotter:
    def test_plot_returns_figure(self, sample_injury_result):
        plotter = InjuryPlotter()
        fig = plotter.plot(sample_injury_result)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_all_positive_deltas_do_not_crash(self, sample_injury_result):
        """All improvements — no negative bars."""
        plotter = InjuryPlotter()
        fig = plotter.plot(sample_injury_result)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_all_negative_deltas_do_not_crash(self, sample_injury_result):
        sample_injury_result.delta = {k: -abs(v) for k, v in sample_injury_result.delta.items()}
        plotter = InjuryPlotter()
        fig = plotter.plot(sample_injury_result)
        assert isinstance(fig, plt.Figure)
        plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# ReportFigure tests
# ─────────────────────────────────────────────────────────────────────────────
class TestReportFigure:
    def test_plot_returns_figure(self, sample_trajectory, sample_profile):
        report = ReportFigure()
        fig = report.plot(sample_trajectory, sample_profile)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_custom_title_accepted(self, sample_trajectory, sample_profile):
        report = ReportFigure()
        fig = report.plot(sample_trajectory, sample_profile, title="Custom Report Title")
        assert isinstance(fig, plt.Figure)
        plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# VisualisationFactory tests
# ─────────────────────────────────────────────────────────────────────────────
class TestVisualisationFactory:
    def test_radar_returns_figure(self, factory, sample_profile):
        fig = factory.radar(sample_profile)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_comparison_returns_figure(self, factory, sample_comparison):
        fig = factory.comparison(sample_comparison)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_career_returns_figure(self, factory, sample_trajectory):
        fig = factory.career(sample_trajectory)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_pass_map_returns_figure(self, factory, sample_events_df):
        fig = factory.pass_map(sample_events_df, "Messi")
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_shot_map_returns_figure(self, factory, sample_events_df):
        fig = factory.shot_map(sample_events_df, "Messi")
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_pressure_map_returns_figure(self, factory, sample_events_df):
        fig = factory.pressure_map(sample_events_df, "Messi")
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_clusters_returns_figure(self, factory, sample_cluster_result):
        fig = factory.clusters(sample_cluster_result)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_injury_returns_figure(self, factory, sample_injury_result):
        fig = factory.injury(sample_injury_result)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_report_returns_figure(self, factory, sample_trajectory, sample_profile):
        fig = factory.report(sample_trajectory, sample_profile)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_unknown_chart_type_raises(self, factory):
        with pytest.raises(ValueError, match="Unknown chart type"):
            factory._get("flying_saucer")

    def test_instances_are_reused(self, factory, sample_profile):
        """Factory should return the same chart instance on repeated calls."""
        _ = factory.radar(sample_profile)
        instance1 = factory._instances.get("radar")
        _ = factory.radar(sample_profile)
        instance2 = factory._instances.get("radar")
        assert instance1 is instance2
        plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# BaseChart save / to_bytes tests
# ─────────────────────────────────────────────────────────────────────────────
class TestBaseChartIO:
    def test_to_bytes_returns_nonempty_bytes(self, sample_profile):
        chart = RadarChart()
        fig = chart.plot(sample_profile)
        data = chart.to_bytes(fig)
        assert isinstance(data, bytes)
        assert len(data) > 1000   # a real PNG is always > 1KB

    def test_save_creates_file(self, sample_profile, tmp_path):
        chart = RadarChart()
        fig = chart.plot(sample_profile)
        out = tmp_path / "test_radar.png"
        result = chart.save(fig, out)
        assert result.exists()
        assert result.stat().st_size > 1000

    def test_save_creates_parent_dirs(self, sample_profile, tmp_path):
        chart = RadarChart()
        fig = chart.plot(sample_profile)
        deep_path = tmp_path / "nested" / "deep" / "chart.png"
        chart.save(fig, deep_path)
        assert deep_path.exists()

    def test_factory_to_bytes(self, factory, sample_profile):
        fig = factory.radar(sample_profile)
        data = factory.to_bytes(fig)
        assert len(data) > 0
