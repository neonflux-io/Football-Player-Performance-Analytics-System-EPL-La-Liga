"""
visualizations.py
─────────────────────────────────────────────────────────────────────────────
Phase 5 — Visualisation Module

Every chart class receives a typed result dataclass from Phase 3/4 and
returns a matplotlib Figure.  No data fetching, no analytics logic here.

Class hierarchy
───────────────
    BaseChart  (ABC)
    ├── RadarChart      – single-player pizza / comparison spider
    ├── CareerPlotter   – season trend lines + peak annotation
    └── PitchMapper     – pass maps, shot maps, pressure heatmaps

    ReportFigure        – one-page summary card (composes the three above)
    VisualisationFactory– Factory: create any chart by string name

OOP concepts
────────────
    • Template Method   – BaseChart._setup_figure() / .save() used by all
    • Factory Pattern   – VisualisationFactory.make() decouples callers
                          from concrete chart classes
    • Composition       – ReportFigure holds three chart objects
    • Open/Closed       – add a new chart type = register in CHART_REGISTRY,
                          zero changes to existing classes
    • Encapsulation     – colour palettes and style constants are private
                          class-level dicts, not global variables
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

import matplotlib
matplotlib.use("Agg")          # non-interactive: safe for server / Streamlit

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from mplsoccer import Pitch, Radar, VerticalPitch

from src.analyzer import ComparisonResult, PercentileProfile, TrajectoryResult
from src.advanced_analysis import ClusterResult, InjuryImpactResult
from src.config import cfg
from src.utils import get_logger

log = get_logger()
warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Style constants (dark theme)
# ─────────────────────────────────────────────────────────────────────────────
_DARK_BG       = "#0d1117"
_PANEL_BG      = "#161b22"
_ACCENT_GREEN  = "#39d353"
_ACCENT_BLUE   = "#58a6ff"
_ACCENT_ORANGE = "#f78166"
_ACCENT_PURPLE = "#bc8cff"
_TEXT_PRIMARY  = "#e6edf3"
_TEXT_MUTED    = "#8b949e"
_GRID_COLOR    = "#21262d"

# Matplotlib rcParams applied globally when this module is imported
plt.rcParams.update({
    "figure.facecolor":  _DARK_BG,
    "axes.facecolor":    _PANEL_BG,
    "axes.edgecolor":    _GRID_COLOR,
    "axes.labelcolor":   _TEXT_PRIMARY,
    "text.color":        _TEXT_PRIMARY,
    "xtick.color":       _TEXT_MUTED,
    "ytick.color":       _TEXT_MUTED,
    "grid.color":        _GRID_COLOR,
    "grid.linewidth":    0.5,
    "font.family":       "DejaVu Sans",
    "figure.dpi":        cfg.FIGURE_DPI,
})


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base: BaseChart
# ─────────────────────────────────────────────────────────────────────────────
class BaseChart(ABC):
    """
    Abstract base for all chart classes.

    TEMPLATE METHOD PATTERN:
    ─────────────────────────
    Defines the skeleton:  _setup_figure() → plot() → save()
    Subclasses only implement plot() — the infrastructure is inherited.

    This guarantees consistent figure sizing, DPI, background colour,
    and save logic across every chart type.
    """

    _DEFAULT_FIGSIZE: ClassVar[Tuple[float, float]] = (12, 8)

    def __init__(self, figsize: Optional[Tuple[float, float]] = None) -> None:
        self._figsize = figsize or self._DEFAULT_FIGSIZE
        self._fig: Optional[Figure] = None

    def _setup_figure(
        self,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> Figure:
        """Create and style a new Figure. Called at the start of plot()."""
        fig = plt.figure(
            figsize=figsize or self._figsize,
            facecolor=_DARK_BG,
        )
        return fig

    @abstractmethod
    def plot(self, *args, **kwargs) -> Figure:
        """
        Render the chart and return the Figure.

        Each subclass implements its own rendering logic here.
        """
        ...

    def save(self, fig: Figure, path: Path) -> Path:
        """
        Save *fig* to *path*.  Creates parent directories automatically.

        Args:
            fig:  Matplotlib Figure to save.
            path: Destination file path (extension determines format).

        Returns:
            The resolved file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path,
            dpi=cfg.FIGURE_DPI,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)
        log.info(f"Chart saved → {path}")
        return path

    def to_bytes(self, fig: Figure, fmt: str = "png") -> bytes:
        """
        Serialise *fig* to bytes in *fmt* format (for Streamlit st.image).

        Args:
            fig: Matplotlib Figure.
            fmt: Image format string ("png", "svg", etc.).

        Returns:
            Raw image bytes.
        """
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, dpi=cfg.FIGURE_DPI,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# 1. RadarChart
# ─────────────────────────────────────────────────────────────────────────────
class RadarChart(BaseChart):
    """
    Spider / radar chart showing a player's percentile profile.

    Supports two modes:
      • Single player  – filled polygon with percentile labels
      • Comparison     – two overlapping polygons in different colours

    Uses mplsoccer's Radar class which handles the trigonometry, axis labels,
    and range normalisation automatically.
    """

    _DEFAULT_FIGSIZE = (10, 10)

    # Colour pairs for comparison mode: (fill_colour, edge_colour)
    _PLAYER_COLOURS: ClassVar[List[Tuple[str, str]]] = [
        (_ACCENT_GREEN,  "#1a7a32"),
        (_ACCENT_BLUE,   "#1a4a7a"),
        (_ACCENT_ORANGE, "#7a2a1a"),
        (_ACCENT_PURPLE, "#4a1a7a"),
    ]

    def plot(
        self,
        profile: PercentileProfile,
        title: Optional[str] = None,
    ) -> Figure:
        """
        Render a single-player radar from a PercentileProfile.

        Args:
            profile: PercentileProfile result from Phase 3 analyzer.
            title:   Optional override title.

        Returns:
            Matplotlib Figure.
        """
        metrics = list(profile.percentiles.keys())
        values  = [profile.percentiles[m] for m in metrics]

        if not metrics:
            log.warning("RadarChart.plot: no metrics in profile — returning blank figure")
            return self._setup_figure()

        fig = self._draw_radar(
            metrics_list=[metrics],
            values_list=[values],
            player_names=[profile.player_name],
            title=title or f"{profile.player_name}  ·  Percentile Profile",
            subtitle=f"{profile.position_group.title()} | vs {profile.n_peers} peers",
        )
        return fig

    def plot_comparison(
        self,
        result: ComparisonResult,
        title: Optional[str] = None,
    ) -> Figure:
        """
        Render a two-player comparison radar from a ComparisonResult.

        Args:
            result: ComparisonResult from Phase 3 analyzer.
            title:  Optional override title.

        Returns:
            Matplotlib Figure.
        """
        metrics = list(result.percentiles_a.keys())
        values_a = [result.percentiles_a[m] for m in metrics]
        values_b = [result.percentiles_b[m] for m in metrics]

        fig = self._draw_radar(
            metrics_list=[metrics, metrics],
            values_list=[values_a, values_b],
            player_names=[result.player_a_name, result.player_b_name],
            title=title or f"{result.player_a_name}  vs  {result.player_b_name}",
            subtitle="Percentile rank vs. position peers",
        )
        return fig

    def _draw_radar(
        self,
        metrics_list: List[List[str]],
        values_list: List[List[float]],
        player_names: List[str],
        title: str,
        subtitle: str = "",
    ) -> Figure:
        """
        Internal: draw one or two overlapping radar polygons.

        Args:
            metrics_list: List of metric label lists (one per player).
            values_list:  Corresponding percentile values (0–100).
            player_names: Player name labels.
            title:        Figure title.
            subtitle:     Smaller sub-title line.
        """
        metrics = metrics_list[0]
        n = len(metrics)

        # mplsoccer Radar expects ranges for each spoke
        low  = [0]  * n
        high = [100] * n

        radar = Radar(
            params=metrics,
            min_range=low,
            max_range=high,
            num_rings=4,
            ring_width=1,
            center_circle_radius=1,
        )

        fig, axes = radar.setup_axis()
        fig.set_facecolor(_DARK_BG)
        axes.set_facecolor(_DARK_BG)

        # Draw rings and spokes
        radar.draw_circles(
            ax=axes,
            facecolor=_PANEL_BG,
            edgecolor=_GRID_COLOR,
            lw=0.8,
            zorder=2,
        )

        # Draw each player's polygon
        for i, (vals, name) in enumerate(zip(values_list, player_names)):
            colour, edge_colour = self._PLAYER_COLOURS[i % len(self._PLAYER_COLOURS)]
            radar.draw_radar_solid(
                vals,
                ax=axes,
                kwargs={
                    "facecolor": colour,
                    "alpha": 0.35 if len(values_list) > 1 else 0.45,
                    "edgecolor": colour,
                    "lw": 2,
                    "zorder": 3 + i,
                },
            )
            # Outer edge line (no fill) for crispness
            radar.draw_radar_solid(
                vals,
                ax=axes,
                kwargs={
                    "facecolor": "none",
                    "edgecolor": colour,
                    "lw": 2.5,
                    "zorder": 5 + i,
                },
            )

        # Spoke labels
        radar.draw_range_labels(
            ax=axes,
            fontsize=8,
            color=_TEXT_MUTED,
            zorder=7,
        )
        radar.draw_param_labels(
            ax=axes,
            fontsize=10,
            color=_TEXT_PRIMARY,
            fontweight="bold",
            zorder=7,
        )

        # ── Title block ───────────────────────────────────────────────────────
        fig.text(
            0.5, 0.97, title,
            ha="center", va="top",
            fontsize=16, fontweight="bold",
            color=_TEXT_PRIMARY,
        )
        if subtitle:
            fig.text(
                0.5, 0.93, subtitle,
                ha="center", va="top",
                fontsize=11, color=_TEXT_MUTED,
            )

        # ── Legend ────────────────────────────────────────────────────────────
        if len(player_names) > 1:
            patches = [
                mpatches.Patch(
                    color=self._PLAYER_COLOURS[i][0],
                    label=name,
                    alpha=0.8,
                )
                for i, name in enumerate(player_names)
            ]
            axes.legend(
                handles=patches,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.08),
                fontsize=11,
                frameon=False,
                labelcolor=_TEXT_PRIMARY,
                ncol=len(player_names),
            )

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. CareerPlotter
# ─────────────────────────────────────────────────────────────────────────────
class CareerPlotter(BaseChart):
    """
    Season-by-season trend line chart with peak annotation.

    Renders up to four metrics on a 2×2 subplot grid, each with:
      • Line + scatter (per-90 value per season)
      • Shaded area under the line
      • Annotated peak season (gold star marker + label)
      • Linear trend line (dashed)
    """

    _DEFAULT_FIGSIZE = (14, 10)

    # Default metrics shown on the 2×2 grid
    _DEFAULT_METRICS: ClassVar[List[Tuple[str, str]]] = [
        ("goals_p90",       "Goals / 90"),
        ("xg_p90",          "xG / 90"),
        ("assists_p90",     "Assists / 90"),
        ("key_passes_p90",  "Key Passes / 90"),
    ]

    def plot(
        self,
        trajectory: TrajectoryResult,
        metrics: Optional[List[Tuple[str, str]]] = None,
    ) -> Figure:
        """
        Render career trajectory chart from a TrajectoryResult.

        Args:
            trajectory: TrajectoryResult from PerformanceAnalyzer.career_trajectory().
            metrics:    List of (column_name, display_label) pairs to plot.
                        Defaults to _DEFAULT_METRICS.

        Returns:
            Matplotlib Figure.
        """
        df = trajectory.df
        metrics = metrics or self._DEFAULT_METRICS

        # Filter to metrics actually present in the DataFrame
        available = [(col, lbl) for col, lbl in metrics if col in df.columns]
        if not available:
            log.warning("CareerPlotter: no plottable metrics found in trajectory df")
            return self._setup_figure()

        n_plots = len(available)
        ncols = min(n_plots, 2)
        nrows = (n_plots + 1) // 2

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(self._figsize[0], max(6, nrows * 4.5)),
            facecolor=_DARK_BG,
        )
        axes = np.array(axes).flatten()

        seasons = df["season_name"].tolist() if "season_name" in df.columns else list(range(len(df)))
        x = np.arange(len(df))

        for ax_idx, (col, label) in enumerate(available):
            ax = axes[ax_idx]
            ax.set_facecolor(_PANEL_BG)
            y = df[col].fillna(0).values.astype(float)

            # ── Shaded area ───────────────────────────────────────────────────
            ax.fill_between(x, y, alpha=0.18, color=_ACCENT_GREEN, zorder=1)

            # ── Main line + scatter ───────────────────────────────────────────
            ax.plot(x, y, color=_ACCENT_GREEN, lw=2.5, zorder=3, solid_capstyle="round")
            ax.scatter(x, y, color=_ACCENT_GREEN, s=60, zorder=4, edgecolors=_DARK_BG, lw=1.5)

            # ── Trend line (dashed) ───────────────────────────────────────────
            if len(x) >= 2:
                from numpy.polynomial.polynomial import polyfit as npfit
                coeffs = np.polyfit(x, y, 1)
                trend_y = np.polyval(coeffs, x)
                ax.plot(x, trend_y, color=_TEXT_MUTED, lw=1.2, ls="--",
                        alpha=0.7, zorder=2, label="Trend")

            # ── Peak annotation ───────────────────────────────────────────────
            if label == trajectory.peak_metric or col.startswith("goals"):
                peak_y_idx = int(np.argmax(y))
                peak_val   = y[peak_y_idx]
                ax.scatter(
                    [x[peak_y_idx]], [peak_val],
                    color="#FFD700", s=140, zorder=5,
                    marker="*", edgecolors=_DARK_BG, lw=1,
                )
                ax.annotate(
                    f"Peak\n{peak_val:.2f}",
                    xy=(x[peak_y_idx], peak_val),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=8, color="#FFD700", fontweight="bold",
                    va="bottom",
                )

            # ── Value labels on each data point ───────────────────────────────
            for xi, yi in zip(x, y):
                ax.annotate(
                    f"{yi:.2f}",
                    xy=(xi, yi),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=8, color=_TEXT_MUTED,
                )

            # ── Axes styling ──────────────────────────────────────────────────
            ax.set_xticks(x)
            ax.set_xticklabels(
                seasons, rotation=30, ha="right", fontsize=9, color=_TEXT_MUTED
            )
            ax.set_ylabel(label, color=_TEXT_PRIMARY, fontsize=10)
            ax.set_title(label, color=_TEXT_PRIMARY, fontsize=11, pad=8, fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color(_GRID_COLOR)
            ax.grid(axis="y", color=_GRID_COLOR, lw=0.5, alpha=0.8)
            ax.set_xlim(-0.4, len(x) - 0.6)
            y_margin = max(y.max() * 0.15, 0.05)
            ax.set_ylim(0, y.max() + y_margin)

        # Hide any unused axes in the grid
        for ax in axes[n_plots:]:
            ax.set_visible(False)

        # ── Figure title ──────────────────────────────────────────────────────
        trend_arrow = "↗" if trajectory.trend_direction == "improving" else \
                      "↘" if trajectory.trend_direction == "declining" else "→"
        title = (
            f"{trajectory.player_name}  ·  Career Trajectory  "
            f"{trend_arrow} {trajectory.trend_direction.title()}"
        )
        fig.suptitle(title, fontsize=15, fontweight="bold",
                     color=_TEXT_PRIMARY, y=1.01)

        fig.tight_layout(rect=[0, 0, 1, 0.98])
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. PitchMapper
# ─────────────────────────────────────────────────────────────────────────────
class PitchMapper(BaseChart):
    """
    Renders on-pitch visualisations using mplsoccer's Pitch class.

    Three chart types:
      • pass_map      – arrows from pass origin to destination
      • shot_map      – bubbles scaled by xG at shot location
      • pressure_map  – heatmap of pressure event density

    Each method accepts a raw events DataFrame (from StatsBombLoader.events())
    filtered to one player, so it can be called directly from the dashboard.
    """

    _DEFAULT_FIGSIZE = (12, 8)

    # Pitch orientation constants
    _PITCH_TYPE = "statsbomb"   # StatsBomb uses 120×80 yard coordinates

    def plot(self, events_df: pd.DataFrame, player_name: str = "", chart_type: str = "pass_map") -> Figure:
        """
        Generic dispatcher. Calls pass_map, shot_map, or pressure_map.

        Args:
            events_df:   Events DataFrame from StatsBombLoader (one match or season).
            player_name: Used in the figure title.
            chart_type:  "pass_map" | "shot_map" | "pressure_map"

        Returns:
            Matplotlib Figure.
        """
        dispatch = {
            "pass_map":     self.pass_map,
            "shot_map":     self.shot_map,
            "pressure_map": self.pressure_map,
        }
        if chart_type not in dispatch:
            raise ValueError(f"Unknown chart_type '{chart_type}'. Choose from {list(dispatch)}")
        return dispatch[chart_type](events_df, player_name)

    # ── Pass map ──────────────────────────────────────────────────────────────
    def pass_map(
        self,
        events_df: pd.DataFrame,
        player_name: str = "",
    ) -> Figure:
        """
        Arrow map of all passes for a player / match.

        Successful passes are green arrows; failed passes are faded red.

        Args:
            events_df:   Events DataFrame (should contain 'type_name' == 'Pass').
            player_name: Label for the figure title.

        Returns:
            Matplotlib Figure.
        """
        passes = events_df[events_df.get("type_name", pd.Series()) == "Pass"].copy()

        if passes.empty:
            log.warning("pass_map: no pass events found in DataFrame")
            return self._blank_pitch(f"{player_name} — No pass data available")

        # Extract locations
        loc_x, loc_y, end_x, end_y, success = [], [], [], [], []

        for _, row in passes.iterrows():
            # StatsBomb stores location as [x, y] in 'location' column
            loc = row.get("location")
            end = row.get("pass_end_location")

            if isinstance(loc, list) and len(loc) >= 2:
                loc_x.append(float(loc[0]))
                loc_y.append(float(loc[1]))
            elif pd.notna(row.get("location_x")) and pd.notna(row.get("location_y")):
                loc_x.append(float(row["location_x"]))
                loc_y.append(float(row["location_y"]))
            else:
                continue

            if isinstance(end, list) and len(end) >= 2:
                end_x.append(float(end[0]))
                end_y.append(float(end[1]))
            elif pd.notna(row.get("pass_end_location_x")):
                end_x.append(float(row["pass_end_location_x"]))
                end_y.append(float(row.get("pass_end_location_y", loc_y[-1])))
            else:
                end_x.append(loc_x[-1] + 5)   # fallback: short forward pass
                end_y.append(loc_y[-1])

            outcome = row.get("pass_outcome_name", None)
            success.append(outcome is None or pd.isna(outcome))  # None = complete

        if not loc_x:
            return self._blank_pitch(f"{player_name} — Could not parse pass locations")

        fig, ax = plt.subplots(figsize=self._figsize, facecolor=_DARK_BG)
        pitch = Pitch(
            pitch_type=self._PITCH_TYPE,
            pitch_color=_PANEL_BG,
            line_color=_GRID_COLOR,
            linewidth=1.5,
        )
        pitch.draw(ax=ax)

        success_arr = np.array(success)

        # Complete passes — green
        if success_arr.any():
            pitch.arrows(
                np.array(loc_x)[success_arr],
                np.array(loc_y)[success_arr],
                np.array(end_x)[success_arr],
                np.array(end_y)[success_arr],
                ax=ax,
                color=_ACCENT_GREEN,
                alpha=0.55,
                width=1.5,
                headwidth=4,
                headlength=4,
                zorder=3,
            )

        # Incomplete passes — orange/red
        if (~success_arr).any():
            pitch.arrows(
                np.array(loc_x)[~success_arr],
                np.array(loc_y)[~success_arr],
                np.array(end_x)[~success_arr],
                np.array(end_y)[~success_arr],
                ax=ax,
                color=_ACCENT_ORANGE,
                alpha=0.35,
                width=1.2,
                headwidth=3,
                headlength=3,
                zorder=2,
            )

        # ── Legend ────────────────────────────────────────────────────────────
        ax.legend(
            handles=[
                mpatches.Patch(color=_ACCENT_GREEN,  label=f"Complete ({success_arr.sum()})"),
                mpatches.Patch(color=_ACCENT_ORANGE, label=f"Incomplete ({(~success_arr).sum()})"),
            ],
            loc="lower left", fontsize=9, frameon=False, labelcolor=_TEXT_PRIMARY,
        )

        title = f"{player_name}  ·  Pass Map" if player_name else "Pass Map"
        n_passes = len(loc_x)
        completion = round(success_arr.mean() * 100, 1) if n_passes else 0
        ax.set_title(
            f"{title}\n{n_passes} passes  ·  {completion}% completion",
            color=_TEXT_PRIMARY, fontsize=12, fontweight="bold", pad=12,
        )
        fig.tight_layout()
        return fig

    # ── Shot map ──────────────────────────────────────────────────────────────
    def shot_map(
        self,
        events_df: pd.DataFrame,
        player_name: str = "",
    ) -> Figure:
        """
        Bubble map of shots scaled and coloured by xG value.

        Goals have a star marker; non-goals have a circle.
        Larger bubble = higher xG.

        Args:
            events_df:   Events DataFrame containing shot events.
            player_name: Label for the figure title.

        Returns:
            Matplotlib Figure (vertical half-pitch).
        """
        shots = events_df[events_df.get("type_name", pd.Series()) == "Shot"].copy()

        if shots.empty:
            log.warning("shot_map: no shot events found")
            return self._blank_pitch(f"{player_name} — No shot data available")

        fig, ax = plt.subplots(figsize=(10, 8), facecolor=_DARK_BG)
        pitch = VerticalPitch(
            pitch_type=self._PITCH_TYPE,
            pitch_color=_PANEL_BG,
            line_color=_GRID_COLOR,
            half=True,
            linewidth=1.5,
        )
        pitch.draw(ax=ax)

        # Parse locations
        shot_x, shot_y, xg_vals, is_goal = [], [], [], []

        for _, row in shots.iterrows():
            loc = row.get("location")
            if isinstance(loc, list) and len(loc) >= 2:
                shot_x.append(float(loc[0]))
                shot_y.append(float(loc[1]))
            elif pd.notna(row.get("location_x")):
                shot_x.append(float(row["location_x"]))
                shot_y.append(float(row.get("location_y", 40)))
            else:
                continue

            xg = row.get("shot_statsbomb_xg", 0.05)
            xg_vals.append(float(xg) if pd.notna(xg) else 0.05)
            outcome = str(row.get("shot_outcome_name", ""))
            is_goal.append("Goal" in outcome)

        if not shot_x:
            return self._blank_pitch(f"{player_name} — Could not parse shot locations")

        shot_x   = np.array(shot_x)
        shot_y   = np.array(shot_y)
        xg_vals  = np.array(xg_vals)
        is_goal  = np.array(is_goal)

        # Bubble size: scale xG to dot area
        sizes = (xg_vals * 1200).clip(min=80)

        # Non-goal shots
        if (~is_goal).any():
            ax.scatter(
                shot_y[~is_goal], shot_x[~is_goal],   # mplsoccer vert: y, x
                s=sizes[~is_goal],
                c=_ACCENT_BLUE, alpha=0.7, edgecolors=_TEXT_PRIMARY, lw=0.8,
                zorder=3, label="Shot",
            )
        # Goals
        if is_goal.any():
            ax.scatter(
                shot_y[is_goal], shot_x[is_goal],
                s=sizes[is_goal] * 1.4,
                c=_ACCENT_GREEN, alpha=0.95, edgecolors="#FFD700", lw=1.5,
                marker="*", zorder=5, label="Goal",
            )

        # xG sum annotation
        total_xg = xg_vals.sum()
        goals    = is_goal.sum()
        ax.text(
            0.02, 0.97,
            f"Goals: {goals}   xG: {total_xg:.2f}   xG overpf: {goals - total_xg:+.2f}",
            transform=ax.transAxes, color=_TEXT_PRIMARY, fontsize=9,
            va="top", ha="left", style="italic",
        )

        ax.legend(
            loc="lower left", fontsize=9, frameon=False, labelcolor=_TEXT_PRIMARY,
        )
        title = f"{player_name}  ·  Shot Map" if player_name else "Shot Map"
        ax.set_title(
            f"{title}\n{len(shot_x)} shots  ·  Bubble size = xG",
            color=_TEXT_PRIMARY, fontsize=12, fontweight="bold", pad=12,
        )
        fig.tight_layout()
        return fig

    # ── Pressure heatmap ──────────────────────────────────────────────────────
    def pressure_map(
        self,
        events_df: pd.DataFrame,
        player_name: str = "",
    ) -> Figure:
        """
        Gaussian KDE heatmap of pressure event locations.

        Hot zones show where on the pitch the player applied the most pressure.

        Args:
            events_df:   Events DataFrame containing pressure events.
            player_name: Label for the figure title.

        Returns:
            Matplotlib Figure.
        """
        pressures = events_df[
            events_df.get("type_name", pd.Series()) == "Pressure"
        ].copy()

        if pressures.empty:
            log.warning("pressure_map: no pressure events found")
            return self._blank_pitch(f"{player_name} — No pressure data available")

        fig, ax = plt.subplots(figsize=self._figsize, facecolor=_DARK_BG)
        pitch = Pitch(
            pitch_type=self._PITCH_TYPE,
            pitch_color=_PANEL_BG,
            line_color=_GRID_COLOR,
            linewidth=1.5,
        )
        pitch.draw(ax=ax)

        # Parse locations
        px, py = [], []
        for _, row in pressures.iterrows():
            loc = row.get("location")
            if isinstance(loc, list) and len(loc) >= 2:
                px.append(float(loc[0]))
                py.append(float(loc[1]))
            elif pd.notna(row.get("location_x")):
                px.append(float(row["location_x"]))
                py.append(float(row.get("location_y", 40)))

        if not px:
            return self._blank_pitch(f"{player_name} — Could not parse pressure locations")

        # Kernel density heatmap via mplsoccer
        bin_statistic = pitch.bin_statistic(
            np.array(px), np.array(py),
            statistic="count", bins=(24, 16),
        )
        pitch.heatmap(
            bin_statistic, ax=ax,
            cmap="Greens",
            edgecolors=_PANEL_BG,
            linewidth=0.3,
            alpha=0.85,
        )

        # Scatter dots on top for individual events
        ax.scatter(px, py, color=_ACCENT_GREEN, s=12, alpha=0.25, zorder=3)

        title = f"{player_name}  ·  Pressure Heatmap" if player_name else "Pressure Heatmap"
        ax.set_title(
            f"{title}\n{len(px)} pressure events",
            color=_TEXT_PRIMARY, fontsize=12, fontweight="bold", pad=12,
        )
        fig.tight_layout()
        return fig

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _blank_pitch(self, message: str) -> Figure:
        """Return a blank styled pitch with a centred message."""
        fig, ax = plt.subplots(figsize=self._figsize, facecolor=_DARK_BG)
        pitch = Pitch(
            pitch_type=self._PITCH_TYPE,
            pitch_color=_PANEL_BG,
            line_color=_GRID_COLOR,
        )
        pitch.draw(ax=ax)
        ax.text(
            60, 40, message,
            ha="center", va="center", fontsize=13,
            color=_TEXT_MUTED, style="italic",
        )
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. ClusterPlotter
# ─────────────────────────────────────────────────────────────────────────────
class ClusterPlotter(BaseChart):
    """
    Visualises the PositionalClusterer output from Phase 4.

    Two charts:
      • scatter_plot  – 2-D PCA projection coloured by cluster
      • profile_bars  – grouped bar chart showing centroid metric values
    """

    _DEFAULT_FIGSIZE = (14, 6)

    def plot(self, cluster_result: ClusterResult, highlight_player: Optional[str] = None) -> Figure:
        """
        Two-panel figure: PCA scatter (left) + profile bars (right).

        Args:
            cluster_result:   ClusterResult from Phase 4 PositionalClusterer.
            highlight_player: Optional player name to highlight in the scatter.

        Returns:
            Matplotlib Figure.
        """
        fig = plt.figure(figsize=self._figsize, facecolor=_DARK_BG)
        gs  = GridSpec(1, 2, figure=fig, wspace=0.35)

        ax_scatter = fig.add_subplot(gs[0, 0])
        ax_bars    = fig.add_subplot(gs[0, 1])

        self._draw_pca_scatter(ax_scatter, cluster_result, highlight_player)
        self._draw_profile_bars(ax_bars, cluster_result)

        fig.suptitle(
            f"Player Archetype Clustering  ·  {cluster_result.n_clusters} groups  "
            f"·  Silhouette: {cluster_result.silhouette:.2f}",
            fontsize=14, fontweight="bold", color=_TEXT_PRIMARY, y=1.01,
        )
        fig.tight_layout()
        return fig

    def _draw_pca_scatter(
        self,
        ax: plt.Axes,
        result: ClusterResult,
        highlight_player: Optional[str],
    ) -> None:
        """Reduce metrics to 2D via PCA and scatter-plot by cluster colour."""
        from sklearn.decomposition import PCA

        df = result.cluster_df
        metric_cols = [c for c in df.columns
                       if c not in ("player_id", "player_name", "position",
                                    "minutes_played", "cluster_id", "archetype")]

        if not metric_cols or len(df) < 2:
            ax.text(0.5, 0.5, "Not enough data for PCA",
                    ha="center", va="center", color=_TEXT_MUTED, transform=ax.transAxes)
            return

        X = df[metric_cols].fillna(0).values
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X)

        colors = [_ACCENT_GREEN, _ACCENT_BLUE, _ACCENT_ORANGE, _ACCENT_PURPLE,
                  "#ff9500", "#ff2d55", "#5ac8fa", "#4cd964"]

        ax.set_facecolor(_PANEL_BG)
        ax.spines[["top","right"]].set_visible(False)
        ax.spines[["left","bottom"]].set_color(_GRID_COLOR)

        for cid in sorted(df["cluster_id"].unique()):
            mask = df["cluster_id"] == cid
            label = result.labels.get(int(cid), f"Cluster {cid}")
            colour = colors[cid % len(colors)]
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=colour, s=55, alpha=0.75, edgecolors=_DARK_BG, lw=0.5,
                label=label, zorder=3,
            )

        # Highlight specific player
        if highlight_player:
            mask = df["player_name"].str.lower().str.contains(highlight_player.lower(), na=False)
            if mask.any():
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c="#FFD700", s=180, marker="*", zorder=6,
                    edgecolors=_DARK_BG, lw=1, label=highlight_player,
                )

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                      color=_TEXT_MUTED, fontsize=9)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                      color=_TEXT_MUTED, fontsize=9)
        ax.set_title("Style Archetypes (PCA)", color=_TEXT_PRIMARY,
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, frameon=False, labelcolor=_TEXT_PRIMARY,
                  loc="best", markerscale=0.8)
        ax.grid(True, color=_GRID_COLOR, lw=0.4, alpha=0.6)

    def _draw_profile_bars(self, ax: plt.Axes, result: ClusterResult) -> None:
        """Grouped horizontal bars showing average metric values per cluster."""
        profiles = result.profiles.copy()
        metric_cols = [c for c in profiles.columns
                       if c not in ("cluster_id", "archetype", "n_players")]

        if not metric_cols:
            return

        # Normalise to 0–1 within each metric for visual comparability
        profiles_norm = profiles[metric_cols].copy()
        for col in metric_cols:
            col_max = profiles_norm[col].max()
            if col_max > 0:
                profiles_norm[col] = profiles_norm[col] / col_max

        n_clusters = len(profiles)
        n_metrics  = len(metric_cols)
        bar_h      = 0.8 / n_clusters
        y_base     = np.arange(n_metrics)

        colors = [_ACCENT_GREEN, _ACCENT_BLUE, _ACCENT_ORANGE, _ACCENT_PURPLE,
                  "#ff9500", "#ff2d55", "#5ac8fa", "#4cd964"]

        ax.set_facecolor(_PANEL_BG)
        ax.spines[["top","right"]].set_visible(False)
        ax.spines[["left","bottom"]].set_color(_GRID_COLOR)

        for i, (_, row) in enumerate(profiles.iterrows()):
            norm_row = profiles_norm.iloc[i]
            offset   = (i - n_clusters / 2) * bar_h + bar_h / 2
            colour   = colors[i % len(colors)]
            label    = result.labels.get(int(row["cluster_id"]), f"Cluster {i}")

            ax.barh(
                y_base + offset,
                norm_row[metric_cols].values,
                height=bar_h * 0.85,
                color=colour, alpha=0.8,
                label=f"{label} (n={int(row.get('n_players',0))})",
            )

        ax.set_yticks(y_base)
        ax.set_yticklabels(
            [m.replace("/90", "\n/90").replace(" ", "\n", 1) for m in metric_cols],
            fontsize=8, color=_TEXT_PRIMARY,
        )
        ax.set_xlabel("Normalised value (0–1)", color=_TEXT_MUTED, fontsize=9)
        ax.set_title("Cluster Metric Profiles", color=_TEXT_PRIMARY,
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, frameon=False, labelcolor=_TEXT_PRIMARY,
                  loc="lower right")
        ax.grid(axis="x", color=_GRID_COLOR, lw=0.4, alpha=0.6)


# ─────────────────────────────────────────────────────────────────────────────
# 5. InjuryPlotter
# ─────────────────────────────────────────────────────────────────────────────
class InjuryPlotter(BaseChart):
    """
    Visualises pre/post absence metric comparison from Phase 4 InjuryImpactResult.

    Two panels:
      • Bar chart comparing pre vs. post per-90 metrics
      • Delta (change) chart highlighting improvements / regressions
    """

    _DEFAULT_FIGSIZE = (14, 6)

    def plot(self, result: InjuryImpactResult) -> Figure:  # type: ignore[override]
        """
        Render pre/post injury comparison figure.

        Args:
            result: InjuryImpactResult from Phase 4.

        Returns:
            Matplotlib Figure.
        """
        metrics  = list(result.pre_metrics.keys())
        pre_vals = [result.pre_metrics[m]  for m in metrics]
        post_vals= [result.post_metrics[m] for m in metrics]
        deltas   = [result.delta[m]        for m in metrics]

        fig = plt.figure(figsize=self._figsize, facecolor=_DARK_BG)
        gs  = GridSpec(1, 2, figure=fig, wspace=0.4)
        ax_bars  = fig.add_subplot(gs[0, 0])
        ax_delta = fig.add_subplot(gs[0, 1])

        x = np.arange(len(metrics))
        width = 0.38

        for ax in [ax_bars, ax_delta]:
            ax.set_facecolor(_PANEL_BG)
            ax.spines[["top","right"]].set_visible(False)
            ax.spines[["left","bottom"]].set_color(_GRID_COLOR)

        # ── Bar chart: pre vs post ────────────────────────────────────────────
        ax_bars.bar(x - width/2, pre_vals,  width, label="Pre-absence",
                    color=_ACCENT_BLUE,   alpha=0.85, edgecolor=_DARK_BG)
        ax_bars.bar(x + width/2, post_vals, width, label="Post-return",
                    color=_ACCENT_GREEN,  alpha=0.85, edgecolor=_DARK_BG)

        ax_bars.set_xticks(x)
        ax_bars.set_xticklabels(metrics, rotation=30, ha="right",
                                fontsize=9, color=_TEXT_PRIMARY)
        ax_bars.set_ylabel("Per 90 min", color=_TEXT_MUTED, fontsize=9)
        ax_bars.set_title("Performance: Pre vs Post Absence",
                          color=_TEXT_PRIMARY, fontsize=11, fontweight="bold")
        ax_bars.legend(fontsize=9, frameon=False, labelcolor=_TEXT_PRIMARY)
        ax_bars.grid(axis="y", color=_GRID_COLOR, lw=0.4, alpha=0.6)

        # ── Delta chart ───────────────────────────────────────────────────────
        delta_colors = [_ACCENT_GREEN if d >= 0 else _ACCENT_ORANGE for d in deltas]
        bars = ax_delta.bar(x, deltas, color=delta_colors, alpha=0.85,
                            edgecolor=_DARK_BG, width=0.55)

        # Value labels
        for bar, d in zip(bars, deltas):
            ypos = bar.get_height() + 0.005 if d >= 0 else bar.get_height() - 0.015
            ax_delta.text(bar.get_x() + bar.get_width()/2, ypos,
                          f"{d:+.3f}", ha="center", fontsize=8,
                          color=_ACCENT_GREEN if d >= 0 else _ACCENT_ORANGE,
                          fontweight="bold")

        ax_delta.axhline(0, color=_TEXT_MUTED, lw=1, ls="--", alpha=0.6)
        ax_delta.set_xticks(x)
        ax_delta.set_xticklabels(metrics, rotation=30, ha="right",
                                 fontsize=9, color=_TEXT_PRIMARY)
        ax_delta.set_ylabel("Post − Pre  (per 90)", color=_TEXT_MUTED, fontsize=9)
        ax_delta.set_title("Metric Changes After Return",
                           color=_TEXT_PRIMARY, fontsize=11, fontweight="bold")
        ax_delta.grid(axis="y", color=_GRID_COLOR, lw=0.4, alpha=0.6)

        # ── Figure title ──────────────────────────────────────────────────────
        fig.suptitle(
            f"{result.player_name}  ·  Injury Impact Analysis\n"
            f"Affected season: {result.affected_season_name}  "
            f"·  Recovery score: {result.recovery_score:.2f}",
            fontsize=13, fontweight="bold", color=_TEXT_PRIMARY, y=1.03,
        )
        fig.tight_layout()
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. ReportFigure  (Composition)
# ─────────────────────────────────────────────────────────────────────────────
class ReportFigure(BaseChart):
    """
    One-page player summary card combining radar + career trend + stat table.

    COMPOSITION: ReportFigure owns instances of RadarChart and CareerPlotter.
    It delegates rendering to them, then assembles the results into a single
    figure using GridSpec sub-figure embedding.
    """

    _DEFAULT_FIGSIZE = (18, 12)

    def __init__(self) -> None:
        super().__init__()
        # Compose sub-chart objects (COMPOSITION)
        self._radar   = RadarChart()
        self._plotter = CareerPlotter()

    def plot(  # type: ignore[override]
        self,
        trajectory: TrajectoryResult,
        profile: PercentileProfile,
        title: Optional[str] = None,
    ) -> Figure:
        """
        Render a one-page report card.

        Layout:
          ┌──────────────────────────────────┐
          │  Player name + identity header    │
          ├──────────────┬───────────────────┤
          │  Radar chart │  Career trend     │
          │  (left)      │  (right, 2 rows)  │
          ├──────────────┴───────────────────┤
          │  Stat summary table              │
          └──────────────────────────────────┘

        Args:
            trajectory: From PerformanceAnalyzer.career_trajectory().
            profile:    From PerformanceAnalyzer.player_percentile_profile().
            title:      Optional override for the header.

        Returns:
            Matplotlib Figure.
        """
        fig = plt.figure(figsize=self._figsize, facecolor=_DARK_BG)

        # ── Layout ────────────────────────────────────────────────────────────
        gs = GridSpec(
            3, 2,
            figure=fig,
            height_ratios=[0.08, 0.65, 0.27],
            hspace=0.35,
            wspace=0.3,
        )

        ax_header = fig.add_subplot(gs[0, :])
        ax_radar  = fig.add_subplot(gs[1, 0])
        ax_trend  = fig.add_subplot(gs[1, 1])
        ax_table  = fig.add_subplot(gs[2, :])

        # ── Header ────────────────────────────────────────────────────────────
        ax_header.set_facecolor(_DARK_BG)
        ax_header.axis("off")
        player_name = trajectory.player_name
        header_text = title or (
            f"{player_name}  ·  {profile.position_group.title()}"
        )
        ax_header.text(
            0.5, 0.85, header_text,
            ha="center", va="top", transform=ax_header.transAxes,
            fontsize=20, fontweight="bold", color=_TEXT_PRIMARY,
        )
        ax_header.text(
            0.5, 0.1,
            f"Percentile profile vs {profile.n_peers} position peers  "
            f"·  {len(trajectory.df)} seasons loaded",
            ha="center", va="bottom", transform=ax_header.transAxes,
            fontsize=11, color=_TEXT_MUTED,
        )

        # ── Radar (embedded into ax_radar subplot) ────────────────────────────
        self._draw_mini_radar(ax_radar, profile)

        # ── Career trend (goals/90 only in report card) ───────────────────────
        self._draw_mini_trend(ax_trend, trajectory)

        # ── Stat table ────────────────────────────────────────────────────────
        self._draw_stat_table(ax_table, trajectory, profile)

        return fig

    def _draw_mini_radar(self, ax: plt.Axes, profile: PercentileProfile) -> None:
        """Draw simplified radar directly onto an existing axes."""
        metrics = list(profile.percentiles.keys())
        values  = [profile.percentiles[m] for m in metrics]
        n = len(metrics)

        if n < 3:
            ax.text(0.5, 0.5, "Not enough metrics", ha="center", va="center",
                    transform=ax.transAxes, color=_TEXT_MUTED)
            return

        # Draw using polar projection inside the given axes
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        values_plot = values + [values[0]]
        angles_plot = angles + [angles[0]]

        ax2 = ax.figure.add_axes(ax.get_position(), projection="polar",
                                 facecolor=_PANEL_BG)
        ax2.set_facecolor(_PANEL_BG)
        ax.remove()

        ax2.plot(angles_plot, values_plot, color=_ACCENT_GREEN, lw=2, zorder=3)
        ax2.fill(angles_plot, values_plot, color=_ACCENT_GREEN, alpha=0.3, zorder=2)

        ax2.set_xticks(angles)
        ax2.set_xticklabels(
            [m.replace("/90", "\n/90") for m in metrics],
            size=7, color=_TEXT_PRIMARY,
        )
        ax2.set_ylim(0, 100)
        ax2.set_yticks([25, 50, 75, 100])
        ax2.set_yticklabels(["25", "50", "75", "100"], size=6, color=_TEXT_MUTED)
        ax2.yaxis.grid(True, color=_GRID_COLOR, lw=0.5)
        ax2.xaxis.grid(True, color=_GRID_COLOR, lw=0.5)
        ax2.spines["polar"].set_color(_GRID_COLOR)
        ax2.set_title("Percentile Radar", color=_TEXT_PRIMARY,
                      fontsize=10, pad=15, fontweight="bold")

    def _draw_mini_trend(self, ax: plt.Axes, trajectory: TrajectoryResult) -> None:
        """Draw goals/90 trend line on a given axes."""
        ax.set_facecolor(_PANEL_BG)
        ax.spines[["top","right"]].set_visible(False)
        ax.spines[["left","bottom"]].set_color(_GRID_COLOR)

        df = trajectory.df
        col = "goals_p90"
        if col not in df.columns:
            ax.text(0.5, 0.5, "No goals data", ha="center", va="center",
                    transform=ax.transAxes, color=_TEXT_MUTED)
            return

        y = df[col].fillna(0).values.astype(float)
        x = np.arange(len(y))
        seasons = df["season_name"].tolist() if "season_name" in df.columns else [str(i) for i in x]

        ax.fill_between(x, y, alpha=0.2, color=_ACCENT_GREEN)
        ax.plot(x, y, color=_ACCENT_GREEN, lw=2.5, marker="o",
                markerfacecolor=_DARK_BG, markeredgecolor=_ACCENT_GREEN,
                markersize=7, zorder=3)

        if len(x) >= 2:
            coeffs  = np.polyfit(x, y, 1)
            trend_y = np.polyval(coeffs, x)
            ax.plot(x, trend_y, ls="--", color=_TEXT_MUTED, lw=1.2, alpha=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels(seasons, rotation=25, ha="right", fontsize=8, color=_TEXT_MUTED)
        ax.set_ylabel("Goals / 90", color=_TEXT_PRIMARY, fontsize=9)
        ax.set_title("Goals/90 Trend", color=_TEXT_PRIMARY, fontsize=10, fontweight="bold")
        ax.grid(axis="y", color=_GRID_COLOR, lw=0.4, alpha=0.6)
        ax.set_xlim(-0.4, len(x) - 0.6)
        ax.set_ylim(0, max(y.max() * 1.25, 0.1))

    def _draw_stat_table(
        self,
        ax: plt.Axes,
        trajectory: TrajectoryResult,
        profile: PercentileProfile,
    ) -> None:
        """Render a formatted stat summary table."""
        ax.set_facecolor(_PANEL_BG)
        ax.axis("off")

        metrics_display = list(profile.raw_values.items())[:8]
        if not metrics_display:
            return

        col_labels  = [m for m, _ in metrics_display]
        raw_row     = [f"{v:.3f}" for _, v in metrics_display]
        pct_row     = [f"{profile.percentiles.get(m, 0):.0f}th" for m, _ in metrics_display]

        table_data  = [raw_row, pct_row]
        row_labels  = ["Per 90", "Percentile"]

        table = ax.table(
            cellText=table_data,
            rowLabels=row_labels,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)

        # Style cells
        for (r, c), cell in table.get_celld().items():
            cell.set_facecolor(_PANEL_BG if r % 2 == 0 else _DARK_BG)
            cell.set_edgecolor(_GRID_COLOR)
            cell.set_text_props(color=_TEXT_PRIMARY)
            if r == 0:
                cell.set_facecolor(_ACCENT_GREEN)
                cell.set_text_props(color=_DARK_BG, fontweight="bold")
            if c == -1:
                cell.set_facecolor(_PANEL_BG)
                cell.set_text_props(color=_ACCENT_GREEN, fontweight="bold")


# ─────────────────────────────────────────────────────────────────────────────
# 7. VisualisationFactory  (Factory Pattern)
# ─────────────────────────────────────────────────────────────────────────────
class VisualisationFactory:
    """
    Factory: create any chart object by name.

    FACTORY PATTERN:
    ─────────────────
    The Streamlit dashboard imports VisualisationFactory only.  It never
    needs to know about RadarChart, CareerPlotter, PitchMapper, etc.
    Adding a new chart type = one entry in CHART_REGISTRY + the class.

    Usage:
        factory = VisualisationFactory()
        fig = factory.radar(profile)
        fig = factory.comparison(comparison_result)
        fig = factory.career(trajectory_result)
        fig = factory.report(trajectory, profile)
    """

    # OPEN/CLOSED: register new chart types here
    CHART_REGISTRY: ClassVar[Dict[str, Type[BaseChart]]] = {
        "radar":    RadarChart,
        "career":   CareerPlotter,
        "pitch":    PitchMapper,
        "cluster":  ClusterPlotter,
        "injury":   InjuryPlotter,
        "report":   ReportFigure,
    }

    def __init__(self) -> None:
        # Lazily-instantiated chart objects (one instance per type)
        self._instances: Dict[str, BaseChart] = {}

    def _get(self, chart_type: str) -> BaseChart:
        """Get or create a chart instance by type name."""
        if chart_type not in self.CHART_REGISTRY:
            raise ValueError(
                f"Unknown chart type '{chart_type}'. "
                f"Available: {list(self.CHART_REGISTRY)}"
            )
        if chart_type not in self._instances:
            self._instances[chart_type] = self.CHART_REGISTRY[chart_type]()
        return self._instances[chart_type]

    # ── Convenience methods (typed, IDE-friendly) ─────────────────────────────
    def radar(self, profile: PercentileProfile, title: Optional[str] = None) -> Figure:
        """Single-player radar chart."""
        return self._get("radar").plot(profile, title=title)

    def comparison(self, result: ComparisonResult, title: Optional[str] = None) -> Figure:
        """Two-player comparison radar."""
        return self._get("radar").plot_comparison(result, title=title)  # type: ignore[attr-defined]

    def career(self, trajectory: TrajectoryResult) -> Figure:
        """Career trajectory trend chart."""
        return self._get("career").plot(trajectory)

    def pass_map(self, events_df: pd.DataFrame, player_name: str = "") -> Figure:
        """Pass arrow map."""
        return self._get("pitch").plot(events_df, player_name, "pass_map")  # type: ignore[attr-defined]

    def shot_map(self, events_df: pd.DataFrame, player_name: str = "") -> Figure:
        """xG shot bubble map."""
        return self._get("pitch").plot(events_df, player_name, "shot_map")  # type: ignore[attr-defined]

    def pressure_map(self, events_df: pd.DataFrame, player_name: str = "") -> Figure:
        """Pressure heatmap."""
        return self._get("pitch").plot(events_df, player_name, "pressure_map")  # type: ignore[attr-defined]

    def clusters(self, result: ClusterResult, highlight: Optional[str] = None) -> Figure:
        """Cluster PCA scatter + profile bars."""
        return self._get("cluster").plot(result, highlight_player=highlight)  # type: ignore[attr-defined]

    def injury(self, result: InjuryImpactResult) -> Figure:
        """Injury impact pre/post comparison."""
        return self._get("injury").plot(result)

    def report(
        self,
        trajectory: TrajectoryResult,
        profile: PercentileProfile,
        title: Optional[str] = None,
    ) -> Figure:
        """One-page player report card."""
        return self._get("report").plot(trajectory, profile, title=title)  # type: ignore[attr-defined]

    def save(self, fig: Figure, path: Path) -> Path:
        """Save any figure to disk."""
        # Use CareerPlotter as a concrete BaseChart for IO operations
        return self._get("career").save(fig, path)

    def to_bytes(self, fig: Figure) -> bytes:
        """Serialise figure to PNG bytes for Streamlit."""
        return self._get("career").to_bytes(fig)
