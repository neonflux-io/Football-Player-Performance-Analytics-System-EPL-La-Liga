"""
dashboards/components.py
─────────────────────────────────────────────────────────────────────────────
Reusable Streamlit UI components shared across dashboard pages.

Each function takes data and renders a self-contained UI block.
No state management here — that lives in state.py.
No analytics here — that lives in analyzer.py.
This is purely presentation.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

from src.analyzer import PercentileProfile, TrajectoryResult, ComparisonResult
from src.config import cfg


def render_figure(fig: Figure, caption: str = "", use_container_width: bool = True) -> None:
    """
    Render a matplotlib Figure in Streamlit via PNG bytes.

    Streamlit's st.pyplot() has memory leak issues in long sessions;
    converting to bytes and using st.image() is more robust.

    Args:
        fig:                 Matplotlib Figure to display.
        caption:             Optional caption below the image.
        use_container_width: Stretch image to fill column width.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=cfg.FIGURE_DPI,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    st.image(buf, caption=caption, use_container_width=use_container_width)


def metric_card(label: str, value: Any, delta: Optional[float] = None,
                help_text: str = "") -> None:
    """
    Render a styled metric card using st.metric.

    Args:
        label:     Metric label.
        value:     Display value (formatted externally).
        delta:     Optional change value (green if positive, red if negative).
        help_text: Tooltip text shown on hover.
    """
    st.metric(
        label=label,
        value=value,
        delta=f"{delta:+.3f}" if delta is not None else None,
        help=help_text,
    )


def player_identity_header(player_name: str, position: str,
                            nationality: str, seasons_loaded: int) -> None:
    """Render the player name + metadata header block."""
    st.markdown(f"## {player_name}")
    cols = st.columns(3)
    cols[0].markdown(f"**Position:** {position}")
    cols[1].markdown(f"**Nationality:** {nationality}")
    cols[2].markdown(f"**Seasons loaded:** {seasons_loaded}")
    st.divider()


def percentile_bar_table(profile: PercentileProfile) -> None:
    """
    Render a colour-coded percentile table with inline progress bars.

    Each row shows: metric | raw value | percentile bar.
    Green = high percentile, orange = mid, red = low.

    Args:
        profile: PercentileProfile from PerformanceAnalyzer.
    """
    if not profile.percentiles:
        st.info("No percentile data available.")
        return

    rows = []
    for metric, pct in profile.percentiles.items():
        raw = profile.raw_values.get(metric, 0.0)
        colour = _percentile_colour(pct)
        bar = _progress_bar_html(pct, colour)
        rows.append({
            "Metric":     metric,
            "Per 90":     f"{raw:.3f}",
            "Percentile": f"{pct:.0f}th",
            "Rank bar":   bar,
        })

    df = pd.DataFrame(rows)
    st.markdown(
        df.to_html(escape=False, index=False),
        unsafe_allow_html=True,
    )


def comparison_table(result: ComparisonResult) -> None:
    """
    Render a side-by-side comparison table with colour-coded winner cells.

    Args:
        result: ComparisonResult from PerformanceAnalyzer.compare_players().
    """
    df = result.comparison_df.copy()
    if df.empty:
        st.info("No comparison data.")
        return

    # Highlight winner column
    def _style_edge(row):
        styles = [""] * len(row)
        edge = row.get("Edge", "")
        col_a = f"{result.player_a_name} (raw)"
        col_b = f"{result.player_b_name} (raw)"
        try:
            idx_a = list(row.index).index(col_a)
            idx_b = list(row.index).index(col_b)
            if edge == result.player_a_name:
                styles[idx_a] = "background-color: #1a3a1a; color: #39d353; font-weight: bold"
            elif edge == result.player_b_name:
                styles[idx_b] = "background-color: #1a1a3a; color: #58a6ff; font-weight: bold"
        except ValueError:
            pass
        return styles

    styled = df.style.apply(_style_edge, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)


def career_metrics_row(career_totals: Dict[str, float]) -> None:
    """
    Display 4 headline career stats in a single metrics row.

    Args:
        career_totals: Dict from player.career_totals().
    """
    cols = st.columns(4)
    stats = [
        ("⚽ Goals",    "goals",          None),
        ("🎯 Assists",  "assists",         None),
        ("📊 xG",       "xg",             None),
        ("⏱️ Minutes",  "minutes_played",  None),
    ]
    for col, (label, key, fmt) in zip(cols, stats):
        val = career_totals.get(key, 0)
        display = f"{val:.1f}" if isinstance(val, float) else str(int(val))
        col.metric(label, display)


def sidebar_data_loader(state) -> bool:
    """
    Render the sidebar competition/season selector and load button.

    Returns True if data is loaded and ready.

    Args:
        state: AppState instance.
    """
    st.sidebar.header("⚙️ Data Configuration")

    # Competition selector
    comp_options = {
        "La Liga (richest open data)": 11,
        "Premier League":              2,
    }
    comp_label = st.sidebar.selectbox(
        "Competition",
        options=list(comp_options.keys()),
        index=0,
        help="La Liga has the most complete StatsBomb open data.",
    )
    competition_id = comp_options[comp_label]

    # Season options for each competition
    season_options_map = {
        11: {   # La Liga
            "2021/2022 (most recent)": 90,
            "2020/2021":               42,
            "2019/2020":               26,
            "2018/2019":               2,
        },
        2: {    # EPL — limited open data
            "2003/2004":               37,
        },
    }
    season_options = season_options_map.get(competition_id, {"2021/2022": 90})
    season_label = st.sidebar.selectbox(
        "Season",
        options=list(season_options.keys()),
        index=0,
    )
    season_id = season_options[season_label]

    # Demo mode (cap matches for fast loading)
    demo_mode = st.sidebar.checkbox(
        "⚡ Demo mode (10 matches)",
        value=True,
        help="Load only 10 matches for a fast demo. Uncheck for full season analysis.",
    )
    max_matches = 10 if demo_mode else None

    st.sidebar.divider()

    # Load button
    load_clicked = st.sidebar.button(
        "🔄 Load Data",
        type="primary",
        use_container_width=True,
        help="Fetch StatsBomb open data and build the analytics engine.",
    )

    if load_clicked:
        state.clear()
        with st.sidebar:
            with st.spinner("Fetching StatsBomb data…"):
                success = state.load(competition_id, season_id, max_matches)
        if success:
            st.sidebar.success(
                f"✅ Loaded {len(state.player_names())} players"
            )
        else:
            st.sidebar.error(f"❌ {state.load_error}")

    # Status indicator
    if state.is_loaded:
        league = state.league
        st.sidebar.markdown(
            f"**Loaded:** {league.name}  \n"
            f"**Players:** {len(state.player_names())}  \n"
            f"**Teams:** {len(state.team_names())}"
        )

    return state.is_loaded


# ── Internal helpers ──────────────────────────────────────────────────────────
def _percentile_colour(pct: float) -> str:
    """Return a hex colour for a given percentile value."""
    if pct >= 80:
        return "#39d353"   # green
    if pct >= 50:
        return "#f0a500"   # amber
    return "#f78166"       # red


def _progress_bar_html(pct: float, colour: str) -> str:
    """Return an HTML inline progress bar for use in a DataFrame cell."""
    width = max(int(pct), 2)
    return (
        f'<div style="background:#21262d;border-radius:4px;height:14px;width:120px">'
        f'<div style="background:{colour};width:{width}%;height:100%;border-radius:4px"></div>'
        f'</div>'
    )
