"""
dashboards/app.py
─────────────────────────────────────────────────────────────────────────────
Football Player Performance Analytics System — Streamlit Dashboard

Launch with:
    cd football_analytics
    streamlit run dashboards/app.py

Pages
──────
    🏠 Overview          – League-wide top scorers, team selector
    👤 Player Profile    – Individual radar, career trend, stat table
    ⚔️  Comparison        – Head-to-head radar + delta table
    🔬 Advanced Analysis – Injury impact & clustering archetypes
    📋 Team Report       – Squad depth, goal contributions

Architecture
────────────
    • AppState (state.py)        – session-state singleton for data objects
    • VisualisationFactory       – all chart rendering
    • PerformanceAnalyzer        – all analytics
    • AdvancedAnalyzer (Phase 4) – injury impact + clustering

OOP concept: the dashboard is a CONTROLLER — it reads user input,
delegates to the model layer (analyzers), and passes results to the
view layer (VisualisationFactory / components.py).  Zero analytics
or rendering logic lives in this file itself.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when running via `streamlit run`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Football Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Project imports ───────────────────────────────────────────────────────────
from dashboards.components import (
    career_metrics_row,
    comparison_table,
    percentile_bar_table,
    player_identity_header,
    render_figure,
    sidebar_data_loader,
)
from dashboards.state import AppState
from src.visualizations import VisualisationFactory

# ── Global singletons (one factory per session, shared across pages) ──────────
@st.cache_resource
def get_factory() -> VisualisationFactory:
    """Cache the VisualisationFactory so chart instances are reused."""
    return VisualisationFactory()


# ── Custom CSS (dark theme polish) ────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark background override */
    .stApp { background-color: #0d1117; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 12px 16px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #161b22; }

    /* Tab headers */
    .stTabs [data-baseweb="tab"] {
        font-size: 15px;
        font-weight: 600;
        color: #8b949e;
    }
    .stTabs [aria-selected="true"] { color: #39d353 !important; }

    /* DataFrames */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* Dividers */
    hr { border-color: #21262d; }

    /* Info / warning boxes */
    .stAlert { border-radius: 8px; }

    /* Headers */
    h1, h2, h3 { color: #e6edf3; }
    p, li { color: #c9d1d9; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
state   = AppState()
factory = get_factory()

data_ready = sidebar_data_loader(state)

# Navigation
st.sidebar.divider()
st.sidebar.header("🗂️ Navigation")
PAGE_OPTIONS = [
    "🏠 Overview",
    "👤 Player Profile",
    "⚔️ Head-to-Head",
    "🔬 Advanced Analysis",
    "📋 Team Report",
]
page = st.sidebar.radio(
    "Go to page",
    options=PAGE_OPTIONS,
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.markdown(
    "_Football Analytics System_  \n"
    "_Built with StatsBomb open data_  \n"
    "_EPL & La Liga_",
    help="Phase 6 of a 7-phase OOP portfolio project"
)


# ─────────────────────────────────────────────────────────────────────────────
# GATE: require data before showing any page
# ─────────────────────────────────────────────────────────────────────────────
if not data_ready:
    st.markdown("# ⚽ Football Player Performance Analytics")
    st.markdown("### Select a competition and season in the sidebar, then click **Load Data**.")

    col1, col2, col3 = st.columns(3)
    col1.markdown("""
    **Phase 1** · StatsBomb Loader
    Fast, cached JSON fetching from the StatsBomb open-data GitHub repository.
    """)
    col2.markdown("""
    **Phase 2–4** · Analytics Engine
    Player, Team, League domain models + PerformanceAnalyzer + AdvancedAnalyzer (clustering, injury modelling).
    """)
    col3.markdown("""
    **Phase 5–6** · Visualisation
    Radar charts, career trend plots, pitch maps, and this Streamlit dashboard.
    """)

    if state.load_error:
        st.error(f"**Load error:** {state.load_error}")
        st.info("StatsBomb data is fetched from GitHub — ensure you have internet access, "
                "or check the cache directory for previously downloaded files.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: 🏠 OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    league   = state.league
    analyzer = state.analyzer

    st.markdown(f"# ⚽ {league.name} Overview")
    st.markdown(f"Season data loaded · **{len(state.player_names())} players** across **{len(state.team_names())} teams**")
    st.divider()

    # ── KPI metrics row ───────────────────────────────────────────────────────
    all_players = league.all_players()
    total_goals   = sum(r.goals   for p in all_players for r in p.seasons)
    total_assists = sum(r.assists for p in all_players for r in p.seasons)
    total_xg      = sum(r.xg      for p in all_players for r in p.seasons)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("🏟️ Teams",         len(state.team_names()))
    kpi2.metric("👤 Players",        len(state.player_names()))
    kpi3.metric("⚽ Goals (loaded)", total_goals)
    kpi4.metric("📊 Total xG",      f"{total_xg:.1f}")

    st.divider()

    # ── Top scorers table ─────────────────────────────────────────────────────
    st.markdown("### 🏆 Top Scorers")
    try:
        top_scorers = analyzer.league_rankings(
            metric="Goals/90",
            season_id=state.season_id,
            position_group="outfield",
            min_minutes=100,
            top_n=15,
        )
        if not top_scorers.empty:
            st.dataframe(
                top_scorers.style.background_gradient(
                    subset=["Goals/90"],
                    cmap="Greens",
                ),
                use_container_width=True,
                height=420,
            )
    except Exception as e:
        st.warning(f"Could not build rankings: {e}")

    st.divider()

    # ── Two-column: xG leaders + Key Pass leaders ─────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 📊 xG Leaders")
        try:
            xg_rank = analyzer.league_rankings("xG/90", state.season_id,
                                                min_minutes=100, top_n=8)
            if not xg_rank.empty:
                st.dataframe(xg_rank, use_container_width=True,
                             hide_index=False, height=280)
        except Exception as e:
            st.warning(str(e))

    with col_right:
        st.markdown("### 🎯 Assists Leaders")
        try:
            ast_rank = analyzer.league_rankings("Assists/90", state.season_id,
                                                 min_minutes=100, top_n=8)
            if not ast_rank.empty:
                st.dataframe(ast_rank, use_container_width=True,
                             hide_index=False, height=280)
        except Exception as e:
            st.warning(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: 👤 PLAYER PROFILE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "👤 Player Profile":
    analyzer = state.analyzer
    league   = state.league

    st.markdown("# 👤 Player Profile")

    # Player selector
    player_names = state.player_names()
    if not player_names:
        st.warning("No players loaded.")
        st.stop()

    selected_name = st.selectbox(
        "Select a player",
        options=player_names,
        index=0,
        help="Type to search by name",
    )

    player = league.find_player(selected_name)
    if player is None:
        st.error(f"Player '{selected_name}' not found.")
        st.stop()

    # Identity header
    player_identity_header(
        player.name,
        player.position,
        player.nationality,
        len(player.seasons),
    )

    # Career totals row
    career_metrics_row(player.career_totals())
    st.divider()

    # Summary line
    st.markdown(f"**Performance summary:** `{player.summary()}`")
    st.divider()

    # ── Two columns: radar | career trend ────────────────────────────────────
    col_radar, col_trend = st.columns([1, 1])

    with col_radar:
        st.markdown("#### 📡 Percentile Radar")
        with st.spinner("Building radar…"):
            try:
                profile = analyzer.player_percentile_profile(
                    selected_name,
                    season_id=state.season_id,
                    min_minutes=100,
                )
                if profile.percentiles:
                    fig = factory.radar(profile)
                    render_figure(fig)
                else:
                    st.info("Not enough peers for percentile ranking.")
            except Exception as e:
                st.warning(f"Radar unavailable: {e}")

    with col_trend:
        st.markdown("#### 📈 Career Trajectory")
        with st.spinner("Building trend chart…"):
            try:
                trajectory = analyzer.career_trajectory(selected_name)
                if len(trajectory.df) >= 1:
                    fig = factory.career(trajectory)
                    render_figure(fig)
                    if trajectory.trend_direction != "stable":
                        arrow = "↗️" if trajectory.trend_direction == "improving" else "↘️"
                        st.markdown(
                            f"{arrow} **Trend:** {trajectory.trend_direction.title()} "
                            f"(slope {trajectory.trend_slope:+.4f}/season) · "
                            f"**Peak:** {trajectory.peak_season_name} "
                            f"({trajectory.peak_metric} = {trajectory.peak_value:.3f})"
                        )
            except Exception as e:
                st.warning(f"Trajectory unavailable: {e}")

    st.divider()

    # ── Percentile table ──────────────────────────────────────────────────────
    st.markdown("#### 📋 Percentile Breakdown")
    with st.spinner("Computing percentiles…"):
        try:
            profile = analyzer.player_percentile_profile(
                selected_name, season_id=state.season_id, min_minutes=100
            )
            if profile.percentiles:
                percentile_bar_table(profile)
            else:
                st.info("Percentile data not available for this player.")
        except Exception as e:
            st.warning(str(e))

    st.divider()

    # ── Similar players ───────────────────────────────────────────────────────
    st.markdown("#### 🔍 Similar Playing Styles")
    with st.spinner("Finding similar players…"):
        try:
            similar_df = analyzer.find_similar_players(
                selected_name, season_id=state.season_id,
                min_minutes=100, top_n=5,
            )
            if not similar_df.empty:
                st.dataframe(similar_df, use_container_width=True, hide_index=False)
            else:
                st.info("Not enough peers for similarity analysis.")
        except Exception as e:
            st.warning(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ⚔️ HEAD-TO-HEAD
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚔️ Head-to-Head":
    analyzer     = state.analyzer
    player_names = state.player_names()

    st.markdown("# ⚔️ Head-to-Head Comparison")

    if len(player_names) < 2:
        st.warning("Need at least 2 players loaded.")
        st.stop()

    col_a, col_b = st.columns(2)
    player_a_name = col_a.selectbox("Player A", player_names, index=0, key="cmp_a")
    player_b_name = col_b.selectbox(
        "Player B",
        [n for n in player_names if n != player_a_name],
        index=min(1, len(player_names) - 2),
        key="cmp_b",
    )

    if player_a_name == player_b_name:
        st.warning("Please select two different players.")
        st.stop()

    st.divider()

    with st.spinner(f"Comparing {player_a_name} vs {player_b_name}…"):
        try:
            result = analyzer.compare_players(
                player_a_name, player_b_name,
                season_id=state.season_id,
                min_minutes=100,
            )
        except Exception as e:
            st.error(str(e))
            st.stop()

    # ── Comparison radar ──────────────────────────────────────────────────────
    st.markdown("#### 📡 Comparison Radar")
    fig = factory.comparison(result)
    render_figure(fig)
    st.divider()

    # ── Win/loss summary ──────────────────────────────────────────────────────
    wins_a = sum(1 for v in result.edge.values() if v == result.player_a_name)
    wins_b = sum(1 for v in result.edge.values() if v == result.player_b_name)
    draws  = sum(1 for v in result.edge.values() if v == "draw")

    col1, col2, col3 = st.columns(3)
    col1.metric(f"✅ {player_a_name} wins", wins_a)
    col2.metric(f"✅ {player_b_name} wins", wins_b)
    col3.metric("🤝 Draws", draws)

    overall_winner = (
        player_a_name if wins_a > wins_b
        else player_b_name if wins_b > wins_a
        else "Even"
    )
    st.info(f"**Overall edge: {overall_winner}** across {len(result.edge)} metrics")
    st.divider()

    # ── Detailed comparison table ─────────────────────────────────────────────
    st.markdown("#### 📊 Metric-by-Metric Breakdown")
    comparison_table(result)

    st.divider()

    # ── Individual percentile profiles side by side ───────────────────────────
    st.markdown("#### 📡 Individual Percentile Profiles")
    col_pa, col_pb = st.columns(2)

    for col, name, colour in [(col_pa, player_a_name, "green"),
                               (col_pb, player_b_name, "blue")]:
        with col:
            st.markdown(f"**{name}**")
            try:
                profile = analyzer.player_percentile_profile(
                    name, season_id=state.season_id, min_minutes=100
                )
                fig = factory.radar(profile)
                render_figure(fig)
            except Exception as e:
                st.warning(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: 🔬 ADVANCED ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔬 Advanced Analysis":
    advanced     = state.advanced
    analyzer     = state.analyzer
    player_names = state.player_names()

    st.markdown("# 🔬 Advanced Analysis")

    tab_injury, tab_cluster = st.tabs(["💉 Injury Impact", "🎯 Style Archetypes"])

    # ── Tab 1: Injury Impact ──────────────────────────────────────────────────
    with tab_injury:
        st.markdown("### 💉 Injury / Absence Impact Modelling")
        st.markdown(
            "Compares a player's per-90 metrics in the season **before** and **after** "
            "an absence-affected campaign (detected as a low-minutes season)."
        )

        selected_player = st.selectbox(
            "Select player",
            player_names,
            key="injury_player",
            help="Players with ≥ 2 seasons loaded can be analysed.",
        )

        if st.button("🔍 Analyse Absence Impact", type="primary"):
            with st.spinner("Modelling absence impact…"):
                try:
                    result = advanced.injury_impact(selected_player)

                    st.markdown(f"#### Results for {result.player_name}")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Affected season",    result.affected_season_name)
                    col2.metric("Minutes (affected)", f"{result.affected_season_minutes:.0f}")
                    col3.metric("Recovery score",     f"{result.recovery_score:.2f} / 1.00")

                    st.markdown(f"**Interpretation:** {result.interpretation}")
                    st.divider()

                    fig = factory.injury(result)
                    render_figure(fig)

                    st.divider()
                    st.markdown("#### Metric Deltas (Post − Pre)")
                    delta_df = pd.DataFrame([
                        {"Metric": k, "Pre (per 90)": f"{result.pre_metrics[k]:.3f}",
                         "Post (per 90)": f"{result.post_metrics[k]:.3f}",
                         "Delta": f"{v:+.3f}"}
                        for k, v in result.delta.items()
                    ])
                    st.dataframe(delta_df, use_container_width=True, hide_index=True)

                except ValueError as e:
                    st.warning(str(e))
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

    # ── Tab 2: Clustering ─────────────────────────────────────────────────────
    with tab_cluster:
        st.markdown("### 🎯 Player Style Archetypes")
        st.markdown(
            "K-Means clustering groups outfield players into style archetypes "
            "based on their per-90 metric fingerprint. "
            "Similar playing profiles cluster together."
        )

        col_n, col_min = st.columns(2)
        n_clusters  = col_n.slider(  "Number of archetypes", 2, 6, 4)
        min_minutes = col_min.slider("Min minutes filter",   50, 500, 100, step=50)

        highlight_player = st.selectbox(
            "Highlight a player (optional)",
            ["None"] + player_names,
            key="cluster_highlight",
        )

        if st.button("🔬 Run Clustering", type="primary"):
            with st.spinner("Fitting archetypes…"):
                try:
                    from src.advanced_analysis import AdvancedAnalyzer
                    adv = AdvancedAnalyzer(state.league, n_clusters=n_clusters)
                    result = adv.cluster_players(
                        season_id=state.season_id,
                        position_group="outfield",
                        min_minutes=min_minutes,
                    )

                    st.markdown(
                        f"**{result.n_clusters} archetypes** found across "
                        f"**{len(result.cluster_df)} players** | "
                        f"Silhouette: **{result.silhouette:.3f}**"
                    )

                    highlight = None if highlight_player == "None" else highlight_player
                    fig = factory.clusters(result, highlight=highlight)
                    render_figure(fig)

                    st.divider()
                    st.markdown("#### Archetype Profiles")
                    st.dataframe(
                        result.profiles.drop(columns=["cluster_id"], errors="ignore"),
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.divider()
                    st.markdown("#### Player Assignments")
                    display_cols = ["player_name", "position", "archetype", "minutes_played"]
                    avail_cols   = [c for c in display_cols if c in result.cluster_df.columns]
                    st.dataframe(
                        result.cluster_df[avail_cols].sort_values("archetype"),
                        use_container_width=True,
                        hide_index=True,
                        height=350,
                    )

                except ValueError as e:
                    st.warning(str(e))
                except Exception as e:
                    st.error(f"Clustering failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: 📋 TEAM REPORT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📋 Team Report":
    analyzer   = state.analyzer
    team_names = state.team_names()

    st.markdown("# 📋 Team Report")

    if not team_names:
        st.warning("No teams loaded.")
        st.stop()

    selected_team = st.selectbox("Select team", team_names, index=0)

    with st.spinner(f"Building report for {selected_team}…"):
        try:
            report = analyzer.analyse_team(selected_team, state.season_id)
        except Exception as e:
            st.error(str(e))
            st.stop()

    # ── KPI row ───────────────────────────────────────────────────────────────
    st.divider()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👥 Squad size",     report.squad_size)
    c2.metric("⚽ Goals",          report.total_goals)
    c3.metric("🎯 Assists",        report.total_assists)
    c4.metric("🏆 Top scorer",     report.top_scorer_name)
    c5.metric("⭐ Top scorer goals", report.top_scorer_goals)

    st.divider()

    # ── Goal contribution chart ───────────────────────────────────────────────
    col_gc, col_squad = st.columns([1, 1])

    with col_gc:
        st.markdown("#### 🥧 Goal Contribution Share")
        gc_df = report.goal_contribution_df
        if not gc_df.empty:
            # Simple bar chart of top 10 contributors
            top10 = gc_df.head(10).set_index("player_name")
            st.bar_chart(
                top10[["goals", "assists"]],
                height=320,
                color=["#39d353", "#58a6ff"],
            )

    with col_squad:
        st.markdown("#### 📊 Squad Stats")
        squad_df = report.squad_stats_df
        if not squad_df.empty:
            display_cols = [c for c in
                ["player_name", "goals", "assists", "xg", "minutes_played",
                 "pass_completion_rate"]
                if c in squad_df.columns]
            st.dataframe(
                squad_df[display_cols]
                .rename(columns={
                    "player_name": "Player",
                    "goals": "G",
                    "assists": "A",
                    "xg": "xG",
                    "minutes_played": "Mins",
                    "pass_completion_rate": "Pass%",
                })
                .style.background_gradient(subset=["G"], cmap="Greens"),
                use_container_width=True,
                height=350,
                hide_index=True,
            )

    st.divider()

    # ── Full goal contribution table ──────────────────────────────────────────
    st.markdown("#### 🏅 Full Goal Contribution Table")
    gc_display = report.goal_contribution_df
    if not gc_display.empty:
        st.dataframe(
            gc_display.style.background_gradient(
                subset=["contribution_share_%"], cmap="YlGn"
            ),
            use_container_width=True,
            hide_index=True,
        )

    # ── Player profile deep-dive from team page ───────────────────────────────
    st.divider()
    st.markdown("#### 🔎 Deep-dive: individual player from this squad")
    squad_players = sorted(
        p.name for p in state.league.get_team(selected_team, state.season_id).players
    ) if state.league.get_team(selected_team, state.season_id) else []

    if squad_players:
        drill_player = st.selectbox("Select player for radar", squad_players, key="team_drill")
        if st.button("Show Radar", key="team_drill_btn"):
            try:
                profile = analyzer.player_percentile_profile(
                    drill_player, season_id=state.season_id, min_minutes=50
                )
                fig = factory.radar(profile)
                render_figure(fig)
            except Exception as e:
                st.warning(str(e))


# ── Import pandas for delta_df creation in Advanced tab ──────────────────────
import pandas as pd   # noqa: E402 (imported late to avoid Streamlit re-run overhead)
