"""
advanced_analysis.py
--------------------
Phase 4 Advanced Analysis: Per90Engine, InjuryImpactAnalyzer,
PositionalClusterer, AdvancedAnalyzer (Facade).
"""
from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.analyzer import PerformanceAnalyzer
from src.config import cfg
from src.models import Goalkeeper, League, Outfielder, Player, SeasonRecord
from src.utils import get_logger, per_90 as _per90_util, safe_divide, timer

log = get_logger()
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


@dataclass
class InjuryImpactResult:
    player_name: str
    player_id: int
    affected_season_name: str
    affected_season_minutes: float
    pre_metrics: Dict[str, float] = field(default_factory=dict)
    post_metrics: Dict[str, float] = field(default_factory=dict)
    delta: Dict[str, float] = field(default_factory=dict)
    recovery_score: float = 0.0
    interpretation: str = ""
    trajectory_df: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class ClusterResult:
    cluster_df: pd.DataFrame
    profiles: pd.DataFrame
    labels: Dict[int, str]
    n_clusters: int
    silhouette: float = 0.0
    inertia: float = 0.0


@dataclass
class Per90Context:
    stat: str
    player_name: str
    raw_per90: float
    adjusted_per90: float
    z_score: float
    percentile: float
    minutes_played: float
    confidence: str


class Per90Engine:
    HIGH_CONF_MINUTES: int = 1800
    MEDIUM_CONF_MINUTES: int = 900
    SHRINKAGE_K: float = 500.0

    def __init__(self, league: League) -> None:
        self._league = league
        self._stat_cache: Dict[Tuple, Tuple[float, float]] = {}

    def _league_mean_std(self, stat: str, season_id: Optional[int],
                         position_group: str, min_minutes: int = 450) -> Tuple[float, float]:
        key = (stat, season_id, position_group, min_minutes)
        if key in self._stat_cache:
            return self._stat_cache[key]
        values: List[float] = []
        for player in self._league.all_players(season_id=season_id):
            is_gk = isinstance(player, Goalkeeper)
            if position_group == "goalkeeper" and not is_gk: continue
            if position_group == "outfield" and is_gk: continue
            if season_id is not None:
                rec = player.get_season(season_id)
                if rec is None or rec.minutes_played < min_minutes: continue
                values.append(_per90_util(float(getattr(rec, stat, 0.0)), rec.minutes_played))
            else:
                totals = player.career_totals()
                minutes = totals.get("minutes_played", 0)
                if minutes < min_minutes: continue
                values.append(_per90_util(float(totals.get(stat, 0.0)), minutes))
        result = (float(np.mean(values)), float(np.std(values)) or 1.0) if len(values) >= 2 else (0.0, 1.0)
        self._stat_cache[key] = result
        return result

    def compute(self, player: Player, stat: str, season_id: Optional[int] = None,
                min_minutes: int = 450) -> Per90Context:
        position_group = "goalkeeper" if isinstance(player, Goalkeeper) else "outfield"
        if season_id is not None:
            rec = player.get_season(season_id)
            minutes = rec.minutes_played if rec else 0.0
            raw_stat = float(getattr(rec, stat, 0.0)) if rec else 0.0
        else:
            totals = player.career_totals()
            minutes = totals.get("minutes_played", 0.0)
            raw_stat = float(totals.get(stat, 0.0))
        raw_p90 = _per90_util(raw_stat, minutes)
        confidence = "high" if minutes >= self.HIGH_CONF_MINUTES else \
                     "medium" if minutes >= self.MEDIUM_CONF_MINUTES else "low"
        league_mean, league_std = self._league_mean_std(stat, season_id, position_group, min_minutes)
        weight = safe_divide(minutes, minutes + self.SHRINKAGE_K)
        adjusted_p90 = round(weight * raw_p90 + (1 - weight) * league_mean, 4)
        z_score = round(safe_divide(raw_p90 - league_mean, league_std), 3)
        all_vals = [p.per_90(stat, season_id=season_id) for p in self._league.all_players(season_id=season_id)
                    if (isinstance(p, Goalkeeper) if position_group == "goalkeeper" else not isinstance(p, Goalkeeper))]
        percentile = round(float(stats.percentileofscore(all_vals, raw_p90, kind="rank")), 1) if all_vals else 0.0
        return Per90Context(stat=stat, player_name=player.name, raw_per90=round(raw_p90, 4),
                            adjusted_per90=adjusted_p90, z_score=z_score, percentile=percentile,
                            minutes_played=minutes, confidence=confidence)

    def bulk_compute(self, player: Player, stats: List[str],
                     season_id: Optional[int] = None) -> Dict[str, Per90Context]:
        return {s: self.compute(player, s, season_id) for s in stats}


class InjuryImpactAnalyzer:
    ABSENCE_THRESHOLD_RATIO: float = 0.75
    IMPACT_METRICS: List[str] = ["goals","assists","xg","xa","key_passes","dribbles_completed","pressures"]

    def __init__(self, league: League, per90_engine: Per90Engine) -> None:
        self._league = league
        self._engine = per90_engine

    def detect_absence_season(self, player: Player) -> Optional[int]:
        if len(player.seasons) < 2: return None
        minutes_per_season = {rec.season_id: rec.minutes_played for rec in player.seasons}
        mean_minutes = float(np.mean(list(minutes_per_season.values())))
        if mean_minutes == 0: return None
        min_sid = min(minutes_per_season, key=lambda s: minutes_per_season[s])
        return min_sid if minutes_per_season[min_sid] < mean_minutes * self.ABSENCE_THRESHOLD_RATIO else None

    def analyse_absence(self, player_name: str,
                        absence_season_id: Optional[int] = None) -> InjuryImpactResult:
        player = self._league.find_player(player_name)
        if player is None: raise ValueError(f"Player '{player_name}' not found.")
        if len(player.seasons) < 2:
            raise ValueError(f"'{player.name}' needs >= 2 seasons (found {len(player.seasons)}).")
        if absence_season_id is None:
            absence_season_id = self.detect_absence_season(player)
        seasons_sorted = player.seasons
        season_ids = [r.season_id for r in seasons_sorted]
        if absence_season_id is not None and absence_season_id in season_ids:
            idx = season_ids.index(absence_season_id)
            pre_season_id = season_ids[idx-1] if idx > 0 else None
            post_season_id = season_ids[idx+1] if idx < len(season_ids)-1 else None
            affected_rec = player.get_season(absence_season_id)
            affected_name = affected_rec.season_name if affected_rec else "Unknown"
            affected_minutes = affected_rec.minutes_played if affected_rec else 0.0
        else:
            pre_season_id, post_season_id = season_ids[0], season_ids[-1]
            absence_season_id = None
            affected_name, affected_minutes = "No clear absence detected", 0.0

        def _metrics(sid):
            if sid is None: return {m: 0.0 for m in self.IMPACT_METRICS}
            return {m: round(player.per_90(m, season_id=sid), 3) for m in self.IMPACT_METRICS}

        pre_metrics = _metrics(pre_season_id)
        post_metrics = _metrics(post_season_id)
        delta = {m: round(post_metrics.get(m,0) - pre_metrics.get(m,0), 3) for m in self.IMPACT_METRICS}
        positive_deltas = [max(d,0.0) for d in delta.values()]
        total_pre = sum(abs(v) for v in pre_metrics.values()) or 1.0
        recovery_score = round(min(sum(positive_deltas)/total_pre, 1.0), 3)
        n_improved = sum(1 for d in delta.values() if d > 0.01)
        n_declined = sum(1 for d in delta.values() if d < -0.01)
        if absence_season_id is None:
            interpretation = f"No clear injury season detected. Showing first-to-last comparison."
        elif n_improved > n_declined:
            interpretation = f"{player.name} shows strong recovery: {n_improved}/{len(delta)} metrics improved. Score: {recovery_score:.2f}."
        elif n_declined > n_improved:
            interpretation = f"{player.name} shows lingering impact: {n_declined}/{len(delta)} metrics declined. Score: {recovery_score:.2f}."
        else:
            interpretation = f"{player.name} returned to similar form. Score: {recovery_score:.2f}."
        return InjuryImpactResult(
            player_name=player.name, player_id=player.player_id,
            affected_season_name=affected_name, affected_season_minutes=affected_minutes,
            pre_metrics=pre_metrics, post_metrics=post_metrics, delta=delta,
            recovery_score=recovery_score, interpretation=interpretation,
            trajectory_df=player.career_stats_df(),
        )


class PositionalClusterer:
    CLUSTER_METRICS: List[str] = ["Goals/90","Assists/90","xG/90","xA/90",
                                   "Key Passes/90","Dribbles/90","Pressures/90","Carries/90"]
    METRIC_TO_ARCHETYPE: Dict[str, str] = {
        "Goals/90": "Clinical Finisher", "Assists/90": "Creative Playmaker",
        "xG/90": "Volume Striker", "xA/90": "Chance Creator",
        "Key Passes/90": "Technical Architect", "Dribbles/90": "Dynamic Dribbler",
        "Pressures/90": "High-Press Engine", "Carries/90": "Progressive Ball-Carrier",
    }

    def __init__(self, league: League, n_clusters: int = 4,
                 algorithm: Literal["kmeans"] = "kmeans", random_state: int = 42) -> None:
        self._league = league
        self.n_clusters = n_clusters
        self.algorithm = algorithm
        self.random_state = random_state
        self._model: Optional[KMeans] = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_cols: List[str] = []
        self._is_fitted: bool = False

    def _build_feature_matrix(self, season_id, position_group, min_minutes):
        analyzer = PerformanceAnalyzer(self._league)
        peer_df = analyzer._build_percentile_table(season_id, position_group, min_minutes)
        if peer_df.empty: return pd.DataFrame(), pd.DataFrame()
        available = [m for m in self.CLUSTER_METRICS if m in peer_df.columns]
        self._feature_cols = available
        meta_cols = [c for c in ["player_id","player_name","position","minutes_played"] if c in peer_df.columns]
        return peer_df[meta_cols].copy(), peer_df[available].fillna(0).copy()

    def _auto_label_clusters(self, centroids: np.ndarray) -> Dict[int, str]:
        labels: Dict[int, str] = {}
        used: Dict[str, int] = {}
        for cid in range(len(centroids)):
            dominant_idx = int(np.argmax(centroids[cid]))
            dominant_metric = self._feature_cols[dominant_idx] if dominant_idx < len(self._feature_cols) else "Unknown"
            archetype = self.METRIC_TO_ARCHETYPE.get(dominant_metric, f"Cluster {cid}")
            if archetype in used:
                sorted_idxs = np.argsort(centroids[cid])[::-1]
                for idx in sorted_idxs[1:]:
                    m = self._feature_cols[idx] if idx < len(self._feature_cols) else ""
                    alt = self.METRIC_TO_ARCHETYPE.get(m, "")
                    if alt and alt not in used:
                        archetype = f"{alt} (variant)"; break
                else:
                    archetype = f"Cluster {cid}"
            labels[cid] = archetype
            used[archetype] = cid
        return labels

    def fit_and_label(self, season_id=None, position_group="outfield", min_minutes=450) -> ClusterResult:
        meta_df, feature_df = self._build_feature_matrix(season_id, position_group, min_minutes)
        if feature_df.empty: raise ValueError("No qualifying players found for clustering.")
        if len(feature_df) < self.n_clusters:
            raise ValueError(f"Only {len(feature_df)} players but n_clusters={self.n_clusters}.")
        self._scaler = StandardScaler()
        scaled = self._scaler.fit_transform(feature_df.values)
        self._model = KMeans(n_clusters=min(self.n_clusters, len(scaled)),
                             random_state=self.random_state, n_init=10, max_iter=300)
        self._model.fit(scaled)
        self._is_fitted = True
        labels_arr = self._model.labels_
        centroids = self._model.cluster_centers_
        silhouette = 0.0
        if len(set(labels_arr)) > 1:
            try:
                from sklearn.metrics import silhouette_score
                silhouette = round(float(silhouette_score(scaled, labels_arr)), 3)
            except Exception: pass
        archetype_labels = self._auto_label_clusters(centroids)
        result_df = pd.concat([meta_df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)
        result_df["cluster_id"] = labels_arr
        result_df["archetype"] = result_df["cluster_id"].map(archetype_labels)
        centroid_original = self._scaler.inverse_transform(centroids)
        profiles_df = pd.DataFrame(centroid_original, columns=self._feature_cols)
        profiles_df.insert(0, "cluster_id", range(len(centroids)))
        profiles_df.insert(1, "archetype", profiles_df["cluster_id"].map(archetype_labels))
        profiles_df["n_players"] = result_df.groupby("cluster_id").size().values
        log.info(f"Clustering: {self.n_clusters} clusters | silhouette={silhouette:.3f}")
        return ClusterResult(cluster_df=result_df.reset_index(drop=True), profiles=profiles_df,
                             labels=archetype_labels, n_clusters=self.n_clusters,
                             silhouette=silhouette, inertia=float(self._model.inertia_))

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


class AdvancedAnalyzer:
    """Facade over Per90Engine, InjuryImpactAnalyzer, PositionalClusterer."""

    def __init__(self, league: League, n_clusters: int = 4) -> None:
        self._league = league
        self._per90 = Per90Engine(league)
        self._injury = InjuryImpactAnalyzer(league, self._per90)
        self._clusterer = PositionalClusterer(league, n_clusters=n_clusters)
        log.info(f"AdvancedAnalyzer initialised | league='{league.name}' | n_clusters={n_clusters}")

    def rich_per90(self, player_name: str, stats: List[str],
                   season_id: Optional[int] = None) -> Dict[str, Per90Context]:
        player = self._league.find_player(player_name)
        if player is None: raise ValueError(f"Player '{player_name}' not found.")
        return self._per90.bulk_compute(player, stats, season_id)

    def injury_impact(self, player_name: str,
                      absence_season_id: Optional[int] = None) -> InjuryImpactResult:
        return self._injury.analyse_absence(player_name, absence_season_id)

    def detect_absence(self, player_name: str) -> Optional[int]:
        player = self._league.find_player(player_name)
        if player is None: raise ValueError(f"Player '{player_name}' not found.")
        return self._injury.detect_absence_season(player)

    def cluster_players(self, season_id=None, position_group="outfield",
                        min_minutes=450) -> ClusterResult:
        return self._clusterer.fit_and_label(season_id, position_group, min_minutes)

    def per90_summary_table(self, season_id=None, position_group="outfield",
                            min_minutes=450, stats=None) -> pd.DataFrame:
        if stats is None:
            stats = ["goals","assists","xg","xa","key_passes","pressures","dribbles_completed","carries"]
        players = [p for p in self._league.all_players(season_id=season_id)
                   if (isinstance(p, Goalkeeper) if position_group == "goalkeeper"
                       else not isinstance(p, Goalkeeper))]
        rows = []
        for player in players:
            if season_id is not None:
                rec = player.get_season(season_id)
                if rec is None or rec.minutes_played < min_minutes: continue
                mins = rec.minutes_played
            else:
                totals = player.career_totals()
                mins = totals.get("minutes_played", 0)
                if mins < min_minutes: continue
            row: Dict[str, Any] = {"player_name": player.name, "position": player.position, "minutes_played": mins}
            for s in stats:
                ctx = self._per90.compute(player, s, season_id)
                row[f"{s}_raw_p90"] = ctx.raw_per90
                row[f"{s}_adj_p90"] = ctx.adjusted_per90
                row[f"{s}_zscore"] = ctx.z_score
                row[f"{s}_confidence"] = ctx.confidence
            rows.append(row)
        if not rows: return pd.DataFrame()
        df = pd.DataFrame(rows)
        sort_col = "goals_raw_p90" if "goals_raw_p90" in df.columns else df.columns[0]
        return df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    @property
    def league(self) -> League:
        return self._league

    def __repr__(self) -> str:
        return f"AdvancedAnalyzer(league='{self._league.name}', n_clusters={self._clusterer.n_clusters})"
