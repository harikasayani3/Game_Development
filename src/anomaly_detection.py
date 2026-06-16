from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler


def build_match_suspicion_flags(raw_match_df: pd.DataFrame) -> pd.DataFrame:
    """Compute match-level suspicious flags for rule-based scoring."""
    matches = raw_match_df.copy()
    matches["match_duration_minutes"] = matches["match_duration_seconds"] / 60
    matches["kd_ratio"] = matches["kills"] / (matches["deaths"] + 1)
    matches["score_per_min"] = matches["score"] / matches["match_duration_minutes"].replace(0, 1)
    matches["kills_per_min"] = matches["kills"] / matches["match_duration_minutes"].replace(0, 1)
    matches["is_rule_suspicious"] = (
        (matches["kills_per_min"] > 10)
        | (matches["score_per_min"] > 5000)
        | (matches["kd_ratio"] > 25)
        | (matches["kills"] > 100)
        | (matches["match_duration_seconds"] < 30)
    )
    return matches


def compute_rule_scores(player_stats: pd.DataFrame, raw_match_df: pd.DataFrame) -> pd.DataFrame:
    """Assign a rule-based score and list reasons for suspicious behavior."""
    match_flags = build_match_suspicion_flags(raw_match_df)
    flagged_players = match_flags[match_flags["is_rule_suspicious"]].copy()
    player_rule = player_stats[["player_id"]].copy()
    player_rule["rule_score"] = 0.0
    player_rule["reasons"] = ""

    if not flagged_players.empty:
        grouped = flagged_players.groupby("player_id")
        player_rule = player_rule.set_index("player_id")
        for player_id, player_matches in grouped:
            reason_set: List[str] = []
            if (player_matches["kills_per_min"] > 10).any():
                reason_set.append("high_kills_per_min")
            if (player_matches["score_per_min"] > 5000).any():
                reason_set.append("high_score_per_min")
            if (player_matches["kd_ratio"] > 25).any():
                reason_set.append("high_kd_ratio")
            if (player_matches["kills"] > 100).any():
                reason_set.append("high_kills")
            if (player_matches["match_duration_seconds"] < 30).any():
                reason_set.append("short_match_duration")

            player_rule.loc[player_id, "rule_score"] = min(1.0, len(reason_set) / 3.0)
            player_rule.loc[player_id, "reasons"] = ", ".join(reason_set)

        player_rule = player_rule.reset_index()
    return player_rule


def compute_statistical_scores(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Compute a suspiciousness score using aggregated player statistics."""
    features = [
        "average_score_per_min",
        "average_kills_per_min",
        "average_kd_ratio",
        "average_ping",
    ]
    scaler = MinMaxScaler()
    stats = player_stats[features].copy()
    stats["average_ping"] = stats["average_ping"].max() - stats["average_ping"]
    normalized = scaler.fit_transform(stats)
    score = np.mean(normalized, axis=1)
    output = player_stats[["player_id"]].copy()
    output["statistical_score"] = np.clip(score, 0.0, 1.0)
    return output


def compute_ml_scores(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Detect outliers using IsolationForest and convert anomaly outputs to scores."""
    features = [
        "average_score_per_min",
        "average_kills_per_min",
        "average_kd_ratio",
        "average_ping",
    ]
    ml_features = player_stats[features].copy()
    forest = IsolationForest(contamination=0.02, random_state=42)
    forest.fit(ml_features)
    anomaly_distance = forest.decision_function(ml_features)
    inverted = -anomaly_distance.reshape(-1, 1)
    normalized = MinMaxScaler().fit_transform(inverted).reshape(-1)
    output = player_stats[["player_id"]].copy()
    output["ml_score"] = np.clip(normalized, 0.0, 1.0)
    return output


def detect_suspicious_players(player_stats: pd.DataFrame, raw_match_df: pd.DataFrame) -> pd.DataFrame:
    """Combine rule-based and machine-learned signals into a final suspicion score."""
    rule_df = compute_rule_scores(player_stats, raw_match_df)
    stat_df = compute_statistical_scores(player_stats)
    ml_df = compute_ml_scores(player_stats)

    merged = player_stats[["player_id"]].copy()
    merged = merged.merge(rule_df, on="player_id", how="left")
    merged = merged.merge(stat_df, on="player_id", how="left")
    merged = merged.merge(ml_df, on="player_id", how="left")
    merged = merged.fillna({"rule_score": 0.0, "statistical_score": 0.0, "ml_score": 0.0, "reasons": ""})
    merged["suspicion_score"] = (
        merged["rule_score"] * 0.4
        + merged["statistical_score"] * 0.3
        + merged["ml_score"] * 0.3
    )
    merged["suspicion_score"] = np.clip(merged["suspicion_score"], 0.0, 1.0)
    merged["is_suspicious"] = merged["suspicion_score"] >= 0.6
    return merged[["player_id", "suspicion_score", "is_suspicious", "reasons"]]


def save_suspicious_players(suspicion_df: pd.DataFrame, output_path: Path) -> None:
    """Write suspicious player details to CSV."""
    suspicion_df.to_csv(output_path, index=False)
    logging.info("Saved suspicious player report to %s", output_path)


def filter_suspicious_players(player_stats: pd.DataFrame, suspicion_df: pd.DataFrame) -> pd.DataFrame:
    """Remove suspicious players from the active player list."""
    suspicious_ids = suspicion_df.loc[suspicion_df["is_suspicious"], "player_id"].tolist()
    return player_stats[~player_stats["player_id"].isin(suspicious_ids)].copy()
