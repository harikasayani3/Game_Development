from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


def create_matchmaking_groups(player_stats: pd.DataFrame, target_group_size: int = 10) -> pd.DataFrame:
    """Create fair matchmaking groups using clustering by skill and ping."""
    players = player_stats.copy()
    players["group_id"] = ""
    players["fairness_score"] = 1.0
    group_frames: list[pd.DataFrame] = []

    for (region, device), group in players.groupby(["region", "device"]):
        group = group.reset_index(drop=True)
        if group.empty:
            continue

        unique_points = group[["skill_score", "average_ping"]].drop_duplicates()
        n_groups = max(1, int(round(len(group) / target_group_size)))
        n_groups = min(n_groups, len(unique_points), len(group))

        if n_groups == 1:
            group["cluster_label"] = 0
        else:
            kmeans = KMeans(n_clusters=n_groups, random_state=42)
            labels = kmeans.fit_predict(group[["skill_score", "average_ping"]])
            group["cluster_label"] = labels

        group_frames.append(group)

    grouped = pd.concat(group_frames, ignore_index=True) if group_frames else players
    fairness = grouped.groupby(["region", "device", "cluster_label"], as_index=False).agg(
        skill_variance=("skill_score", "var"),
    )
    fairness["skill_variance"] = fairness["skill_variance"].fillna(0.0)

    if fairness.empty:
        fairness = pd.DataFrame(
            columns=["region", "device", "cluster_label", "skill_variance", "fairness_score"]
        )
    else:
        alpha = 0.05
        fairness["fairness_score"] = alpha / (fairness["skill_variance"] + alpha)

    fairness = fairness[["region", "device", "cluster_label", "fairness_score"]]

    matchmaking_df = grouped.merge(fairness, on=["region", "device", "cluster_label"], how="left")
    if "fairness_score" not in matchmaking_df.columns:
        matchmaking_df["fairness_score"] = 1.0
    else:
        matchmaking_df["fairness_score"] = matchmaking_df["fairness_score"].fillna(1.0)
    matchmaking_df["group_id"] = (
        matchmaking_df["region"].str.replace(" ", "_")
        + "_"
        + matchmaking_df["device"].str.replace(" ", "_")
        + "_group_"
        + matchmaking_df["cluster_label"].astype(str)
    )
    result = matchmaking_df[
        ["group_id", "player_id", "region", "device", "skill_tier", "fairness_score"]
    ].copy()
    logging.info("Created matchmaking groups for %d players", len(result))
    return result


def save_matchmaking(matchmaking_df: pd.DataFrame, output_path: Path) -> None:
    """Save matchmaking group assignments to CSV."""
    matchmaking_df.to_csv(output_path, index=False)
    logging.info("Saved matchmaking data to %s", output_path)
