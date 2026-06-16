from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def assign_skill_tier(score: float) -> str:
    """Assign a skill tier based on a normalized skill score."""
    if score < 0.2:
        return "Bronze"
    if score < 0.4:
        return "Silver"
    if score < 0.6:
        return "Gold"
    if score < 0.8:
        return "Platinum"
    return "Diamond"


def predict_player_skills(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Compute a normalized skill score and assign tiers for each player."""
    if player_stats.empty:
        return player_stats.copy()

    features = [
        "average_score",
        "average_kd_ratio",
        "average_kills_per_min",
        "total_matches",
        "average_ping",
    ]
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(player_stats[features])
    weights = np.array([0.35, 0.25, 0.2, 0.1, -0.1])
    raw_score = np.dot(normalized, weights)
    final_score = MinMaxScaler().fit_transform(raw_score.reshape(-1, 1)).reshape(-1)

    output = player_stats.copy()
    output["skill_score"] = np.clip(final_score, 0.0, 1.0)
    output["skill_tier"] = output["skill_score"].apply(assign_skill_tier)
    return output
