from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


def create_leaderboard(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Build a leaderboard sorted by skill and performance metrics."""
    leaderboard = player_stats.copy()
    leaderboard = leaderboard.drop_duplicates(subset=["player_id"])
    leaderboard = leaderboard.sort_values(
        by=["skill_score", "average_kd_ratio", "average_deaths", "average_ping"],
        ascending=[False, False, True, True],
        ignore_index=True,
    )
    leaderboard["global_rank"] = leaderboard.index + 1
    leaderboard["region_rank"] = leaderboard.groupby("region").cumcount() + 1
    logging.info("Created leaderboard for %d players", len(leaderboard))
    return leaderboard[
        [
            "global_rank",
            "region_rank",
            "player_id",
            "region",
            "skill_score",
            "skill_tier",
            "average_kd_ratio",
            "average_deaths",
            "average_ping",
        ]
    ]


def save_leaderboard(leaderboard_df: pd.DataFrame, output_path: Path) -> None:
    """Save the leaderboard data to disk."""
    leaderboard_df.to_csv(output_path, index=False)
    logging.info("Saved leaderboard to %s", output_path)
