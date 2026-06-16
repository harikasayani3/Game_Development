from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


def compute_match_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Create match-level derived metrics for each row."""
    result = df.copy()
    result["match_duration_minutes"] = result["match_duration_seconds"] / 60
    result["kd_ratio"] = result["kills"] / (result["deaths"] + 1)
    result["score_per_min"] = result["score"] / result["match_duration_minutes"].replace(0, 1)
    result["kills_per_min"] = result["kills"] / result["match_duration_minutes"].replace(0, 1)
    logging.info("Computed match-level metrics for %d rows", len(result))
    return result


def _most_frequent_value(series: pd.Series) -> str:
    """Return the most frequent non-null value in a pandas Series."""
    if series.empty:
        return "Unknown"
    modes = series.mode(dropna=True)
    if not modes.empty:
        return str(modes.iloc[0])
    return str(series.iloc[0])


def aggregate_player_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate match-level data into player-level statistics."""
    player_metrics = df.groupby("player_id", as_index=False).agg(
        average_score=("score", "mean"),
        average_kills=("kills", "mean"),
        average_deaths=("deaths", "mean"),
        average_ping=("ping", "mean"),
        average_kd_ratio=("kd_ratio", "mean"),
        average_score_per_min=("score_per_min", "mean"),
        average_kills_per_min=("kills_per_min", "mean"),
        average_match_duration_seconds=("match_duration_seconds", "mean"),
        total_matches=("match_id", "nunique"),
    )

    region_device = df.groupby("player_id", as_index=False).agg(
        region=("region", _most_frequent_value),
        device=("device", _most_frequent_value),
    )

    metrics = player_metrics.merge(region_device, on="player_id", how="left")
    metrics["total_matches"] = metrics["total_matches"].fillna(0).astype(int)
    logging.info("Aggregated player statistics for %d players", len(metrics))
    return metrics


def create_player_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Process raw match data and return player-level feature matrix."""
    enriched_df = compute_match_metrics(df)
    player_stats = aggregate_player_statistics(enriched_df)
    required_columns = [
        "player_id",
        "region",
        "device",
        "average_score",
        "average_kills",
        "average_deaths",
        "average_ping",
        "average_kd_ratio",
        "average_score_per_min",
        "average_kills_per_min",
        "total_matches",
    ]
    missing_columns = [column for column in required_columns if column not in player_stats.columns]
    if missing_columns:
        raise ValueError(f"Player statistics missing required columns: {missing_columns}")
    return player_stats
