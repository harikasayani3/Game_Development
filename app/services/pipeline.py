from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from app.services.feature_engineering import create_player_statistics
from app.services.leaderboard import create_leaderboard, save_leaderboard
from app.services.matchmaking import create_matchmaking_groups, save_matchmaking
from app.services.preprocess import preprocess_match_data
from app.services.skill_prediction import predict_player_skills
from app.services.anomaly_detection import detect_suspicious_players, save_suspicious_players
from app.services.utils import ensure_directory


def run_pipeline(base_dir: Path, input_path: Path | None = None, output_path: Path | None = None) -> dict[str, int]:
    data_path = input_path or base_dir / "data" / "sample_matches.csv"
    output_dir = ensure_directory(output_path or base_dir / "outputs")

    raw_data = preprocess_match_data(data_path)
    player_stats = create_player_statistics(raw_data)
    skill_stats = predict_player_skills(player_stats)
    suspicion_df = detect_suspicious_players(skill_stats, raw_data)

    suspicious_count = int(suspicion_df["is_suspicious"].sum())
    eligible_players = skill_stats.merge(suspicion_df, on="player_id")
    eligible_players = eligible_players[eligible_players["is_suspicious"] == False]

    leaderboard_df = create_leaderboard(eligible_players)
    save_leaderboard(leaderboard_df, output_dir / "leaderboard.csv")

    matchmaking_df = create_matchmaking_groups(eligible_players)
    save_matchmaking(matchmaking_df, output_dir / "matchmaking.csv")

    save_suspicious_players(suspicion_df, output_dir / "suspicious_players.csv")

    logging.info("Pipeline executed: %s players", len(player_stats))
    return {
        "players_processed": len(player_stats),
        "suspicious_players": suspicious_count,
        "matchmaking_groups": matchmaking_df["group_id"].nunique(),
    }
