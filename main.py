from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.anomaly_detection import (
    detect_suspicious_players,
    save_suspicious_players,
)
from src.feature_engineering import create_player_statistics
from src.leaderboard import create_leaderboard, save_leaderboard
from src.matchmaking import create_matchmaking_groups, save_matchmaking
from src.preprocess import preprocess_match_data
from src.skill_prediction import predict_player_skills
from src.utils import format_table, ensure_directory, setup_logging


def display_banner() -> None:
    print("===================================================")
    print("🎮 GAME OPS AI SYSTEM")
    print("===================================================")


def display_menu() -> None:
    print("\n1. Run Complete Pipeline")
    print("2. View Top 10 Leaderboard")
    print("3. View Suspicious Players")
    print("4. View Matchmaking Groups")
    print("5. Exit")


def print_status(message: str) -> None:
    print(f"✓ {message}")


def run_complete_pipeline(base_dir: Path) -> dict[str, Any]:
    data_path = base_dir / "data" / "sample_matches.csv"
    outputs_dir = ensure_directory(base_dir / "outputs")

    raw_data = preprocess_match_data(data_path)
    print_status(f"Loaded {len(raw_data)} records")

    player_stats = create_player_statistics(raw_data)
    print_status("Created player features")

    skill_stats = predict_player_skills(player_stats)
    print_status("Predicted skill tiers")

    suspicion_df = detect_suspicious_players(skill_stats, raw_data)
    save_suspicious_players(suspicion_df, outputs_dir / "suspicious_players.csv")
    print_status("Detected suspicious players")

    eligible_players = skill_stats.merge(suspicion_df, on="player_id")
    eligible_players = eligible_players[eligible_players["is_suspicious"] == False]
    leaderboard_df = create_leaderboard(eligible_players)
    save_leaderboard(leaderboard_df, outputs_dir / "leaderboard.csv")
    print_status("Generated leaderboard")

    matchmaking_df = create_matchmaking_groups(eligible_players)
    save_matchmaking(matchmaking_df, outputs_dir / "matchmaking.csv")
    print_status("Created matchmaking groups")

    print_status("Saved outputs")
    return {
        "raw_data": raw_data,
        "player_stats": player_stats,
        "skill_stats": skill_stats,
        "suspicion_df": suspicion_df,
        "leaderboard_df": leaderboard_df,
        "matchmaking_df": matchmaking_df,
    }


def view_top_leaderboard(base_dir: Path) -> None:
    leaderboard_path = base_dir / "outputs" / "leaderboard.csv"
    if not leaderboard_path.exists():
        print("Leaderboard not found. Please run the pipeline first.")
        return
    import pandas as pd

    leaderboard_df = pd.read_csv(leaderboard_path)
    print("Top 10 Leaderboard")
    print("-------------------")
    print(format_table(leaderboard_df.head(10), headers="keys"))


def view_suspicious_players(base_dir: Path) -> None:
    suspicious_path = base_dir / "outputs" / "suspicious_players.csv"
    if not suspicious_path.exists():
        print("Suspicious players file not found. Please run the pipeline first.")
        return
    import pandas as pd

    suspicious_df = pd.read_csv(suspicious_path)
    print("Suspicious Players")
    print("-------------------")
    print(format_table(suspicious_df.head(10), headers="keys"))


def view_matchmaking_groups(base_dir: Path) -> None:
    matchmaking_path = base_dir / "outputs" / "matchmaking.csv"
    if not matchmaking_path.exists():
        print("Matchmaking file not found. Please run the pipeline first.")
        return
    import pandas as pd

    matchmaking_df = pd.read_csv(matchmaking_path)
    print("Matchmaking Groups")
    print("-------------------")
    print(format_table(matchmaking_df.head(10), headers="keys"))


def main() -> None:
    setup_logging("INFO")
    base_dir = Path(__file__).resolve().parent
    display_banner()

    while True:
        display_menu()
        choice = input("Select an option: ").strip()
        if choice == "1":
            try:
                run_complete_pipeline(base_dir)
            except Exception as error:
                logging.exception("Pipeline failed: %s", error)
                print("An error occurred while running the pipeline. Check logs for details.")
        elif choice == "2":
            view_top_leaderboard(base_dir)
        elif choice == "3":
            view_suspicious_players(base_dir)
        elif choice == "4":
            view_matchmaking_groups(base_dir)
        elif choice == "5":
            print("Exiting Game Ops AI System.")
            break
        else:
            print("Invalid choice. Enter 1-5.")


if __name__ == "__main__":
    main()
