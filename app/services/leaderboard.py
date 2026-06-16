from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.leaderboard import create_leaderboard as _create_leaderboard
from src.leaderboard import save_leaderboard as _save_leaderboard


def create_leaderboard(player_stats: pd.DataFrame) -> pd.DataFrame:
    return _create_leaderboard(player_stats)


def save_leaderboard(leaderboard_df: pd.DataFrame, output_path: Path) -> None:
    _save_leaderboard(leaderboard_df, output_path)
