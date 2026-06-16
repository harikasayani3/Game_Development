from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.matchmaking import create_matchmaking_groups as _create_matchmaking_groups
from src.matchmaking import save_matchmaking as _save_matchmaking


def create_matchmaking_groups(player_stats: pd.DataFrame) -> pd.DataFrame:
    return _create_matchmaking_groups(player_stats)


def save_matchmaking(matchmaking_df: pd.DataFrame, output_path: Path) -> None:
    _save_matchmaking(matchmaking_df, output_path)
