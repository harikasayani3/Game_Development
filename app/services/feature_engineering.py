from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.feature_engineering import create_player_statistics as _create_player_statistics


def create_player_statistics(df: pd.DataFrame) -> pd.DataFrame:
    return _create_player_statistics(df)
