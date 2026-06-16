from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.anomaly_detection import (
    detect_suspicious_players as _detect_suspicious_players,
    save_suspicious_players as _save_suspicious_players,
)


def detect_suspicious_players(player_stats: pd.DataFrame, raw_match_df: pd.DataFrame) -> pd.DataFrame:
    return _detect_suspicious_players(player_stats, raw_match_df)


def save_suspicious_players(suspicion_df: pd.DataFrame, output_path: Path) -> None:
    _save_suspicious_players(suspicion_df, output_path)
