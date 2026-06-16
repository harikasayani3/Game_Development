from __future__ import annotations

import pandas as pd

from src.skill_prediction import predict_player_skills as _predict_player_skills


def predict_player_skills(player_stats: pd.DataFrame) -> pd.DataFrame:
    return _predict_player_skills(player_stats)
