from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.preprocess import generate_synthetic_matches as _generate_synthetic_matches
from src.preprocess import preprocess_match_data as _preprocess_match_data


def generate_synthetic_matches(csv_path: Path, record_count: int = 5000) -> pd.DataFrame:
    return _generate_synthetic_matches(csv_path, record_count=record_count)


def preprocess_match_data(csv_path: Path) -> pd.DataFrame:
    return _preprocess_match_data(csv_path)
