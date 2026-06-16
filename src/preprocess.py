from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .utils import ensure_directory

NUMERIC_COLUMNS = [
    "ping",
    "score",
    "kills",
    "deaths",
    "match_duration_seconds",
]

STANDARD_REGION: Dict[str, str] = {
    "india": "India",
    "in": "India",
    "india": "India",
    "sea": "SEA",
    "europe": "Europe",
    "eu": "Europe",
    "north america": "North America",
    "na": "North America",
    "latin america": "Latin America",
    "latam": "Latin America",
}

STANDARD_DEVICE: Dict[str, str] = {
    "android": "Android",
    "ios": "iOS",
    "pc": "PC",
}


def generate_synthetic_matches(csv_path: Path, record_count: int = 5000) -> pd.DataFrame:
    """Generate a synthetic match dataset and save it to CSV."""
    logging.info("Generating synthetic dataset with %d records", record_count)
    rng = np.random.default_rng(42)
    regions = ["India", "SEA", "Europe", "North America", "Latin America"]
    devices = ["Android", "iOS", "PC"]
    players = [f"P{num:04d}" for num in range(1, 1200)]
    matches = [f"M{num:04d}" for num in range(1, 3000)]
    rows: List[dict] = []

    for row_index in range(record_count):
        player_id = rng.choice(players)
        match_id = rng.choice(matches)
        region = rng.choice(regions, p=[0.3, 0.2, 0.2, 0.2, 0.1])
        device = rng.choice(devices, p=[0.45, 0.35, 0.2])
        match_duration_seconds = int(np.clip(rng.normal(420, 80), 60, 900))
        kills = int(np.clip(rng.normal(10, 6), 0, 70))
        deaths = int(np.clip(rng.normal(8, 5), 0, 40))
        score = int(np.clip(kills * 140 + (match_duration_seconds / 60) * 50 + rng.normal(0, 200), 0, None))
        ping = int(np.clip(rng.normal(80, 35), 20, 250))

        rows.append(
            {
                "player_id": player_id,
                "match_id": match_id,
                "region": region,
                "device": device,
                "ping": ping,
                "score": score,
                "kills": kills,
                "deaths": deaths,
                "match_duration_seconds": match_duration_seconds,
            }
        )

    # Add intentionally suspicious players
    suspicious_players = ["P0500", "P0600", "P0700", "P0800"]
    for player_id in suspicious_players:
        for suspicious_match in range(3):
            match_id = f"MSP{player_id[-3:]}-{suspicious_match}"
            rows.append(
                {
                    "player_id": player_id,
                    "match_id": match_id,
                    "region": rng.choice(regions),
                    "device": rng.choice(devices),
                    "ping": int(np.clip(rng.normal(40, 10), 15, 70)),
                    "score": int(9000 + suspicious_match * 2000),
                    "kills": int(110 + suspicious_match * 5),
                    "deaths": int(np.clip(rng.normal(2, 2), 0, 8)),
                    "match_duration_seconds": int(np.clip(rng.normal(25, 5), 10, 35)),
                }
            )

    df = pd.DataFrame(rows)
    ensure_directory(csv_path.parent)
    df.to_csv(csv_path, index=False)
    logging.info("Saved synthetic dataset to %s", csv_path)
    return df


def load_match_data(csv_path: Path) -> pd.DataFrame:
    """Load raw match data and generate synthetic data when necessary."""
    if not csv_path.exists():
        logging.warning("Input data file %s does not exist. Generating synthetic dataset.", csv_path)
        return generate_synthetic_matches(csv_path)

    logging.info("Loading match data from %s", csv_path)
    return pd.read_csv(csv_path)


def standardize_text(value: str, lookup: Dict[str, str]) -> str:
    """Standardize a text value using a lookup dictionary."""
    if not isinstance(value, str):
        return "Unknown"
    normalized = value.strip().lower()
    return lookup.get(normalized, value.strip().title())


def validate_numeric_columns(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    """Validate and coerce numeric columns to proper types."""
    for column in numeric_columns:
        if column not in df.columns:
            logging.error("Missing expected numeric column: %s", column)
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")
        missing_count = int(df[column].isna().sum())
        if missing_count > 0:
            median_value = int(df[column].median(skipna=True)) if not df[column].dropna().empty else 0
            logging.warning(
                "Filling %d missing values in %s with median value %s", missing_count, column, median_value
            )
            df[column] = df[column].fillna(median_value)
    return df


def preprocess_match_data(csv_path: Path) -> pd.DataFrame:
    """Load and clean raw match data for the pipeline."""
    df = load_match_data(csv_path)
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    if duplicates_removed > 0:
        logging.info("Removed %d duplicate rows", duplicates_removed)

    required_columns = [
        "player_id",
        "match_id",
        "region",
        "device",
        "ping",
        "score",
        "kills",
        "deaths",
        "match_duration_seconds",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df["player_id"] = df["player_id"].astype(str).str.strip()
    df["match_id"] = df["match_id"].astype(str).str.strip()
    df["region"] = df["region"].apply(lambda value: standardize_text(str(value), STANDARD_REGION))
    df["device"] = df["device"].apply(lambda value: standardize_text(str(value), STANDARD_DEVICE))
    df = validate_numeric_columns(df, NUMERIC_COLUMNS)

    df["player_id"] = df["player_id"].replace({"": "Unknown"})
    df = df.dropna(subset=["player_id", "match_id"])
    df = df.reset_index(drop=True)
    logging.info("Preprocessed %d rows", len(df))
    return df
