from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from app.schemas.responses import LeaderboardEntry

router = APIRouter(prefix="", tags=["leaderboard"])


def _load_leaderboard(base_dir: Path) -> pd.DataFrame:
    leaderboard_path = base_dir / "outputs" / "leaderboard.csv"
    if not leaderboard_path.exists():
        raise FileNotFoundError("Leaderboard file not found. Run the pipeline first.")
    return pd.read_csv(leaderboard_path)


@router.get("/leaderboard", response_model=list[LeaderboardEntry])
async def get_leaderboard(region: str | None = Query(None), limit: int | None = Query(10)) -> list[LeaderboardEntry]:
    base_dir = Path(__file__).resolve().parents[2]
    try:
        leaderboard_df = _load_leaderboard(base_dir)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))

    if region:
        leaderboard_df = leaderboard_df[leaderboard_df["region"].str.lower() == region.lower()]
    if limit is not None:
        leaderboard_df = leaderboard_df.head(limit)

    return [LeaderboardEntry(**row) for row in leaderboard_df.to_dict(orient="records")]
