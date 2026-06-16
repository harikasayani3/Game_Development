from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from app.schemas.responses import MatchmakingEntry

router = APIRouter(prefix="", tags=["matchmaking"])


def _load_matchmaking(base_dir: Path) -> pd.DataFrame:
    matchmaking_path = base_dir / "outputs" / "matchmaking.csv"
    if not matchmaking_path.exists():
        raise FileNotFoundError("Matchmaking file not found. Run the pipeline first.")
    return pd.read_csv(matchmaking_path)


@router.get("/matchmaking", response_model=list[MatchmakingEntry])
async def get_matchmaking(region: str | None = Query(None), skill_tier: str | None = Query(None)) -> list[MatchmakingEntry]:
    base_dir = Path(__file__).resolve().parents[2]
    try:
        matchmaking_df = _load_matchmaking(base_dir)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))

    if region:
        matchmaking_df = matchmaking_df[matchmaking_df["region"].str.lower() == region.lower()]
    if skill_tier:
        matchmaking_df = matchmaking_df[matchmaking_df["skill_tier"].str.lower() == skill_tier.lower()]

    return [MatchmakingEntry(**row) for row in matchmaking_df.to_dict(orient="records")]
