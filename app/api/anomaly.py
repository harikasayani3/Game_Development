from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.schemas.responses import SuspiciousPlayerResponse

router = APIRouter(prefix="", tags=["anomaly"])


def _load_suspicious_players(base_dir: Path) -> pd.DataFrame:
    suspicious_path = base_dir / "outputs" / "suspicious_players.csv"
    if not suspicious_path.exists():
        raise FileNotFoundError("Suspicious players file not found. Run the pipeline first.")
    return pd.read_csv(suspicious_path)


@router.get("/suspicious-players", response_model=list[SuspiciousPlayerResponse])
async def get_suspicious_players() -> list[SuspiciousPlayerResponse]:
    base_dir = Path(__file__).resolve().parents[2]
    try:
        suspicious_df = _load_suspicious_players(base_dir)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))

    # Filter to only return players marked as suspicious
    suspicious_df = suspicious_df[suspicious_df["is_suspicious"] == True]
    
    return [
        SuspiciousPlayerResponse(
            player_id=row["player_id"],
            suspicion_score=float(row["suspicion_score"]),
            reasons=row["reasons"],
        )
        for row in suspicious_df.to_dict(orient="records")
    ]
