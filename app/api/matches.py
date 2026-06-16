from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.schemas.requests import PipelineRunRequest
from app.schemas.responses import PipelineRunResponse
from app.services.pipeline import run_pipeline

router = APIRouter(prefix="", tags=["pipeline"])


@router.post("/pipeline/run", response_model=PipelineRunResponse)
async def run_complete_pipeline(request: PipelineRunRequest) -> PipelineRunResponse:
    base_dir = Path(__file__).resolve().parents[2]
    try:
        result = run_pipeline(base_dir, input_path=Path(request.input_path) if request.input_path else None,
                              output_path=Path(request.output_path) if request.output_path else None)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
    return PipelineRunResponse(
        message="Pipeline executed successfully",
        players_processed=result["players_processed"],
        suspicious_players=result["suspicious_players"],
        matchmaking_groups=result["matchmaking_groups"],
    )
