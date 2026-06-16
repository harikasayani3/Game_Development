from __future__ import annotations

import logging
from fastapi import FastAPI

from app.api.anomaly import router as anomaly_router
from app.api.leaderboard import router as leaderboard_router
from app.api.matchmaking import router as matchmaking_router
from app.api.matches import router as matches_router
from app.services.utils import setup_logging


def create_app() -> FastAPI:
    setup_logging("INFO")
    app = FastAPI(
        title="Game Ops AI System API",
        description="Backend service for leaderboard, matchmaking, and suspicious player detection.",
        version="1.0.0",
    )
    app.include_router(matches_router)
    app.include_router(leaderboard_router)
    app.include_router(anomaly_router)
    app.include_router(matchmaking_router)
    return app


app = create_app()


@app.get("/", response_model=dict)
async def health_check() -> dict[str, str]:
    return {"status": "running"}
