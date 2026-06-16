from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class PipelineRunResponse(BaseModel):
    message: str
    players_processed: int
    suspicious_players: int
    matchmaking_groups: int


class LeaderboardEntry(BaseModel):
    global_rank: int
    region_rank: int
    player_id: str
    region: str
    skill_score: float
    skill_tier: str
    average_kd_ratio: float
    average_deaths: float
    average_ping: float


class SuspiciousPlayerResponse(BaseModel):
    player_id: str
    suspicion_score: float
    reasons: str


class MatchmakingEntry(BaseModel):
    group_id: str
    player_id: str
    region: str
    device: str
    skill_tier: str
    fairness_score: float
