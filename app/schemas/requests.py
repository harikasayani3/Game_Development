from __future__ import annotations

from pydantic import BaseModel


class PipelineRunRequest(BaseModel):
    input_path: str | None = None
    output_path: str | None = None
