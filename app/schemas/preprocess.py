from typing import Any

from pydantic import BaseModel


class TimingInfo(BaseModel):
    """Wall-clock milliseconds for each processing step."""
    total_ms: float
    details: dict[str, float] = {}


class GeminiOCRResponse(BaseModel):
    text: str
    fields: dict[str, Any]
    model: str
    timing: TimingInfo | None = None
