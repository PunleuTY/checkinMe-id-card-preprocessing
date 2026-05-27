from typing import Any, Literal

from pydantic import BaseModel, field_validator


class PreprocessRequest(BaseModel):
    image: str  # raw base64-encoded image (no data URI prefix required)

    @field_validator("image")
    @classmethod
    def strip_data_uri(cls, v: str) -> str:
        if "," in v:
            v = v.split(",", 1)[1]
        return v


class TimingInfo(BaseModel):
    """Wall-clock milliseconds for each processing step."""
    total_ms: float
    details: dict[str, float] = {}


class PreprocessResponse(BaseModel):
    segmented_image: str  # base64 — for CamDX OCR
    processed_image: str  # base64 — for storage (normalised + WebP + watermark)
    timing: TimingInfo | None = None


class PreprocessOCRRequest(PreprocessRequest):
    decode_method: Literal["fast", "accurate", "beam"] = "accurate"


class PreprocessOCRResponse(PreprocessResponse):
    text: str
    lines: list[dict[str, Any]]


class GeminiOCRRequest(PreprocessRequest):
    model: str | None = None  # override settings.gemini_model


class GeminiOCRResponse(BaseModel):
    text: str
    fields: dict[str, Any]
    model: str
    timing: TimingInfo | None = None


class GeminiOCRAnnotatedResponse(GeminiOCRResponse):
    regions: dict[str, Any]          # field → [y_min, x_min, y_max, x_max] or null
    annotated_image: str             # base64 JPEG with colored bounding boxes drawn


class QualityCheck(BaseModel):
    valid: bool
    aspect_ratio: float
    mrz_lines_found: int
    key_fields_found: int
    reason: str                      # "OK" or a description of what failed


class GeminiOCRProcessedResponse(GeminiOCRAnnotatedResponse):
    cleaned_image: str               # base64 JPEG — de-skewed, background removed, NO boxes
    quality_check: QualityCheck
    saved_as: str                    # relative path written to disk (always set)
    saved_type: Literal["cleaned", "original"]  # "cleaned" if quality passed, "original" as fallback
