from pydantic import BaseModel, field_validator
import base64


class PreprocessRequest(BaseModel):
    image: str  # raw base64-encoded image (no data URI prefix required)

    @field_validator("image")
    @classmethod
    def strip_data_uri(cls, v: str) -> str:
        if "," in v:
            v = v.split(",", 1)[1]
        return v


class PreprocessResponse(BaseModel):
    segmented_image: str  # base64 — for CamDX OCR
    processed_image: str  # base64 — for storage (normalised + WebP + watermark)
