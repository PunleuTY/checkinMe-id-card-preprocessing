import logging
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.schemas.preprocess import TimingInfo, GeminiOCRResponse
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/gemini-ocr/upload", response_model=GeminiOCRResponse)
async def gemini_ocr_upload(
    file: UploadFile = File(...),
    model: str = Form(settings.gemini_model),
) -> GeminiOCRResponse:
    from gemini.extractor import extract_from_bytes

    t0 = time.perf_counter()
    data = await file.read()
    t1 = time.perf_counter()

    try:
        result = await run_in_threadpool(
            extract_from_bytes, data, file.content_type, model or None
        )
    except ImportError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("gemini extraction failed")
        raise HTTPException(status_code=502, detail=f"Gemini error: {e}")

    t2 = time.perf_counter()

    def _ms(a, b):
        return round((b - a) * 1000, 2)

    return GeminiOCRResponse(
        text=result["text"],
        fields=result["fields"],
        model=result["model"],
        timing=TimingInfo(
            total_ms=_ms(t0, t2),
            details={
                "upload_read_ms": _ms(t0, t1),
                "gemini_api_ms": _ms(t1, t2),
            },
        ),
    )
