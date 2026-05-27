import logging
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from app.schemas.preprocess import (
    TimingInfo,
    PreprocessRequest,
    PreprocessResponse,
    PreprocessOCRRequest,
    PreprocessOCRResponse,
    GeminiOCRRequest,
    GeminiOCRResponse,
)
from app.utils.image import (
    b64_to_bytes,
    b64_to_ndarray,
    bytes_to_ndarray,
    ndarray_to_b64,
    ndarray_to_webp_b64,
)
from app.core.config import settings
from app.pipeline.perspective_transformation import perspective_correct
from app.pipeline.img_segment import segment_card
from app.pipeline.normalize import normalize_resolution

# from app.pipeline.watermark import apply_watermark

logger = logging.getLogger(__name__)

router = APIRouter()


def _run_pipeline(img):
    img = perspective_correct(img)
    img = segment_card(img)
    img = normalize_resolution(img, settings.output_width, settings.output_height)
    return img


def _preprocess_image(image_b64: str):
    try:
        img = b64_to_ndarray(image_b64)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return _run_pipeline(img)


# @router.post("/preprocess", response_model=PreprocessResponse)
# async def preprocess(payload: PreprocessRequest) -> PreprocessResponse:
#     img = _preprocess_image(payload.image)
#     processed_b64 = ndarray_to_webp_b64(img, quality=settings.webp_quality)
#     return PreprocessResponse(
#         segmented_image=ndarray_to_b64(img),
#         processed_image=processed_b64,
#     )


@router.post("/preprocess/upload", response_model=PreprocessResponse)
async def preprocess_upload(file: UploadFile = File(...)) -> PreprocessResponse:
    try:
        img = bytes_to_ndarray(await file.read())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    img = perspective_correct(img)
    img = segment_card(img)
    img = normalize_resolution(img, settings.output_width, settings.output_height)
    processed_b64 = ndarray_to_webp_b64(img, quality=settings.webp_quality)
    return PreprocessResponse(
        segmented_image=ndarray_to_b64(img),
        processed_image=processed_b64,
    )


@router.post("/preprocess-ocr/upload", response_model=PreprocessOCRResponse)
async def preprocess_ocr_upload(
    file: UploadFile = File(...),
    decode_method: str = Form("accurate"),
) -> PreprocessOCRResponse:
    from experiments.ocr.extractor import extract_from_array

    try:
        img = bytes_to_ndarray(await file.read())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    img = perspective_correct(img)
    img = segment_card(img)
    img = normalize_resolution(img, settings.output_width, settings.output_height)
    processed_b64 = ndarray_to_webp_b64(img, quality=settings.webp_quality)

    try:
        ocr_result = extract_from_array(img, decode_method=decode_method)
    except ImportError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PreprocessOCRResponse(
        segmented_image=ndarray_to_b64(img),
        processed_image=processed_b64,
        text=ocr_result["text"],
        lines=ocr_result["lines"],
    )


@router.post("/gemini-ocr/upload", response_model=GeminiOCRResponse)
async def gemini_ocr_upload(
    file: UploadFile = File(...),
    model: str = Form(settings.gemini_model),
) -> GeminiOCRResponse:
    from experiments.gemini.extractor import extract_from_bytes

    data = await file.read()
    try:
        result = await run_in_threadpool(
            extract_from_bytes, data, file.content_type, model or None
        )
    except ImportError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("gemini extraction failed")
        raise HTTPException(status_code=502, detail=f"Gemini error: {e}")

    return GeminiOCRResponse(
        text=result["text"], fields=result["fields"], model=result["model"]
    )


# @router.post("/gemini-ocr", response_model=GeminiOCRResponse)
# async def gemini_ocr(payload: GeminiOCRRequest) -> GeminiOCRResponse:
#     from experiments.gemini.extractor import extract_from_bytes

#     try:
#         data = b64_to_bytes(payload.image)
#     except ValueError as e:
#         raise HTTPException(status_code=422, detail=str(e))

#     try:
#         result = await run_in_threadpool(extract_from_bytes, data, None, payload.model)
#     except ImportError as e:
#         raise HTTPException(status_code=500, detail=str(e))
#     except Exception as e:
#         logger.exception("gemini extraction failed")
#         raise HTTPException(status_code=502, detail=f"Gemini error: {e}")

#     return GeminiOCRResponse(
#         text=result["text"], fields=result["fields"], model=result["model"]
#     )


# @router.post("/preprocess-ocr", response_model=PreprocessOCRResponse)
# async def preprocess_ocr(payload: PreprocessOCRRequest) -> PreprocessOCRResponse:
#     from experiments.ocr.extractor import extract_from_array

#     img = _preprocess_image(payload.image)
#     processed_b64 = ndarray_to_webp_b64(img, quality=settings.webp_quality)

#     try:
#         ocr_result = extract_from_array(img, decode_method=payload.decode_method)
#     except ImportError as e:
#         raise HTTPException(status_code=500, detail=str(e))

#     return PreprocessOCRResponse(
#         segmented_image=ndarray_to_b64(img),
#         processed_image=processed_b64,
#         text=ocr_result["text"],
#         lines=ocr_result["lines"],
#     )
