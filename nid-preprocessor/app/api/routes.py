from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from app.schemas.preprocess import (
    PreprocessRequest,
    PreprocessResponse,
    PreprocessOCRRequest,
    PreprocessOCRResponse,
)
from app.utils.image import b64_to_ndarray, bytes_to_ndarray, ndarray_to_b64, ndarray_to_webp_b64
from app.core.config import settings
from app.pipeline.perspective_transformation import perspective_correct
from app.pipeline.img_segment import segment_card
from app.pipeline.normalize import normalize_resolution
# from app.pipeline.watermark import apply_watermark

router = APIRouter()


def _preprocess_image(image_b64: str):
    try:
        img = b64_to_ndarray(image_b64)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    img = perspective_correct(img)
    img = segment_card(img)
    img = normalize_resolution(img, settings.output_width, settings.output_height)
    return img


@router.post("/preprocess", response_model=PreprocessResponse)
async def preprocess(payload: PreprocessRequest) -> PreprocessResponse:
    img = _preprocess_image(payload.image)
    processed_b64 = ndarray_to_webp_b64(img, quality=settings.webp_quality)
    return PreprocessResponse(
        segmented_image=ndarray_to_b64(img),
        processed_image=processed_b64,
    )


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


@router.post("/preprocess-ocr", response_model=PreprocessOCRResponse)
async def preprocess_ocr(payload: PreprocessOCRRequest) -> PreprocessOCRResponse:
    from experiments.ocr.extractor import extract_from_array

    img = _preprocess_image(payload.image)
    processed_b64 = ndarray_to_webp_b64(img, quality=settings.webp_quality)

    try:
        ocr_result = extract_from_array(img, decode_method=payload.decode_method)
    except ImportError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PreprocessOCRResponse(
        segmented_image=ndarray_to_b64(img),
        processed_image=processed_b64,
        text=ocr_result["text"],
        lines=ocr_result["lines"],
    )
