from fastapi import APIRouter, HTTPException
from app.schemas.preprocess import PreprocessRequest, PreprocessResponse
from app.utils.image import b64_to_ndarray, ndarray_to_b64, ndarray_to_webp_b64
from app.core.config import settings
from app.pipeline.perspective_transformation import perspective_correct
from app.pipeline.img_segment import segment_card
from app.pipeline.normalize import normalize_resolution
from app.pipeline.watermark import apply_watermark

router = APIRouter()


@router.post("/preprocess", response_model=PreprocessResponse)
async def preprocess(payload: PreprocessRequest) -> PreprocessResponse:
    try:
        img = b64_to_ndarray(payload.image)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Step 1: perspective correction
    img = perspective_correct(img)

    # Step 2: segment card (remove warp black corners)
    img = segment_card(img)

    # Step 3: normalize to standard output resolution
    img = normalize_resolution(img, settings.output_width, settings.output_height)

    # Snapshot clean card before watermark (for internal/debug use)
    segmented_b64 = ndarray_to_b64(img)

    # Step 4: apply watermark → final result sent to camDC
    final = apply_watermark(img, settings.watermark_text, settings.watermark_opacity)
    processed_b64 = ndarray_to_webp_b64(final, quality=settings.webp_quality)

    return PreprocessResponse(
        segmented_image=segmented_b64,
        processed_image=processed_b64,
    )
