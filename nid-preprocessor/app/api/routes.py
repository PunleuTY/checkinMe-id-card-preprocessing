from fastapi import APIRouter, HTTPException
from app.schemas.preprocess import PreprocessRequest, PreprocessResponse
from app.utils.image import b64_to_ndarray, ndarray_to_b64, ndarray_to_webp_b64
from app.core.config import settings
from app.pipeline.perspective_transformation import perspective_correct

router = APIRouter()


@router.post("/preprocess", response_model=PreprocessResponse)
async def preprocess(payload: PreprocessRequest) -> PreprocessResponse:
    try:
        img = b64_to_ndarray(payload.image)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Step 1: perspective correction
    img = perspective_correct(img)

    # --- pipeline steps (filled in one by one) ---
    # Step 2: segment_card(img)
    # segmented_image = result of step 2
    # Step 3: normalize_resolution(segmented_image)
    # Step 4: compress to WebP
    # Step 5: apply_watermark(...)
    # processed_image = result of step 5

    segmented_b64 = ndarray_to_b64(img)
    processed_b64 = ndarray_to_webp_b64(img, quality=settings.webp_quality)

    return PreprocessResponse(
        segmented_image=segmented_b64,
        processed_image=processed_b64,
    )
