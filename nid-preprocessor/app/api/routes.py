from fastapi import APIRouter, HTTPException
from app.schemas.preprocess import PreprocessRequest, PreprocessResponse
from app.utils.image import b64_to_ndarray, ndarray_to_b64, ndarray_to_webp_b64
from app.core.config import settings

router = APIRouter()


@router.post("/preprocess", response_model=PreprocessResponse)
async def preprocess(payload: PreprocessRequest) -> PreprocessResponse:
    try:
        img = b64_to_ndarray(payload.image)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # --- pipeline steps (filled in one by one) ---
    # Step 2: perspective_correct(img)
    # Step 3: segment_card(img)
    # segmented_image = result of step 3
    # Step 4: normalize_resolution(segmented_image)
    # Step 5: compress to WebP
    # Step 6: apply_watermark(...)
    # processed_image = result of step 6

    segmented_b64 = ndarray_to_b64(img)       # placeholder
    processed_b64 = ndarray_to_webp_b64(img, quality=settings.webp_quality)  # placeholder

    return PreprocessResponse(
        segmented_image=segmented_b64,
        processed_image=processed_b64,
    )
