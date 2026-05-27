import logging
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from app.schemas.preprocess import (
    TimingInfo,
    PreprocessRequest,
    PreprocessResponse,
    PreprocessOCRRequest,
    PreprocessOCRResponse,
    GeminiOCRRequest,
    GeminiOCRResponse,
    GeminiOCRAnnotatedResponse,
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
    t0 = time.perf_counter()

    try:
        img = bytes_to_ndarray(await file.read())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    t1 = time.perf_counter()
    img = perspective_correct(img)
    t2 = time.perf_counter()
    img = segment_card(img)
    t3 = time.perf_counter()
    img = normalize_resolution(img, settings.output_width, settings.output_height)
    t4 = time.perf_counter()
    processed_b64 = ndarray_to_webp_b64(img, quality=settings.webp_quality)
    t5 = time.perf_counter()

    def _ms(a, b): return round((b - a) * 1000, 2)

    return PreprocessResponse(
        segmented_image=ndarray_to_b64(img),
        processed_image=processed_b64,
        timing=TimingInfo(
            total_ms=_ms(t0, t5),
            details={
                "decode_image_ms": _ms(t0, t1),
                "perspective_correct_ms": _ms(t1, t2),
                "segment_card_ms": _ms(t2, t3),
                "normalize_ms": _ms(t3, t4),
                "encode_webp_ms": _ms(t4, t5),
            },
        ),
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

    def _ms(a, b): return round((b - a) * 1000, 2)

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


@router.post("/gemini-ocr/upload/annotated", response_model=GeminiOCRAnnotatedResponse)
async def gemini_ocr_upload_annotated(
    file: UploadFile = File(...),
    model: str = Form(settings.gemini_model),
) -> GeminiOCRAnnotatedResponse:
    """
    Same as /gemini-ocr/upload but also returns the detected field regions and
    an annotated image (base64 JPEG) with colored bounding boxes drawn on it.
    """
    from experiments.gemini.extractor import extract_with_regions_from_bytes

    t0 = time.perf_counter()
    data = await file.read()
    t1 = time.perf_counter()

    try:
        result = await run_in_threadpool(
            extract_with_regions_from_bytes, data, file.content_type, model or None
        )
    except ImportError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("gemini annotated extraction failed")
        raise HTTPException(status_code=502, detail=f"Gemini error: {e}")

    t2 = time.perf_counter()

    def _ms(a, b): return round((b - a) * 1000, 2)

    return GeminiOCRAnnotatedResponse(
        text=result["text"],
        fields=result["fields"],
        regions=result["regions"],
        annotated_image=result["annotated_image"],
        model=result["model"],
        timing=TimingInfo(
            total_ms=_ms(t0, t2),
            details={
                "upload_read_ms": _ms(t0, t1),
                "gemini_api_ms": _ms(t1, t2),
            },
        ),
    )


@router.get("/gemini-ocr/preview", response_class=HTMLResponse)
async def gemini_ocr_preview():
    """Visual testing page — upload a card image and see annotated regions + fields inline."""
    return HTMLResponse(content="""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Gemini OCR — Visual Preview</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0f1117; color: #e0e0e0; padding: 24px; }
  h1 { font-size: 18px; font-weight: 600; margin-bottom: 20px; color: #fff; }

  .controls { display: flex; gap: 12px; align-items: flex-end; margin-bottom: 20px; flex-wrap: wrap; }
  label { font-size: 12px; color: #aaa; display: block; margin-bottom: 4px; }
  input[type=file] { background: #1e2130; border: 1px solid #333; border-radius: 6px;
    padding: 8px 10px; color: #e0e0e0; font-size: 13px; cursor: pointer; }
  select { background: #1e2130; border: 1px solid #333; border-radius: 6px;
    padding: 8px 10px; color: #e0e0e0; font-size: 13px; }
  button { background: #3b6ef5; border: none; border-radius: 6px; padding: 9px 20px;
    color: #fff; font-size: 13px; font-weight: 600; cursor: pointer; }
  button:hover { background: #2c5ce8; }
  button:disabled { background: #555; cursor: not-allowed; }

  .result { display: none; gap: 24px; margin-top: 8px; }
  .result.visible { display: flex; flex-wrap: wrap; }

  .img-panel { flex: 1; min-width: 300px; }
  .img-panel img { width: 100%; border-radius: 8px; border: 1px solid #333; }

  .fields-panel { flex: 1; min-width: 300px; }
  .timing { font-size: 11px; color: #888; margin-bottom: 12px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; padding: 6px 8px; background: #1a1d2e; color: #aaa;
    font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }
  td { padding: 7px 8px; border-top: 1px solid #222; vertical-align: top; word-break: break-all; }
  td:first-child { color: #7b93d4; font-size: 12px; white-space: nowrap; width: 38%; }
  td:last-child { color: #e0e0e0; }
  .null { color: #555 !important; font-style: italic; }

  .error { color: #ff6b6b; background: #2a1a1a; border: 1px solid #5a2a2a;
    border-radius: 6px; padding: 12px 16px; margin-top: 12px; font-size: 13px; }
  .spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid #555;
    border-top-color: #fff; border-radius: 50%; animation: spin .7s linear infinite;
    vertical-align: middle; margin-right: 6px; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<h1>Gemini OCR — Visual Region Preview</h1>

<div class="controls">
  <div>
    <label>Image file</label>
    <input type="file" id="fileInput" accept="image/*">
  </div>
  <div>
    <label>Model</label>
    <select id="modelSelect">
      <option value="gemini-2.5-flash">gemini-2.5-flash</option>
      <option value="gemini-2.5-pro">gemini-2.5-pro</option>
      <option value="gemini-2.0-flash">gemini-2.0-flash</option>
    </select>
  </div>
  <button id="runBtn" onclick="run()">Run</button>
</div>

<div id="error" class="error" style="display:none"></div>
<div class="result" id="result">
  <div class="img-panel">
    <img id="annotatedImg" src="" alt="Annotated image">
  </div>
  <div class="fields-panel">
    <div class="timing" id="timingInfo"></div>
    <table>
      <thead><tr><th>Field</th><th>Value</th></tr></thead>
      <tbody id="fieldsBody"></tbody>
    </table>
  </div>
</div>

<script>
async function run() {
  const fileInput = document.getElementById('fileInput');
  const model = document.getElementById('modelSelect').value;
  const btn = document.getElementById('runBtn');
  const errDiv = document.getElementById('error');
  const resultDiv = document.getElementById('result');

  if (!fileInput.files.length) { alert('Please select an image file.'); return; }

  errDiv.style.display = 'none';
  resultDiv.classList.remove('visible');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Running…';

  const form = new FormData();
  form.append('file', fileInput.files[0]);
  form.append('model', model);

  try {
    const resp = await fetch('/gemini-ocr/upload/annotated', { method: 'POST', body: form });
    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(detail.detail || resp.statusText);
    }
    const data = await resp.json();

    document.getElementById('annotatedImg').src = 'data:image/jpeg;base64,' + data.annotated_image;

    const t = data.timing;
    document.getElementById('timingInfo').textContent =
      t ? `total ${t.total_ms} ms  |  gemini ${t.details.gemini_api_ms} ms  |  model: ${data.model}` : `model: ${data.model}`;

    const tbody = document.getElementById('fieldsBody');
    tbody.innerHTML = '';
    for (const [key, val] of Object.entries(data.fields)) {
      const isEmpty = val === null || val === undefined || (Array.isArray(val) && val.length === 0);
      const display = isEmpty ? 'null' : (Array.isArray(val) ? val.join(' / ') : String(val));
      tbody.innerHTML += `<tr>
        <td>${key}</td>
        <td class="${isEmpty ? 'null' : ''}">${display}</td>
      </tr>`;
    }

    resultDiv.classList.add('visible');
  } catch (e) {
    errDiv.textContent = 'Error: ' + e.message;
    errDiv.style.display = 'block';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run';
  }
}
</script>
</body>
</html>""")


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
