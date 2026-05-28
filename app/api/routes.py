import logging
import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from app.schemas.preprocess import (
    TimingInfo,
    QualityCheck,
    PreprocessRequest,
    PreprocessResponse,
    PreprocessOCRRequest,
    PreprocessOCRResponse,
    GeminiOCRRequest,
    GeminiOCRResponse,
    GeminiOCRAnnotatedResponse,
    GeminiOCRProcessedResponse,
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

_OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "sample_imgs" / "outputs"


def _validate_cleaned_image(cleaned_bytes: bytes, fields: dict) -> dict:
    """
    Quality gate before persisting a processed card image.

    Checks three independent signals:
      1. Aspect ratio — ID-1 cards are ~1.586:1; a bad Gemini crop produces wrong proportions.
      2. MRZ presence — MRZ spans the full bottom edge; if missing, the card wasn't fully captured.
      3. Key field count — if fewer than 3 core fields extracted, the crop region was wrong.

    Returns a dict matching the QualityCheck schema.
    """
    import io
    from PIL import Image

    img = Image.open(io.BytesIO(cleaned_bytes))
    w, h = img.size
    long_side, short_side = max(w, h), min(w, h)
    ratio = round(long_side / short_side, 3) if short_side > 0 else 0.0
    ratio_ok = 1.2 <= ratio <= 2.1  # ID-1 is 1.586; allow tolerance for slight mis-crops

    mrz_count = sum(1 for k in ("MRZ1", "MRZ2", "MRZ3") if fields.get(k))
    mrz_ok = mrz_count >= 2

    critical = ("idNumber", "lastNameKh", "firstNameKh", "dob", "lastNameEn", "firstNameEn")
    field_count = sum(1 for k in critical if fields.get(k))
    fields_ok = field_count >= 3

    valid = ratio_ok and (mrz_ok or fields_ok)

    reasons = []
    if not ratio_ok:
        reasons.append(f"aspect ratio {ratio} outside 1.2–2.1 (expected ~1.59 for ID-1)")
    if not mrz_ok:
        reasons.append(f"only {mrz_count}/3 MRZ lines detected")
    if not fields_ok:
        reasons.append(f"only {field_count}/6 key fields extracted")

    return {
        "valid": valid,
        "aspect_ratio": ratio,
        "mrz_lines_found": mrz_count,
        "key_fields_found": field_count,
        "reason": "OK" if not reasons else "; ".join(reasons),
    }


def _save_cleaned_image(image_bytes: bytes, original_filename: str) -> str:
    """
    Save cleaned image bytes to sample_imgs/outputs/.
    Returns the relative path string, e.g. 'sample_imgs/outputs/id1_20260527_143201.jpg'.
    """
    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(original_filename).stem if original_filename else "card"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{stem}_{timestamp}.jpg"
    dest = _OUTPUTS_DIR / filename
    dest.write_bytes(image_bytes)
    logger.info("saved cleaned image → %s", dest)
    return f"sample_imgs/outputs/{filename}"


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


# DISABLED — not part of the detect+extract integration. Decorator commented out so
# FastAPI does not register the route; function kept for reference / future restore.
# See propose-gemini-dev-integration-architecture.md.
# @router.post("/preprocess/upload", response_model=PreprocessResponse)
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

    def _ms(a, b):
        return round((b - a) * 1000, 2)

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


# DISABLED — not part of the detect+extract integration (OpenCV + kiri-OCR path).
# Decorator commented out so the route is not registered; function kept for reference.
# @router.post("/preprocess-ocr/upload", response_model=PreprocessOCRResponse)
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


# DISABLED — annotation/bounding-box route, not part of the detect+extract integration.
# Decorator commented out so the route is not registered; function kept for reference.
# @router.post("/gemini-ocr/upload/annotated", response_model=GeminiOCRAnnotatedResponse)
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

    def _ms(a, b):
        return round((b - a) * 1000, 2)

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


# DISABLED — cleaned-image preprocessing + quality-gate route (2 Gemini calls), not part
# of the detect+extract integration. Decorator commented out; function kept for reference.
# @router.post("/gemini-ocr/upload/processed", response_model=GeminiOCRProcessedResponse)
async def gemini_ocr_upload_processed(
    file: UploadFile = File(...),
    model: str = Form(settings.gemini_model),
) -> GeminiOCRProcessedResponse:
    """
    Full Gemini service: Stage 1 preprocess (detect card, de-skew, remove background)
    then Stage 2 annotate + extract on the cleaned image.

    Returns the cleaned image (store this — no boxes) plus the annotated preview,
    fields, regions and timing. This is two Gemini calls.
    """
    from experiments.gemini.extractor import process_card_from_bytes

    t0 = time.perf_counter()
    data = await file.read()
    t1 = time.perf_counter()

    try:
        result = await run_in_threadpool(
            process_card_from_bytes, data, file.content_type, model or None
        )
    except ImportError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("gemini full processing failed")
        raise HTTPException(status_code=502, detail=f"Gemini error: {e}")

    t2 = time.perf_counter()

    import base64
    cleaned_bytes = base64.b64decode(result["cleaned_image"])
    quality = _validate_cleaned_image(cleaned_bytes, result["fields"])
    if quality["valid"]:
        saved_as = _save_cleaned_image(cleaned_bytes, file.filename or "card.jpg")
        saved_type = "cleaned"
    else:
        logger.warning("quality gate failed (%s) — storing original image", quality["reason"])
        saved_as = _save_cleaned_image(data, file.filename or "card.jpg")
        saved_type = "original"

    def _ms(a, b):
        return round((b - a) * 1000, 2)

    return GeminiOCRProcessedResponse(
        text=result["text"],
        fields=result["fields"],
        regions=result["regions"],
        annotated_image=result["annotated_image"],
        cleaned_image=result["cleaned_image"],
        quality_check=QualityCheck(**quality),
        saved_as=saved_as,
        saved_type=saved_type,
        model=result["model"],
        timing=TimingInfo(
            total_ms=_ms(t0, t2),
            details={
                "upload_read_ms": _ms(t0, t1),
                "gemini_pipeline_ms": _ms(t1, t2),
            },
        ),
    )


# DISABLED — visual testing hub for the annotated/processed modes, not part of the
# detect+extract integration. Decorator commented out; function kept for reference.
# @router.get("/gemini-ocr/preview", response_class=HTMLResponse)
async def gemini_ocr_preview():
    """Visual testing hub — all three Gemini service modes in one page."""
    return HTMLResponse(content="""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Gemini OCR — Preview</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0f1117; color: #e0e0e0; padding: 24px; }
  h1 { font-size: 18px; font-weight: 600; color: #fff; margin-bottom: 4px; }
  .subtitle { font-size: 12px; color: #666; margin-bottom: 20px; }

  /* mode tabs */
  .tabs { display: flex; gap: 4px; margin-bottom: 20px; }
  .tab { padding: 7px 16px; border-radius: 6px; font-size: 13px; font-weight: 500;
    cursor: pointer; border: 1px solid #333; background: #1a1d2e; color: #888;
    transition: all .15s; }
  .tab:hover { color: #ccc; }
  .tab.active { background: #3b6ef5; border-color: #3b6ef5; color: #fff; }

  /* controls */
  .controls { display: flex; gap: 12px; align-items: flex-end; margin-bottom: 20px; flex-wrap: wrap; }
  label { font-size: 12px; color: #aaa; display: block; margin-bottom: 4px; }
  input[type=file] { background: #1e2130; border: 1px solid #333; border-radius: 6px;
    padding: 8px 10px; color: #e0e0e0; font-size: 13px; cursor: pointer; }
  select { background: #1e2130; border: 1px solid #333; border-radius: 6px;
    padding: 8px 10px; color: #e0e0e0; font-size: 13px; }
  button#runBtn { background: #3b6ef5; border: none; border-radius: 6px; padding: 9px 22px;
    color: #fff; font-size: 13px; font-weight: 600; cursor: pointer; }
  button#runBtn:hover { background: #2c5ce8; }
  button#runBtn:disabled { background: #555; cursor: not-allowed; }

  /* mode hint */
  .mode-hint { font-size: 12px; color: #666; margin-bottom: 20px; padding: 8px 12px;
    background: #161820; border-left: 3px solid #3b6ef5; border-radius: 0 6px 6px 0; }

  /* results */
  .result { display: none; gap: 24px; margin-top: 4px; }
  .result.visible { display: flex; flex-wrap: wrap; }

  .img-stack { display: flex; flex-direction: column; gap: 16px; flex: 1.4; min-width: 300px; }
  .img-single { flex: 1.4; min-width: 300px; }
  .img-block img { width: 100%; border-radius: 8px; border: 1px solid #2a2d3e; }
  .img-label { font-size: 11px; font-weight: 600; margin-bottom: 6px; display: flex;
    align-items: center; gap: 8px; color: #ccc; }
  .badge { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 4px;
    text-transform: uppercase; letter-spacing: .04em; }
  .badge-db   { background: #14532d; color: #86efac; }
  .badge-preview { background: #2a2d3e; color: #94a3b8; }

  /* fields table */
  .fields-panel { flex: 1; min-width: 280px; }
  .timing { font-size: 11px; color: #666; margin-bottom: 10px; line-height: 1.6; }
  .timing span { color: #94a3b8; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; padding: 6px 8px; background: #161820; color: #555;
    font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; }
  td { padding: 7px 8px; border-top: 1px solid #1a1d2e; vertical-align: top; word-break: break-all; }
  td:first-child { color: #7b93d4; font-size: 12px; white-space: nowrap; width: 40%; }
  td:last-child { color: #e0e0e0; }
  .null-val { color: #383d50 !important; font-style: italic; }

  .error { color: #f87171; background: #1c1010; border: 1px solid #3f1414;
    border-radius: 6px; padding: 12px 16px; margin-top: 12px; font-size: 13px; }
  .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid #444;
    border-top-color: #fff; border-radius: 50%; animation: spin .7s linear infinite;
    vertical-align: middle; margin-right: 6px; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<h1>Gemini OCR</h1>
<p class="subtitle">Visual testing hub — switch modes to compare pipeline stages</p>

<div class="tabs">
  <div class="tab active" onclick="setMode('full')">Full pipeline</div>
  <div class="tab" onclick="setMode('annotated')">Annotate + extract</div>
  <div class="tab" onclick="setMode('extract')">Extract only</div>
</div>

<div class="mode-hint" id="modeHint"></div>

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
  <!-- img area: swapped by JS depending on mode -->
  <div id="imgArea"></div>
  <div class="fields-panel">
    <div class="timing" id="timingInfo"></div>
    <table>
      <thead><tr><th>Field</th><th>Value</th></tr></thead>
      <tbody id="fieldsBody"></tbody>
    </table>
  </div>
</div>

<script>
const MODES = {
  full: {
    endpoint: '/gemini-ocr/upload/processed',
    hint: 'Stage 1: Gemini detects card corners → PIL de-skews + crops background. Stage 2: annotate + extract on cleaned image. 2 Gemini calls.',
    timingKey: 'gemini_pipeline_ms',
  },
  annotated: {
    endpoint: '/gemini-ocr/upload/annotated',
    hint: 'Gemini extracts fields and returns bounding box coordinates. Colored boxes are drawn locally with PIL. 1 Gemini call.',
    timingKey: 'gemini_api_ms',
  },
  extract: {
    endpoint: '/gemini-ocr/upload',
    hint: 'Gemini extracts structured fields only. No bounding boxes, no preprocessing. Fastest — 1 Gemini call.',
    timingKey: 'gemini_api_ms',
  },
};

let currentMode = 'full';

function setMode(mode) {
  currentMode = mode;
  document.querySelectorAll('.tab').forEach((t, i) => {
    t.classList.toggle('active', ['full','annotated','extract'][i] === mode);
  });
  document.getElementById('modeHint').textContent = MODES[mode].hint;
  document.getElementById('result').classList.remove('visible');
  document.getElementById('error').style.display = 'none';
}

function renderImages(data) {
  const area = document.getElementById('imgArea');
  if (currentMode === 'full') {
    area.className = 'img-stack';
    area.innerHTML = `
      <div class="img-block">
        <div class="img-label">Cleaned image <span class="badge badge-db">store in DB</span></div>
        <img src="data:image/jpeg;base64,${data.cleaned_image}" alt="Cleaned">
      </div>
      <div class="img-block">
        <div class="img-label">Annotated <span class="badge badge-preview">preview only</span></div>
        <img src="data:image/jpeg;base64,${data.annotated_image}" alt="Annotated">
      </div>`;
  } else if (currentMode === 'annotated') {
    area.className = 'img-single';
    area.innerHTML = `
      <div class="img-block">
        <div class="img-label">Annotated <span class="badge badge-preview">preview only</span></div>
        <img src="data:image/jpeg;base64,${data.annotated_image}" alt="Annotated">
      </div>`;
  } else {
    area.className = '';
    area.innerHTML = '';
  }
}

function renderTiming(data) {
  const t = data.timing;
  const key = MODES[currentMode].timingKey;
  let html = '';
  if (t) {
    html += `total <span>${t.total_ms} ms</span> &nbsp;|&nbsp; ` +
            `gemini <span>${t.details[key] ?? '—'} ms</span> &nbsp;|&nbsp; ` +
            `model <span>${data.model}</span>`;
  } else {
    html += `model <span>${data.model}</span>`;
  }
  if (data.quality_check) {
    const qc = data.quality_check;
    if (qc.valid) {
      html += `<br>quality <span style="color:#86efac">✓ passed</span>` +
              ` &nbsp;·&nbsp; saved cleaned → <span style="color:#86efac">${data.saved_as}</span>`;
    } else {
      html += `<br>quality <span style="color:#f87171">✗ failed — ${qc.reason}</span>` +
              ` &nbsp;·&nbsp; saved original → <span style="color:#fbbf24">${data.saved_as}</span>`;
    }
  }
  document.getElementById('timingInfo').innerHTML = html;
}

function renderFields(fields) {
  const tbody = document.getElementById('fieldsBody');
  tbody.innerHTML = '';
  for (const [key, val] of Object.entries(fields)) {
    const empty = val === null || val === undefined || (Array.isArray(val) && !val.length);
    const display = empty ? 'null' : (Array.isArray(val) ? val.join(' · ') : String(val));
    tbody.innerHTML += `<tr>
      <td>${key}</td>
      <td class="${empty ? 'null-val' : ''}">${display}</td>
    </tr>`;
  }
}

async function run() {
  const fileInput = document.getElementById('fileInput');
  if (!fileInput.files.length) { alert('Please select an image file.'); return; }

  const btn = document.getElementById('runBtn');
  const errDiv = document.getElementById('error');
  const resultDiv = document.getElementById('result');

  errDiv.style.display = 'none';
  resultDiv.classList.remove('visible');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Running…';

  const form = new FormData();
  form.append('file', fileInput.files[0]);
  form.append('model', document.getElementById('modelSelect').value);

  try {
    const resp = await fetch(MODES[currentMode].endpoint, { method: 'POST', body: form });
    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(detail.detail || resp.statusText);
    }
    const data = await resp.json();
    renderImages(data);
    renderTiming(data);
    renderFields(data.fields || {});
    resultDiv.classList.add('visible');
  } catch (e) {
    errDiv.textContent = 'Error: ' + e.message;
    errDiv.style.display = 'block';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run';
  }
}

// init hint on load
setMode('full');
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
