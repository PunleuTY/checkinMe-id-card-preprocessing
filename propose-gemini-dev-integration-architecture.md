# Gemini ID Card Processing — End-to-End Architecture

## Overview

The service exposes a FastAPI application that accepts raw ID card images and runs
them through a fully Gemini-driven pipeline: card detection → de-skew → field
extraction → region annotation. A quality gate decides whether to persist the
cleaned or original image.

---

## End-to-End Flow

```
Client (Swagger UI / preview page / API consumer)
│
│  POST /gemini-ocr/upload/processed
│  multipart: file=<image>, model=<gemini-model>
│
▼
┌─────────────────────────────────────────────────────────┐
│  FastAPI  ·  main.py  +  app/api/routes.py              │
│  gemini_ocr_upload_processed()                          │
└────────────────────────┬────────────────────────────────┘
                         │ run_in_threadpool()
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 1 — Card Detection & Preprocessing               │
│  experiments/gemini/extractor.py                        │
│                                                         │
│  process_card_from_bytes()                              │
│    │                                                    │
│    ├─► preprocess_card_from_bytes()                     │
│    │     │                                              │
│    │     ├─► _detect_card_corners()                     │
│    │     │     • Gemini call #1                         │
│    │     │     • Prompt: PROMPT_DETECT_CARD             │
│    │     │     • Returns: [[y,x] × 4] corners (0–1000) │
│    │     │                                              │
│    │     └─► _find_perspective_coeffs() + PIL warp      │
│    │           • Converts 0–1000 coords → pixel coords  │
│    │           • PIL PERSPECTIVE transform (BICUBIC)    │
│    │           • Auto-rotates to landscape              │
│    │           • Output: cleaned JPEG bytes             │
│    │           • Fallback: original bytes if no card    │
│    │                                                    │
└────┼────────────────────────────────────────────────────┘
     │ cleaned_bytes
     ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 2 — Annotation & Extraction (on cleaned image)   │
│  experiments/gemini/extractor.py                        │
│                                                         │
│  extract_with_regions_from_bytes()                      │
│    │                                                    │
│    ├─► Gemini call #2                                   │
│    │     • Prompt: PROMPT_WITH_REGIONS                  │
│    │     • Input: cleaned image                         │
│    │     • Output: { text, fields, regions }            │
│    │       - fields: idNumber, lastNameKh, firstNameKh, │
│    │                 lastNameEn, firstNameEn, dob,      │
│    │                 gender, address, pob, issuedDate,  │
│    │                 expiredDate, distinguishing-        │
│    │                 Features[], MRZ1, MRZ2, MRZ3       │
│    │       - regions: field → [y,x,y,x] bounding boxes │
│    │                                                    │
│    ├─► _parse_mrz()                                     │
│    │     • Programmatic MRZ2/MRZ3 parse                 │
│    │     • Backfills/corrects: dob, gender,             │
│    │       expiredDate, lastNameEn, firstNameEn         │
│    │                                                    │
│    └─► annotate_image()                                 │
│          • PIL draws colored boxes on cleaned image     │
│          • Color map: _FIELD_COLORS (per field type)    │
│          • Output: annotated JPEG (base64)              │
│                                                         │
└────────────────────────┬────────────────────────────────┘
                         │ { text, fields, regions,
                         │   annotated_image, cleaned_image }
                         ▼
┌─────────────────────────────────────────────────────────┐
│  QUALITY GATE                                           │
│  app/api/routes.py · _validate_cleaned_image()          │
│                                                         │
│  Signal 1 — Aspect ratio                               │
│    cleaned image long/short ratio must be 1.2 – 2.1    │
│    (ID-1 card standard: 85.6 × 54.0 mm = 1.586)        │
│                                                         │
│  Signal 2 — MRZ presence                               │
│    at least 2 of 3 MRZ lines must be non-null          │
│                                                         │
│  Signal 3 — Key field count                             │
│    at least 3 of 6 core fields must be non-null        │
│    (idNumber, lastNameKh, firstNameKh,                  │
│     lastNameEn, firstNameEn, dob)                       │
│                                                         │
│  valid = ratio_ok AND (mrz_ok OR fields_ok)             │
│                                                         │
└────────────────────────┬────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
           valid                not valid
              │                     │
              ▼                     ▼
      save cleaned image    save original image
      saved_type="cleaned"  saved_type="original"
              │                     │
              └──────────┬──────────┘
                         │ sample_imgs/outputs/
                         │ {stem}_{YYYYMMDD_HHMMSS}.jpg
                         │ app/api/routes.py
                         │ _save_cleaned_image()
                         ▼
┌─────────────────────────────────────────────────────────┐
│  HTTP Response  ·  app/schemas/preprocess.py            │
│  GeminiOCRProcessedResponse                             │
│                                                         │
│  text            — full card transcription              │
│  fields          — structured field dict                │
│  regions         — bounding boxes per field             │
│  cleaned_image   — base64 JPEG, no boxes  ← store this  │
│  annotated_image — base64 JPEG, with boxes ← preview   │
│  quality_check   — { valid, aspect_ratio,               │
│                      mrz_lines_found,                   │
│                      key_fields_found, reason }         │
│  saved_as        — relative path on disk                │
│  saved_type      — "cleaned" | "original"               │
│  model           — Gemini model ID used                 │
│  timing          — { total_ms, details{} }              │
└─────────────────────────────────────────────────────────┘
```

---

## File Map

| File | Responsibility |
|---|---|
| `main.py` | FastAPI app entry point, mounts router |
| `app/core/config.py` | Settings: `gemini_model`, `gemini_api_key`, output dimensions |
| `app/api/routes.py` | All HTTP endpoints, timing, quality gate, save logic |
| `app/schemas/preprocess.py` | Pydantic request/response models incl. `QualityCheck` |
| `app/utils/image.py` | Image encode/decode helpers (base64 ↔ ndarray ↔ bytes) |
| `experiments/gemini/extractor.py` | All Gemini logic: prompts, calls, MRZ parse, annotate, orchestration |
| `experiments/gemini/pipeline.py` | CLI entrypoint for running extraction directly from terminal |

### Key symbols inside `experiments/gemini/extractor.py`

| Symbol | What it does |
|---|---|
| `PROMPT` | Basic extraction prompt (fields only) |
| `PROMPT_WITH_REGIONS` | Extraction + bounding box coordinates |
| `PROMPT_DETECT_CARD` | Card corner detection for Stage 1 preprocessing |
| `_FIELD_COLORS` | Color map for annotation boxes |
| `_sniff_mime()` | Detect image MIME type from magic bytes |
| `_build_client()` | Build `google.genai.Client` from API key / env |
| `_parse_json()` | Tolerant JSON parse, strips markdown fences |
| `_yymmdd_to_ddmmyyyy()` | Convert MRZ date format → DD/MM/YYYY |
| `_parse_mrz()` | Programmatic MRZ2/MRZ3 parse → field corrections |
| `_detect_card_corners()` | Gemini call #1 — returns 4 pixel corner points |
| `_find_perspective_coeffs()` | Solve PIL perspective transform coefficients |
| `preprocess_card_from_bytes()` | Stage 1: detect corners → PIL warp → cleaned JPEG |
| `annotate_image()` | Draw colored bounding boxes + labels onto image |
| `extract_from_bytes()` | Basic extraction (no regions, no annotation) |
| `extract_with_regions_from_bytes()` | Extraction + regions + annotation |
| `process_card_from_bytes()` | Full orchestrator: Stage 1 + Stage 2 |

---

## API Endpoints

| Method | Path | Mode | Gemini calls |
|---|---|---|---|
| `POST` | `/gemini-ocr/upload` | Extract only | 1 |
| `POST` | `/gemini-ocr/upload/annotated` | Annotate + extract | 1 |
| `POST` | `/gemini-ocr/upload/processed` | Full pipeline | 2 |
| `POST` | `/preprocess/upload` | OpenCV pipeline only | 0 |
| `POST` | `/preprocess-ocr/upload` | OpenCV + CamDX OCR | 0 |
| `GET`  | `/gemini-ocr/preview` | Visual testing hub (HTML) | — |
| `GET`  | `/health` | Health check | — |

---

## Visual Testing Hub

`GET /gemini-ocr/preview` serves a self-contained HTML page with three mode tabs:

- **Full pipeline** — calls `/upload/processed`, shows cleaned image (DB badge) + annotated image (preview badge) + fields table + quality result + saved path
- **Annotate + extract** — calls `/upload/annotated`, shows annotated image + fields
- **Extract only** — calls `/upload`, shows fields only

---

## Configuration (`.env`)

```
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash     # or gemini-2.5-pro
OUTPUT_WIDTH=1024
OUTPUT_HEIGHT=640
WEBP_QUALITY=85
```

Loaded via `app/core/config.py` using `pydantic-settings`.
