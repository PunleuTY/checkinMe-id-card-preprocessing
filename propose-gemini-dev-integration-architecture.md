# Gemini ID Card Processing — Dev Integration Architecture

## Overview

The goal of this integration is to **replace the development CamDX OCR service with a
Gemini-driven service** for reading Cambodian National ID (NID) cards. The Gemini
service is a **drop-in replacement**: it accepts a raw ID card image and returns the
**same field structure** the CamDX service returned, so nothing downstream in the
Laravel backend has to change.

The service is intentionally simple and **stateless**. It runs the card through two
logical stages:

1. **Stage 1 — Text Detection & Recognition.** Gemini reads every character on the
   raw card image (Khmer + Latin), producing a full text transcription.
2. **Stage 2 — Field Extraction.** The needed NID fields are pulled out of what was
   read, and the MRZ lines are parsed deterministically to confirm/correct the
   key fields.

**Storage policy (this phase):** the service does **not** produce or store a
cleaned/de-skewed image or an annotated/boxed image. The only artifact persisted is
the **original uploaded image** — image clean-up and bounding-box preview are out of
scope for now and can be layered back in later.

---

## End-to-End Flow

```text
Client (CheckinMe app / Swagger UI / API consumer)
│
│  uploads raw ID card image
▼
┌─────────────────────────────────────────────────────────┐
│  Laravel backend                                        │
│  GeminiService::scanNid($idImage)                       │
│    • stores the ORIGINAL image                          │
│    • opens a GeminiResult audit row (request_data)      │
│    • forwards the image to the Gemini OCR service       │
└────────────────────────┬────────────────────────────────┘
                         │  POST /gemini-ocr/upload
                         │  multipart: file=<image>, model=<gemini-model>
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Gemini OCR service  ·  FastAPI                         │
│  main.py + app/api/routes.py                            │
│  gemini_ocr_upload()  →  run_in_threadpool()            │
│                         │                               │
│                         ▼                               │
│  experiments/gemini/extractor.py                        │
│  extract_from_bytes()                                   │
│                                                         │
│  ┌───────────────────────────────────────────────────┐ │
│  │ STAGE 1 — Text Detection & Recognition            │ │
│  │   • Single Gemini call on the RAW image           │ │
│  │   • Prompt: PROMPT                                │ │
│  │   • Reads all Khmer + Latin text → "text"         │ │
│  └───────────────────────────────────────────────────┘ │
│                         │                               │
│                         ▼                               │
│  ┌───────────────────────────────────────────────────┐ │
│  │ STAGE 2 — Field Extraction                        │ │
│  │   • Same Gemini call returns structured "fields"  │ │
│  │   • _parse_mrz() parses MRZ2/MRZ3 programmatically│ │
│  │     and backfills/corrects: dob, gender,          │ │
│  │     expiredDate, lastNameEn, firstNameEn          │ │
│  └───────────────────────────────────────────────────┘ │
└────────────────────────┬────────────────────────────────┘
                         │ { text, fields, model, timing }
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Laravel backend                                        │
│  GeminiService::scanNid()  (continued)                  │
│    • writes status_code + response_data to GeminiResult │
│      (NO base64 image blobs are stored)                 │
│    • returns $json['fields']  ← CamDX-compatible shape  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Consumer controller                                    │
│  GeminiPersonalInfoController::readIdCard()             │
│    • maps fields → personal-info response               │
│      (name, name_en, gender, dob, id_card_number, …)    │
└─────────────────────────────────────────────────────────┘
```

---

## Output Contract (identical to CamDX)

Stage 2 returns the same `fields` structure the CamDX service produced, so the
Laravel mappers (`GeminiPersonalInfoController`) work unchanged:

```jsonc
{
  "idNumber":    "101325482",
  "lastNameKh":  "ជន",
  "firstNameKh": "ពេងហុង",
  "dob":         "17/01/2001",      // DD/MM/YYYY
  "gender":      "M",               // "M" | "F"
  "lastNameEn":  "CHORN",           // UPPERCASE
  "firstNameEn": "PENGHONG",        // UPPERCASE
  "expiredDate": "18/01/2030",      // DD/MM/YYYY
  "issuedDate":  "18/01/2020",      // DD/MM/YYYY
  "address":     "ភូមិកាច់ត្រក ឃុំលាយបូរ ស្រុកត្រាំកក់ តាកែវ",
  "pob":         "ឃុំលាយបូរ ស្រុកត្រាំកក់ តាកែវ",
  "MRZ1":        "IDKHM1013254824<<<<<<<<<<<<<<<",
  "MRZ2":        "0101170M2610139KHM<<<<<<<<<<<8",
  "MRZ3":        "CHORN<<PENGHONG<<<<<<<<<<<<<<<"
}
```

The full HTTP response from the service is:

```jsonc
{
  "text":   "<full raw transcription — Stage 1 output>",
  "fields": { /* the structure above — Stage 2 output */ },
  "model":  "gemini-2.5-flash",
  "timing": { "total_ms": 0, "details": { "upload_read_ms": 0, "gemini_api_ms": 0 } }
}
```

Rules enforced by the prompt + MRZ parser:
- `null` for any field that cannot be read with confidence — never guessed.
- Dates are `DD/MM/YYYY` exactly as printed.
- `gender` is exactly `"M"` or `"F"`.
- English names are UPPERCASE; Khmer text is preserved verbatim.
- MRZ-derived values (dob, gender, expiredDate, English names) are deterministic
  and override the vision values when they disagree.

---

## File Map

| File | Responsibility |
|---|---|
| `main.py` | FastAPI app entry point, mounts router |
| `app/core/config.py` | Settings: `gemini_model`, `gemini_api_key` |
| `app/api/routes.py` | HTTP endpoints, timing |
| `app/schemas/preprocess.py` | Pydantic request/response models |
| `experiments/gemini/extractor.py` | Gemini logic: prompt, call, MRZ parse |
| `experiments/gemini/pipeline.py` | CLI entrypoint for running extraction from terminal |

### Key symbols inside `experiments/gemini/extractor.py`

| Symbol | What it does |
|---|---|
| `PROMPT` | Detection + recognition + field-extraction prompt |
| `_sniff_mime()` | Detect image MIME type from magic bytes |
| `_build_client()` | Build `google.genai.Client` from API key / env |
| `_parse_json()` | Tolerant JSON parse, strips markdown fences |
| `_yymmdd_to_ddmmyyyy()` | Convert MRZ date format → DD/MM/YYYY |
| `_parse_mrz()` | Programmatic MRZ2/MRZ3 parse → field corrections (Stage 2) |
| `extract_from_bytes()` | The service entrypoint: Stage 1 + Stage 2 in one call |

> **Deprecated for this phase** (kept in the codebase but not on the production path):
> card-corner detection + de-skew (`_detect_card_corners`,
> `preprocess_card_from_bytes`), bounding-box annotation (`annotate_image`,
> `extract_with_regions_from_bytes`, `_FIELD_COLORS`), the full two-Gemini-call
> orchestrator (`process_card_from_bytes`), and the quality gate / image-save logic
> (`_validate_cleaned_image`, `_save_cleaned_image`). These power the richer
> `/upload/annotated` and `/upload/processed` endpoints and can be promoted back
> later if cleaned/annotated images are needed.

---

## API Endpoints

| Method | Path | Mode | Gemini calls | Role |
|---|---|---|---|---|
| `POST` | `/gemini-ocr/upload` | Detect + recognize + extract | 1 | **Production endpoint — CamDX replacement** |
| `POST` | `/gemini-ocr/upload/annotated` | + bounding boxes | 1 | Debug / preview only |
| `POST` | `/gemini-ocr/upload/processed` | + de-skew + quality gate | 2 | Debug / preview only |
| `GET`  | `/gemini-ocr/preview` | Visual testing hub (HTML) | — | Debug / preview only |
| `GET`  | `/health` | Health check | — | — |

The Laravel `config/gemini.php` `scan_nid` endpoint should point at
**`/gemini-ocr/upload`** (the stateless detect+extract endpoint), not
`/gemini-ocr/upload/processed`.

---

## Laravel Integration

`App\Services\GeminiService::scanNid($idImage)`:

1. Stores the **original** image (no cleaned/annotated variants).
2. Creates a `GeminiResult` audit row with `request_data`.
3. Attaches the image and `POST`s it to `{base_url}/gemini-ocr/upload`.
4. Records `status_code` + `response_data` on the `GeminiResult` row.
   Because the service no longer returns `cleaned_image` / `annotated_image`,
   `response_data` stays small — just `text`, `fields`, `model`, `timing`.
5. Maps HTTP status to a success/error envelope and returns `$json['fields']`.

`App\Http\Controllers\Employee\GeminiPersonalInfoController::readIdCard()` then maps
the CamDX-compatible `fields` into the personal-info response (`name`, `name_en`,
`gender`, `nationality`, `address`, `dob`, `id_card_number`, `id_card_expire_date`, …)
exactly as it did against CamDX.

---

## Changes from the Previous Dev Integration

| Previous (`/upload/processed`) | This integration (`/upload`) |
|---|---|
| Stage 1 = card detection + de-skew + crop (cleaned image) | Stage 1 = text detection & recognition |
| Stage 2 = annotate + extract on cleaned image | Stage 2 = field extraction + MRZ parse |
| 2 Gemini calls | 1 Gemini call |
| Quality gate decides cleaned-vs-original | No quality gate |
| Stores cleaned **or** original image | Stores **original only** |
| Returns `cleaned_image` + `annotated_image` base64 | Returns no image blobs |
| `response_data` bloated with base64 | `response_data` small (text + fields) |

Output `fields` structure is **unchanged** in both — that is the contract that keeps
this a true CamDX replacement.

---

## Configuration

Gemini OCR service (`.env`):

```text
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash     # or gemini-2.5-pro
```

Loaded via `app/core/config.py` using `pydantic-settings`.

Laravel backend (`config/gemini.php`):

```text
GEMINI_SERVICE_BASE_URL=http://localhost:8000
GEMINI_SERVICE_TIMEOUT=30
GEMINI_SERVICE_MODEL=gemini-2.5-flash
GEMINI_SERVICE_ENDPOINTS_SCAN_NID=/gemini-ocr/upload
```
