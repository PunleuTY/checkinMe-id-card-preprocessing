"""
Gemini-based OCR + structured extraction for NID cards.

Sends the raw ID card image straight to Gemini — no local preprocessing — and
asks it to read the card and pull the structured fields the application needs.
Accepts a file path or raw image bytes.

Requires GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment / .env.
"""

import json
import logging
import re

from app.core.config import settings

logger = logging.getLogger(__name__)


PROMPT = """You are an OCR system specialized in Cambodian National ID cards (NID).
  The card contains text in both Khmer script and Latin/English characters.

  The image may be a phone photo or scan that is rotated, skewed, low-resolution,
  or affected by glare, shadows, or background clutter. Read the card regardless.

  Extract only the following fields:
  - idNumber: 9-digit ID number (top area of the card), digits only
  - lastNameKh / firstNameKh: last and first name in Khmer script
  - lastNameEn / firstNameEn: last and first name in English/Latin (UPPERCASE)
  - dob: date of birth in DD/MM/YYYY
  - gender: "M" for male, "F" for female
  - expiredDate: expiry date in DD/MM/YYYY (កាលបរិច្ឆេទផុតកំណត់)
  - address: current address in Khmer (អាស័យដ្ឋានបច្ចុប្បន្ន)
  - Three MRZ lines at the bottom (machine-readable zone):
      MRZ1: starts with IDKHM followed by the 9-digit ID number and < padding
      MRZ2: 6-digit DOB + check digit + sex (M/F) + 6-digit expiry + check + KHM + padding + composite check
      MRZ3: SURNAME<<GIVENNAME<<< padding (all uppercase, spaces replaced by <)

  Return ONLY valid JSON with exactly this shape (no markdown fences, no extra keys,
  and do NOT include a full text transcription):

  {
    "fields": {
      "idNumber": null,
      "lastNameKh": null,
      "firstNameKh": null,
      "lastNameEn": null,
      "firstNameEn": null,
      "dob": null,
      "gender": null,
      "expiredDate": null,
      "address": null,
      "MRZ1": null,
      "MRZ2": null,
      "MRZ3": null
    }
  }

  Rules:
  - Use null for any field you cannot read with confidence. Never guess or invent values.
  - Dates must be DD/MM/YYYY exactly as printed on the card.
  - gender must be exactly "M" or "F", nothing else.
  - English name fields (lastNameEn, firstNameEn) must be UPPERCASE.
  - Preserve all Khmer Unicode characters verbatim — do not transliterate.
  - For MRZ lines: copy every character exactly including all < characters, digits, and letters.
    Valid MRZ characters are only: A-Z, 0-9, and <
"""


def _sniff_mime(data: bytes) -> str:
    if data[:4] == b"\x89PNG":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _build_client(api_key: str | None):
    try:
        from google import genai
    except ImportError:
        raise ImportError(
            "google-genai is not installed. "
            "Run: pip install google-genai  "
            "(or pip install -r experiments/requirements.txt)"
        )

    key = api_key or settings.gemini_api_key or None
    # When key is None, the SDK falls back to GEMINI_API_KEY / GOOGLE_API_KEY env vars.
    return genai.Client(api_key=key) if key else genai.Client()


def _parse_json(raw: str) -> dict:
    """Tolerant JSON parse — strips ``` fences if the model added them."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    return json.loads(cleaned)


def _yymmdd_to_ddmmyyyy(yymmdd: str) -> str | None:
    """Convert MRZ date YYMMDD → DD/MM/YYYY, century inferred by threshold."""
    if not re.fullmatch(r"\d{6}", yymmdd):
        return None
    yy = int(yymmdd[:2])
    mm = yymmdd[2:4]
    dd = yymmdd[4:6]
    year = 2000 + yy if yy <= 30 else 1900 + yy
    return f"{dd}/{mm}/{year}"


def _parse_mrz(fields: dict) -> dict:
    """
    Parse MRZ2 and MRZ3 programmatically to produce reliable field values.

    MRZ is machine-readable and has a fixed format, so these values are more
    trustworthy than vision extraction for dates, gender, and English names.
    Returns a dict of corrections/backfills to apply on top of vision results.
    """
    corrections = {}

    mrz2 = (fields.get("MRZ2") or "").replace(" ", "")
    mrz3 = (fields.get("MRZ3") or "").replace(" ", "")

    # MRZ2: YYMMDD(6) + check(1) + sex(1) + expiry_YYMMDD(6) + check(1) + KHM(3) + ...
    if len(mrz2) >= 15:
        dob_raw = mrz2[0:6]
        gender_raw = mrz2[7] if len(mrz2) > 7 else None
        expiry_raw = mrz2[8:14]

        dob = _yymmdd_to_ddmmyyyy(dob_raw)
        if dob:
            corrections["dob"] = dob

        if gender_raw in ("M", "F"):
            corrections["gender"] = gender_raw

        expiry = _yymmdd_to_ddmmyyyy(expiry_raw)
        if expiry:
            corrections["expiredDate"] = expiry

    # MRZ3: LASTNAME<<FIRSTNAME<<<<<<<<<<<<<<
    if mrz3:
        parts = mrz3.split("<<", 1)
        last = parts[0].replace("<", " ").strip()
        first = parts[1].replace("<", " ").strip() if len(parts) > 1 else None
        if last:
            corrections["lastNameEn"] = last
        if first:
            corrections["firstNameEn"] = first

    return corrections


def extract_from_bytes(
    image_bytes: bytes,
    mime_type: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> dict:
    """
    Run Gemini extraction on raw image bytes.

    Returns {"fields": dict, "model": str, "raw": str}.
    MRZ lines are parsed programmatically to backfill/correct vision results.
    """
    from google.genai import types

    client = _build_client(api_key)
    model_name = model if (model and model != "string") else settings.gemini_model
    mime = (
        mime_type
        if (mime_type and mime_type.startswith("image/"))
        else _sniff_mime(image_bytes)
    )

    logger.info(
        "gemini: extracting with model=%s mime=%s (%d bytes)",
        model_name,
        mime,
        len(image_bytes),
    )

    # Disable "thinking" for flash models (deterministic OCR doesn't need it and it
    # dominates latency/token cost). Pro models reject thinking_budget=0, so guard it.
    config_kwargs = dict(response_mime_type="application/json", temperature=0)
    if "flash" in model_name:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

    resp = client.models.generate_content(
        model=model_name,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
            PROMPT,
        ],
        config=types.GenerateContentConfig(**config_kwargs),
    )

    raw = resp.text or ""
    try:
        data = _parse_json(raw)
        fields = data.get("fields", {})
    except (json.JSONDecodeError, AttributeError):
        logger.warning("gemini: response was not valid JSON, returning empty fields")
        fields = {}

    # MRZ-derived values are deterministic — use them to backfill/correct vision fields
    mrz_corrections = _parse_mrz(fields)
    for key, val in mrz_corrections.items():
        existing = fields.get(key)
        if not existing:
            fields[key] = val
            logger.debug("mrz backfill: %s = %r", key, val)
        elif existing != val:
            logger.info("mrz correction: %s %r → %r", key, existing, val)
            fields[key] = val

    logger.info("gemini: extracted %d fields", len(fields))
    return {"fields": fields, "model": model_name, "raw": raw}


def extract_from_file(
    image_path: str, model: str | None = None, api_key: str | None = None
) -> dict:
    """Run Gemini extraction directly on an image file path."""
    with open(image_path, "rb") as f:
        data = f.read()
    return extract_from_bytes(data, None, model=model, api_key=api_key)
