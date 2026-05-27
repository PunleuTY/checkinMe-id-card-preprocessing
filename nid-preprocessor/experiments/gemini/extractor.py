"""
Gemini-based OCR + structured extraction for NID cards.

Sends the raw ID card image straight to Gemini — no local preprocessing — and
asks it to read the text and pull structured fields. Accepts a file path or raw
image bytes.

Requires GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment / .env.
"""

import json
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

PROMPT = """You are an OCR and information-extraction system for national ID cards
(Cambodian NID — text is in Khmer and Latin/English).

The image is a phone photo or scan and may be rotated, skewed, low-resolution, or
have glare, shadows, noise, or background clutter. Read the card regardless and
extract every readable piece of text.

Return ONLY valid JSON, no markdown fences, with exactly this shape:

{
  "text": "<full transcription of all visible text, keep Khmer and Latin as-is>",
  "fields": {
    "id_number": null,
    "name_khmer": null,
    "name_latin": null,
    "date_of_birth": null,
    "sex": null,
    "height": null,
    "place_of_birth": null,
    "address": null,
    "issue_date": null,
    "expiry_date": null
  }
}

Rules:
- Use null for any field you cannot read with confidence. Do NOT guess or invent values.
- Keep dates exactly as printed on the card.
- Preserve Khmer Unicode characters verbatim.
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


def extract_from_bytes(
    image_bytes: bytes,
    mime_type: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> dict:
    """
    Run Gemini extraction on raw image bytes.

    Returns {"text": str, "fields": dict, "model": str, "raw": str}.
    """
    from google.genai import types

    client = _build_client(api_key)
    # Swagger pre-fills optional string fields with the literal "string"; treat as unset.
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

    resp = client.models.generate_content(
        model=model_name,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
            PROMPT,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
        ),
    )

    raw = resp.text or ""
    try:
        data = _parse_json(raw)
        text = data.get("text", "")
        fields = data.get("fields", {})
    except (json.JSONDecodeError, AttributeError):
        logger.warning("gemini: response was not valid JSON, returning raw text")
        text, fields = raw, {}

    logger.info("gemini: extracted %d chars, %d fields", len(text), len(fields))

    return {"text": text, "fields": fields, "model": model_name, "raw": raw}


def extract_from_file(
    image_path: str, model: str | None = None, api_key: str | None = None
) -> dict:
    """Run Gemini extraction directly on an image file path."""
    with open(image_path, "rb") as f:
        data = f.read()
    return extract_from_bytes(data, None, model=model, api_key=api_key)
