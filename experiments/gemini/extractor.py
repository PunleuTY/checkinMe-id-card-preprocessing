"""
Gemini-based OCR + structured extraction for NID cards.

Sends the raw ID card image straight to Gemini — no local preprocessing — and
asks it to read the text and pull structured fields. Accepts a file path or raw
image bytes.

Requires GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment / .env.
"""

import json
import logging
import re

from app.core.config import settings

logger = logging.getLogger(__name__)

# PROMPT VERSION 1

PROMPT = """You are an OCR system specialized in Cambodian National ID cards (NID).
  The card contains text in both Khmer script and Latin/English characters.

  The image may be a phone photo or scan that is rotated, skewed, low-resolution,
  or affected by glare, shadows, or background clutter. Read the card regardless.

  Cambodian NID cards contain:
  - A 9-digit ID number (top area of the card)
  - Last name and first name in Khmer script (ឈ្មោះជាអក្សរខ្មែរ)
  - Last name and first name in English/Latin (UPPERCASE)
  - Date of birth in DD/MM/YYYY format (កន្លែងកំណើត)
  - Gender: M for male, F for female
  - Place of birth in Khmer (ទីកន្លែងកំណើត / POB)
  - Address in Khmer (អាស័យដ្ឋានបច្ចុប្បន្ន )
  - Issue date in DD/MM/YYYY (កាលបរិច្ឆេទចេញ)
  - Expiry date in DD/MM/YYYY (កាលបរិច្ឆេទផុតកំណត់)
  - Distinguishing physical features (ភិនភាគចំណាំពិសេស): a list of short Khmer phrases
    describing unique physical marks on the owner's face or body that appear in
    the section just above the MRZ, e.g. "ប្រជ្រុយនៅក្រោមច្រមុះ ០,៦ ស.ម" (mole under the nose).
    Preserve the exact Khmer text; return as an array of strings.
  - Three MRZ lines at the bottom (machine-readable zone):
      MRZ1: starts with IDKHM followed by the 9-digit ID number and < padding
      MRZ2: 6-digit DOB + check digit + sex (M/F) + 6-digit expiry + check + KHM + padding + composite check
      MRZ3: SURNAME<<GIVENNAME<<< padding (all uppercase, spaces replaced by <)

  Return ONLY valid JSON with exactly this shape (no markdown fences, no extra keys):

  {
    "text": "<full raw transcription of every visible character on the card>",
    "fields": {
      "idNumber": null,
      "lastNameKh": null,
      "firstNameKh": null,
      "dob": null,
      "gender": null,
      "lastNameEn": null,
      "firstNameEn": null,
      "expiredDate": null,
      "issuedDate": null,
      "address": null,
      "pob": null,
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
  - distinguishingFeatures must be an array of strings (empty array [] if none found).
    Each element is one physical mark phrase exactly as printed in Khmer.
  - For MRZ lines: copy every character exactly including all < characters, digits, and letters.
    Valid MRZ characters are only: A-Z, 0-9, and <
"""

PROMPT_WITH_REGIONS = """You are an OCR system specialized in Cambodian National ID cards (NID).
  The card contains text in both Khmer script and Latin/English characters.

  The image may be a phone photo or scan that is rotated, skewed, low-resolution,
  or affected by glare, shadows, or background clutter. Read the card regardless.

  Cambodian NID cards contain:
  - A 9-digit ID number (top area of the card)
  - Last name and first name in Khmer script (ឈ្មោះជាអក្សរខ្មែរ)
  - Last name and first name in English/Latin (UPPERCASE)
  - Date of birth in DD/MM/YYYY format (កន្លែងកំណើត)
  - Gender: M for male, F for female
  - Place of birth in Khmer (ទីកន្លែងកំណើត / POB)
  - Address in Khmer (អាស័យដ្ឋានបច្ចុប្បន្ន)
  - Issue date in DD/MM/YYYY (កាលបរិច្ឆេទចេញ)
  - Expiry date in DD/MM/YYYY (កាលបរិច្ឆេទផុតកំណត់)
  - Distinguishing physical features (ភិនភាគចំណាំពិសេស): short Khmer phrases for
    unique physical marks just above the MRZ.
  - Three MRZ lines at the bottom (machine-readable zone)

  Return ONLY valid JSON with exactly this shape (no markdown fences, no extra keys):

  {
    "text": "<full raw transcription of every visible character on the card>",
    "fields": {
      "idNumber": null,
      "lastNameKh": null,
      "firstNameKh": null,
      "dob": null,
      "gender": null,
      "lastNameEn": null,
      "firstNameEn": null,
      "expiredDate": null,
      "issuedDate": null,
      "address": null,
      "pob": null,
      "distinguishingFeatures": [],
      "MRZ1": null,
      "MRZ2": null,
      "MRZ3": null
    },
    "regions": {
      "idNumber": null,
      "lastNameKh": null,
      "firstNameKh": null,
      "dob": null,
      "gender": null,
      "lastNameEn": null,
      "firstNameEn": null,
      "expiredDate": null,
      "issuedDate": null,
      "address": null,
      "pob": null,
      "distinguishingFeatures": null,
      "MRZ1": null,
      "MRZ2": null,
      "MRZ3": null
    }
  }

  Rules:
  - Use null for any field or region you cannot read or locate with confidence.
  - Dates must be DD/MM/YYYY exactly as printed on the card.
  - gender must be exactly "M" or "F", nothing else.
  - English name fields (lastNameEn, firstNameEn) must be UPPERCASE.
  - Preserve all Khmer Unicode characters verbatim — do not transliterate.
  - distinguishingFeatures in fields must be an array of strings (empty [] if none found).
  - For MRZ lines: copy every character exactly including all < characters.
    Valid MRZ characters are only: A-Z, 0-9, and <
  - Each region value is [y_min, x_min, y_max, x_max] with coordinates in 0–1000
    (0,0 = top-left corner of the image, 1000,1000 = bottom-right corner).
    Use null if the field region cannot be located on the image.
"""

# Color per field for bounding box annotation (RGB hex)
_FIELD_COLORS: dict[str, str] = {
    "idNumber":               "#FF4444",
    "lastNameKh":             "#4488FF",
    "firstNameKh":            "#4488FF",
    "lastNameEn":             "#1155CC",
    "firstNameEn":            "#1155CC",
    "dob":                    "#22AA44",
    "expiredDate":            "#22AA44",
    "issuedDate":             "#22AA44",
    "gender":                 "#AA44CC",
    "address":                "#FF8800",
    "pob":                    "#FF8800",
    "distinguishingFeatures": "#00AAAA",
    "MRZ1":                   "#CCAA00",
    "MRZ2":                   "#CCAA00",
    "MRZ3":                   "#CCAA00",
}


def annotate_image(image_bytes: bytes, regions: dict) -> bytes:
    """
    Draw Gemini-detected bounding boxes onto the image.

    Each region value must be [y_min, x_min, y_max, x_max] in 0–1000 scale.
    Returns JPEG bytes of the annotated image.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise ImportError("Pillow is not installed. Run: pip install Pillow")

    import io

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_w, img_h = img.size
    draw = ImageDraw.Draw(img)

    # Try system font, fall back to PIL default
    font = None
    for font_path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            font = ImageFont.truetype(font_path, size=13)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()

    for field, bbox in regions.items():
        if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue

        color = _FIELD_COLORS.get(field, "#888888")
        y_min, x_min, y_max, x_max = bbox
        px0 = int(x_min / 1000 * img_w)
        py0 = int(y_min / 1000 * img_h)
        px1 = int(x_max / 1000 * img_w)
        py1 = int(y_max / 1000 * img_h)

        # Draw 3-px thick rectangle outline
        for d in range(3):
            draw.rectangle([px0 - d, py0 - d, px1 + d, py1 + d], outline=color)

        # Label: filled background + white text above the box
        label_bbox = draw.textbbox((0, 0), field, font=font)
        lbl_w = label_bbox[2] - label_bbox[0] + 8
        lbl_h = label_bbox[3] - label_bbox[1] + 4
        lbl_x = px0
        lbl_y = max(0, py0 - lbl_h)
        draw.rectangle([lbl_x, lbl_y, lbl_x + lbl_w, lbl_y + lbl_h], fill=color)
        draw.text((lbl_x + 4, lbl_y + 2), field, fill="#FFFFFF", font=font)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def extract_with_regions_from_bytes(
    image_bytes: bytes,
    mime_type: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> dict:
    """
    Run Gemini extraction with bounding box detection, then annotate the image.

    Returns the normal extraction result plus:
      "regions": dict of field → [y_min, x_min, y_max, x_max] (0–1000 scale)
      "annotated_image": base64 JPEG of the image with colored boxes drawn on it
    """
    import base64
    from google.genai import types

    client = _build_client(api_key)
    model_name = model if (model and model != "string") else settings.gemini_model
    mime = (
        mime_type
        if (mime_type and mime_type.startswith("image/"))
        else _sniff_mime(image_bytes)
    )

    logger.info(
        "gemini: extracting+regions with model=%s mime=%s (%d bytes)",
        model_name,
        mime,
        len(image_bytes),
    )

    resp = client.models.generate_content(
        model=model_name,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
            PROMPT_WITH_REGIONS,
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
        regions = data.get("regions", {})
    except (json.JSONDecodeError, AttributeError):
        logger.warning("gemini: response was not valid JSON, returning raw text")
        text, fields, regions = raw, {}, {}

    # MRZ-derived corrections applied same as normal extraction
    mrz_corrections = _parse_mrz(fields)
    for key, val in mrz_corrections.items():
        existing = fields.get(key)
        if not existing:
            fields[key] = val
        elif existing != val:
            logger.info("mrz correction: %s %r → %r", key, existing, val)
            fields[key] = val

    # Draw Gemini's detected regions onto the original image
    try:
        annotated_bytes = annotate_image(image_bytes, regions)
        annotated_b64 = base64.b64encode(annotated_bytes).decode("utf-8")
    except Exception:
        logger.warning("annotate_image failed, returning original image")
        annotated_b64 = base64.b64encode(image_bytes).decode("utf-8")

    logger.info(
        "gemini: extracted %d fields, %d regions annotated",
        len(fields),
        sum(1 for v in regions.values() if v),
    )
    return {
        "text": text,
        "fields": fields,
        "regions": regions,
        "annotated_image": annotated_b64,
        "model": model_name,
        "raw": raw,
    }


# ---------------------------------------------------------------------------
# Stage 1 — Gemini-driven preprocessing (detect card, de-skew, drop background)
# ---------------------------------------------------------------------------

PROMPT_DETECT_CARD = """You are a document-localization system.

The image is a phone photo or scan of a single Cambodian National ID card,
possibly rotated, skewed, or surrounded by background clutter (table, hand, etc.).

Locate the four physical corners of the ID card itself (the rectangular card edge,
NOT the text inside it).

Return ONLY valid JSON, no markdown fences, with exactly this shape:

{
  "corners": [[y, x], [y, x], [y, x], [y, x]]
}

Rules:
- Provide exactly four [y, x] points, ordered: top-left, top-right, bottom-right, bottom-left
  (relative to the card's own orientation as it appears in the image).
- Each coordinate is on a 0–1000 scale, where [0,0] is the top-left of the IMAGE
  and [1000,1000] is the bottom-right of the IMAGE.
- Track the card's true corners even if it is rotated or tilted.
- If no ID card is visible, return {"corners": null}.
"""


def _find_perspective_coeffs(dst_pts: list, src_pts: list) -> list:
    """
    Solve the 8 perspective coefficients that PIL's Image.transform(PERSPECTIVE)
    needs to map each OUTPUT pixel (dst_pts) back to the INPUT image (src_pts).
    """
    import numpy as np

    matrix = []
    for (dx, dy), (sx, sy) in zip(dst_pts, src_pts):
        matrix.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy])
        matrix.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy])
    A = np.array(matrix, dtype=float)
    B = np.array(src_pts, dtype=float).reshape(8)
    res = np.linalg.solve(A, B)
    return res.tolist()


def _detect_card_corners(
    image_bytes: bytes,
    mime_type: str | None,
    model: str | None,
    api_key: str | None,
) -> list | None:
    """
    Ask Gemini for the card's four corners, returned as pixel (x, y) points
    ordered TL, TR, BR, BL. Returns None if no card is detected.
    """
    from google.genai import types

    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is not installed. Run: pip install Pillow")

    import io

    client = _build_client(api_key)
    model_name = model if (model and model != "string") else settings.gemini_model
    mime = (
        mime_type
        if (mime_type and mime_type.startswith("image/"))
        else _sniff_mime(image_bytes)
    )

    resp = client.models.generate_content(
        model=model_name,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
            PROMPT_DETECT_CARD,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
        ),
    )

    try:
        data = _parse_json(resp.text or "")
        corners = data.get("corners")
    except (json.JSONDecodeError, AttributeError):
        logger.warning("detect_card: response was not valid JSON")
        return None

    if not corners or not isinstance(corners, list) or len(corners) != 4:
        logger.warning("detect_card: no usable corners returned")
        return None

    img_w, img_h = Image.open(io.BytesIO(image_bytes)).size
    pts = []
    for point in corners:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            logger.warning("detect_card: malformed corner %r", point)
            return None
        y, x = point  # Gemini returns [y, x] on a 0–1000 scale
        pts.append((x / 1000 * img_w, y / 1000 * img_h))
    return pts


def preprocess_card_from_bytes(
    image_bytes: bytes,
    mime_type: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> bytes:
    """
    Stage 1: detect the ID card via Gemini, then de-skew + crop it out of the
    background with a perspective warp. Returns cleaned JPEG bytes.

    Falls back to the original image bytes if the card cannot be located or the
    warp fails — never raises on a detection miss.
    """
    import io
    import math

    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is not installed. Run: pip install Pillow")

    corners = _detect_card_corners(image_bytes, mime_type, model, api_key)
    if corners is None:
        logger.warning("preprocess: card not detected — returning original image")
        return image_bytes

    tl, tr, br, bl = corners

    def _dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    out_w = int(round((_dist(tr, tl) + _dist(br, bl)) / 2))
    out_h = int(round((_dist(bl, tl) + _dist(br, tr)) / 2))
    if out_w < 10 or out_h < 10:
        logger.warning("preprocess: degenerate card quad — returning original image")
        return image_bytes

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Output rectangle corners (TL, TR, BR, BL) map back to detected src corners.
        dst_pts = [(0, 0), (out_w, 0), (out_w, out_h), (0, out_h)]
        src_pts = [tl, tr, br, bl]
        coeffs = _find_perspective_coeffs(dst_pts, src_pts)
        warped = img.transform((out_w, out_h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

        # Keep landscape orientation (cards are wider than tall).
        if warped.height > warped.width:
            warped = warped.rotate(-90, expand=True)

        buf = io.BytesIO()
        warped.save(buf, format="JPEG", quality=92)
        logger.info("preprocess: cleaned card → %dx%d", warped.width, warped.height)
        return buf.getvalue()
    except Exception:
        logger.warning("preprocess: warp failed — returning original image")
        return image_bytes


def process_card_from_bytes(
    image_bytes: bytes,
    mime_type: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> dict:
    """
    Full Gemini service: preprocess (Stage 1) then annotate + extract (Stage 2).

    Returns:
      "cleaned_image"   — base64 JPEG, de-skewed + background-removed, NO boxes
                          (this is the image to persist in production)
      "annotated_image" — base64 JPEG of the cleaned image WITH detected boxes
                          (preview/debug only — do not store)
      plus "text", "fields", "regions", "model".
    """
    import base64

    cleaned_bytes = preprocess_card_from_bytes(image_bytes, mime_type, model, api_key)

    # Stage 2 runs entirely on the cleaned image, so regions/boxes align with it.
    result = extract_with_regions_from_bytes(
        cleaned_bytes, "image/jpeg", model=model, api_key=api_key
    )
    result["cleaned_image"] = base64.b64encode(cleaned_bytes).decode("utf-8")
    return result


"""PROMPT_V1
You are an OCR and information-extraction system for national ID cards
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

    Returns {"text": str, "fields": dict, "model": str, "raw": str}.
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

    logger.info("gemini: extracted %d chars, %d fields", len(text), len(fields))
    return {"text": text, "fields": fields, "model": model_name, "raw": raw}


def extract_from_file(
    image_path: str, model: str | None = None, api_key: str | None = None
) -> dict:
    """Run Gemini extraction directly on an image file path."""
    with open(image_path, "rb") as f:
        data = f.read()
    return extract_from_bytes(data, None, model=model, api_key=api_key)
