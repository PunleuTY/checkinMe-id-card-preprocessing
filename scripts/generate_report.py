"""
Generate the Gemini OCR Service report.

Produces a professional + technical report on the Gemini service used for Khmer
NID card text detection & recognition, following the same content structure as
`gemini-ocr-service-report.md`.

Two modes:

  1. Static (default) — writes the full report with the model-comparison table
     left as fill-in placeholders. Works offline, no API key needed.

  2. Benchmark (--benchmark) — runs the production extractor (`extract_from_bytes`)
     over the images in `sample_imgs/` with BOTH gemini-2.5-flash and gemini-2.5-pro,
     measures latency / JSON-validity / field-fill-rate, and injects the real
     numbers into the §6 comparison table. Requires GEMINI_API_KEY.

Output:
  - Markdown (always).
  - HTML (--html) — a styled, shareable document, if the `markdown` package is
    installed (pip install markdown).

Usage:
  python -m scripts.generate_report
  python -m scripts.generate_report --benchmark
  python -m scripts.generate_report --benchmark --html -o reports/
  python -m scripts.generate_report --benchmark --models gemini-2.5-flash gemini-2.5-pro
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "sample_imgs"

# The fields the production prompt is expected to return (used for fill-rate).
EXPECTED_FIELDS = [
    "idNumber", "lastNameKh", "firstNameKh", "dob", "gender",
    "lastNameEn", "firstNameEn", "expiredDate", "issuedDate",
    "address", "pob", "MRZ1", "MRZ2", "MRZ3",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def _list_sample_images(limit: int | None) -> list[Path]:
    imgs = sorted(
        p for p in SAMPLE_DIR.glob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    return imgs[:limit] if limit else imgs


def benchmark_model(model: str, images: list[Path]) -> dict:
    """
    Run the production extractor over every image with one model.

    Returns measured aggregates: avg/median latency (ms), JSON-valid count,
    and average field-fill rate. Accuracy is NOT measured here — it needs a
    human-verified ground truth, so those rows stay as fill-in placeholders.
    """
    from gemini.extractor import extract_from_bytes

    latencies: list[float] = []
    json_ok = 0
    fill_rates: list[float] = []
    errors = 0

    for img in images:
        data = img.read_bytes()
        t0 = time.perf_counter()
        try:
            result = extract_from_bytes(data, model=model)
        except Exception as e:  # noqa: BLE001 — report, don't crash the whole run
            errors += 1
            print(f"  ! {model} {img.name}: {e}", file=sys.stderr)
            continue
        latencies.append((time.perf_counter() - t0) * 1000)

        fields = result.get("fields") or {}
        if fields:  # extract_from_bytes returns {} only on a JSON parse fallback
            json_ok += 1
            filled = sum(1 for k in EXPECTED_FIELDS if fields.get(k))
            fill_rates.append(filled / len(EXPECTED_FIELDS))

        print(f"  · {model} {img.name}: {latencies[-1]:.0f} ms" if latencies else "")

    n = len(images)
    return {
        "model": model,
        "n": n,
        "errors": errors,
        "avg_ms": round(statistics.mean(latencies)) if latencies else None,
        "median_ms": round(statistics.median(latencies)) if latencies else None,
        "json_ok": json_ok,
        "fill_rate": round(statistics.mean(fill_rates) * 100, 1) if fill_rates else None,
    }


def _fmt_ms(v) -> str:
    return f"{v} ms" if v is not None else "_____ ms"


def _fmt_pct(v) -> str:
    return f"{v} %" if v is not None else "_____ %"


def comparison_table(results: dict | None) -> str:
    """Render the §6.3 measured-results table — live numbers or placeholders."""
    flash = (results or {}).get("gemini-2.5-flash", {})
    pro = (results or {}).get("gemini-2.5-pro", {})

    n = flash.get("n") or pro.get("n") or "N"

    def cell(d: dict, key: str, fmt) -> str:
        return fmt(d.get(key)) if d else fmt(None)

    if results:
        note = (
            f"> Measured on {n} sample card(s) from `sample_imgs/` on "
            f"{datetime.now():%Y-%m-%d %H:%M}. Latency is wall-clock around the "
            f"`extract_from_bytes` call. Accuracy rows require a human-verified "
            f"ground truth and are left blank for manual scoring."
        )
        avg = f"| Avg. latency per card | {cell(flash,'avg_ms',_fmt_ms)} | {cell(pro,'avg_ms',_fmt_ms)} |"
        med = f"| Median latency per card | {cell(flash,'median_ms',_fmt_ms)} | {cell(pro,'median_ms',_fmt_ms)} |"
        jok = (
            f"| JSON well-formed (no parse fallback) "
            f"| {flash.get('json_ok','_')} / {n} | {pro.get('json_ok','_')} / {n} |"
        )
        fill = f"| Avg. field-fill rate | {cell(flash,'fill_rate',_fmt_pct)} | {cell(pro,'fill_rate',_fmt_pct)} |"
    else:
        note = (
            "> Fill these in with the numbers you recorded. Use the same set of "
            "sample cards for both models (`sample_imgs/`) so the comparison is fair. "
            "Run this script with `--benchmark` to auto-populate the measurable rows."
        )
        avg = "| Avg. latency per card (`gemini_api_ms`) | _____ ms | _____ ms |"
        med = "| Median latency per card | _____ ms | _____ ms |"
        jok = "| JSON well-formed (no parse fallback) | _____ / N | _____ / N |"
        fill = "| Avg. field-fill rate | _____ % | _____ % |"

    return f"""{note}

| Metric | Gemini 2.5 Flash | Gemini 2.5 Pro |
|---|---|---|
{avg}
{med}
| Field accuracy — clear cards | _____ % | _____ % |
| Field accuracy — difficult cards (glare/skew/low-res) | _____ % | _____ % |
| Khmer name/address correctness | _____ | _____ |
| ID number / dates / gender correctness¹ | _____ | _____ |
{jok}
{fill}
| Relative cost per call | 1× (baseline) | ~ ___× |

¹ Note: `dob`, `gender`, `expiredDate`, and English names are **MRZ-verified**, so
both models should score near-identically on these whenever the MRZ is readable — the
real differentiator is the Khmer fields and overall reading on hard images."""


# ---------------------------------------------------------------------------
# Report body
# ---------------------------------------------------------------------------

def build_report(results: dict | None) -> str:
    generated = datetime.now().strftime("%Y-%m-%d")
    return f"""# Gemini OCR Service — Khmer NID Card Text Detection & Recognition

*Generated: {generated}*

**Report scope:** the Gemini service only — the component that reads a Cambodian
National ID (NID) card image and returns its data as structured fields, ready to be
hosted and integrated with the CheckinMe AI server. It also compares the two Gemini
models that were tested for this job: **Gemini 2.5 Flash** and **Gemini 2.5 Pro**.

---

## 1. Executive summary

The CheckinMe app lets a user photograph their ID card. We need to turn that photo
into clean, structured data — name, ID number, date of birth, gender, address, and
so on — automatically.

Previously this was done with a separate OCR service (CamDX). This service **replaces
that** with Google's **Gemini** AI vision model. A user uploads a raw card photo; the
service sends it to Gemini, which both **reads** the text on the card (Khmer + English)
and **organises** it into the exact fields the CheckinMe backend already expects.

Two key points worth keeping in mind:

- **It is a drop-in replacement.** The data shape it returns is identical to the old
  service, so the rest of the CheckinMe system does not need to change.
- **It is "stateless."** It does not keep a database or remember anything between
  requests. One photo in, one set of fields out.

We tested two versions of the AI model — **Flash** (faster, cheaper) and **Pro**
(slower, more accurate) — to decide which to run in production. The comparison and
recommendation are in [Section 6](#6-model-comparison-gemini-25-flash-vs-25-pro).

---

## 2. What the service does

```text
  User photo of an ID card
            │
            ▼
   ┌─────────────────────────────┐
   │  Gemini OCR service         │
   │  (FastAPI, one HTTP call)   │
   │                             │
   │  Stage 1: read all text     │  ← Gemini "sees" the card
   │  Stage 2: extract fields    │  ← organise into named fields
   │  Stage 2b: MRZ double-check │  ← verify key fields mathematically
   └─────────────────────────────┘
            │
            ▼
   Structured JSON fields  →  CheckinMe backend
```

The whole thing is **one call to Gemini** plus a small amount of local logic to
verify the result. There is no local image cleanup on the production path — Gemini
itself handles rotated, skewed, glare-affected, or low-resolution photos.

---

## 3. How it works

### Stage 1 — Text detection & recognition

The raw image bytes are sent straight to Gemini together with a specialised prompt
(`PROMPT` in `gemini/extractor.py`). The prompt tells Gemini:

- This is a Cambodian NID card with **both Khmer script and Latin/English** text.
- The photo may be rotated, skewed, low-res, or have glare/shadows — read it anyway.
- Exactly which fields the card contains, with the Khmer field labels for each
  (e.g. `កាលបរិច្ឆេទចេញ` = issue date), and how the three **MRZ** lines (the
  machine-readable zone at the bottom) are structured.

Gemini returns a `text` value: a full raw transcription of every visible character.

### Stage 2 — Field extraction

The **same** Gemini call also returns a structured `fields` object. The prompt forces
a strict contract:

- Return **only valid JSON**, no markdown, exactly the agreed keys.
- Use `null` for anything that can't be read with confidence — **never guess**.
- Dates as `DD/MM/YYYY`, gender exactly `M`/`F`, English names UPPERCASE, Khmer
  preserved verbatim (no transliteration).
- `temperature=0` and `response_mime_type="application/json"` are set on the API call
  so output is deterministic and machine-parseable.

### Stage 2b — MRZ verification (the reliability trick)

The MRZ lines at the bottom of the card follow a fixed international format, so they
can be parsed **mathematically** rather than trusting the AI's reading. `_parse_mrz()`
decodes MRZ2 and MRZ3 to recover:

- `dob`, `gender`, `expiredDate` (from MRZ2)
- `lastNameEn`, `firstNameEn` (from MRZ3)

These deterministic values **backfill or override** the vision-read fields when they
disagree. This is what makes the key identity fields trustworthy even if the visual
OCR slips.

### The output contract (identical to CamDX)

```jsonc
{{
  "text":   "<full raw transcription>",
  "fields": {{
    "idNumber":    "101325482",
    "lastNameKh":  "ជន",
    "firstNameKh": "ពេងហុង",
    "dob":         "17/01/2001",
    "gender":      "M",
    "lastNameEn":  "CHORN",
    "firstNameEn": "PENGHONG",
    "expiredDate": "18/01/2030",
    "issuedDate":  "18/01/2020",
    "address":     "ភូមិកាច់ត្រក ឃុំលាយបូរ ស្រុកត្រាំកក់ តាកែវ",
    "pob":         "ឃុំលាយបូរ ស្រុកត្រាំកក់ តាកែវ",
    "MRZ1":        "IDKHM1013254824<<<<<<<<<<<<<<<",
    "MRZ2":        "0101170M2610139KHM<<<<<<<<<<<8",
    "MRZ3":        "CHORN<<PENGHONG<<<<<<<<<<<<<<<"
  }},
  "model":  "gemini-2.5-flash",
  "timing": {{ "total_ms": 0, "details": {{ "upload_read_ms": 0, "gemini_api_ms": 0 }} }}
}}
```

### The prompt — how Gemini is instructed

The service has no machine-learning model of its own to train. Its behaviour is
controlled almost entirely by the **prompt** — the written set of instructions sent to
Gemini alongside the image. The prompt *is* the configuration: it defines the task,
the output shape, and the rules. The production path uses a single prompt, `PROMPT`,
in `gemini/extractor.py`.

The prompt is structured in four parts:

1. **Role & context.** It tells Gemini it is "an OCR system specialised in Cambodian
   National ID cards," that the card carries **both Khmer script and Latin/English**,
   and that the photo may be rotated, skewed, low-resolution, or affected by glare,
   shadows, or background clutter — *"Read the card regardless."* This primes the model
   to handle messy real-world photos instead of expecting a clean scan.

2. **A description of the card's layout.** It enumerates every field the card contains
   and pairs each with its Khmer label so the model knows where to look — e.g. the
   9-digit ID number at the top, name in Khmer (`ឈ្មោះជាអក្សរខ្មែរ`), date of birth,
   gender, place of birth (`ទីកន្លែងកំណើត`), address (`អាស័យដ្ឋានបច្ចុប្បន្ន`), issue
   date (`កាលបរិច្ឆេទចេញ`), expiry date (`កាលបរិច្ឆេទផុតកំណត់`), and the three **MRZ**
   lines — including the exact internal format of each MRZ line.

3. **The exact output schema.** It shows Gemini the precise JSON skeleton to return —
   a `text` field plus a `fields` object with exactly the agreed keys (`idNumber`,
   `lastNameKh`, `firstNameKh`, `dob`, `gender`, `lastNameEn`, `firstNameEn`,
   `expiredDate`, `issuedDate`, `address`, `pob`, `MRZ1`, `MRZ2`, `MRZ3`) — and *no
   extra keys, no markdown fences*. This is what keeps the output a true drop-in
   replacement for CamDX.

4. **Strict rules.** The closing rules remove ambiguity and prevent the model from
   inventing data:
   - Use `null` for anything that can't be read with confidence — **never guess**.
   - Dates must be `DD/MM/YYYY` exactly as printed.
   - `gender` must be exactly `"M"` or `"F"`.
   - English name fields must be UPPERCASE; Khmer must be preserved verbatim — **no
     transliteration**.
   - For MRZ lines, copy every character exactly, including all `<` padding; valid
     MRZ characters are only `A–Z`, `0–9`, and `<`.

The prompt works together with two API settings that make the output reliable:
`temperature=0` (deterministic — the same card gives the same answer) and
`response_mime_type="application/json"` (the model returns parseable JSON, not prose).

> **Other prompts in the file, not on the production path:** `PROMPT_WITH_REGIONS`
> (same task plus bounding-box coordinates for each field), `PROMPT_DETECT_CARD` (asks
> only for the card's four corners, used by the de-skew stage), and `PROMPT_V1` (an
> earlier draft, kept for reference). These belong to the disabled image-cleanup /
> annotation modes and can be promoted back if those features are revived.

### Key code map

| Symbol (`gemini/extractor.py`) | Responsibility |
|---|---|
| `PROMPT` | Detection + recognition + field-extraction instructions |
| `extract_from_bytes()` | **Production entrypoint** — Stage 1 + Stage 2 in one Gemini call |
| `extract_from_file()` | Same, from a file path (CLI/testing) |
| `_parse_mrz()` | Deterministic MRZ2/MRZ3 parse → field corrections |
| `_yymmdd_to_ddmmyyyy()` | MRZ date `YYMMDD` → `DD/MM/YYYY` |
| `_parse_json()` | Tolerant JSON parse (strips stray markdown fences) |
| `_sniff_mime()` | Detect JPEG/PNG/WebP from magic bytes |
| `_build_client()` | Build `google.genai.Client` from API key / env |

> **Present but not on the production path** (kept for future use): card-corner
> detection + de-skew (`preprocess_card_from_bytes`), bounding-box annotation
> (`annotate_image`, `extract_with_regions_from_bytes`), the 2-call orchestrator
> (`process_card_from_bytes`), and the quality gate / image-save logic. Their HTTP
> routes are disabled. They can be promoted back if cleaned/annotated images are
> needed later.

---

## 4. The production endpoint

| Method | Path | Gemini calls | Role |
|---|---|---|---|
| `POST` | `/gemini-ocr/upload` | 1 | **Production — CamDX replacement** |
| `GET`  | `/health` | — | Liveness check |

`POST /gemini-ocr/upload` takes a `multipart/form-data` upload:
`file=<image>`, optional `model=<gemini-model>`. The Gemini call runs in a thread pool
(`run_in_threadpool`) so the async server stays responsive while waiting on the API.
Errors map cleanly to HTTP: `500` if the SDK is missing, `502` on a Gemini failure.

The disabled routes (`/upload/annotated`, `/upload/processed`, `/preview`) belong to
the richer image-cleanup mode and are deliberately not registered for this phase.

---

## 5. Hosting & integration

### Hosting (Google Cloud Run)

- Containerised via `Dockerfile` (Python 3.12-slim) and served by **gunicorn +
  uvicorn worker**: `--workers 1 --threads 8 --timeout 300`. One worker with threads
  is the right shape because the work is **I/O-bound** (waiting on the Gemini API),
  not CPU-bound.
- `--timeout 300` is deliberately larger than the Laravel caller's timeout so a slow
  **Pro** call doesn't get killed mid-request.
- `deploy/deploy.sh` deploys to Cloud Run from source. Defaults: region
  `asia-southeast1` (Singapore, closest to Cambodia), `1 CPU / 1Gi`, concurrency 8,
  scale-to-zero (`min-instances 0`, pay per request), max 5 instances.
- The **Gemini API key lives in Secret Manager**, never baked into the image; the
  model is passed as an env var (`GEMINI_MODEL`).

### Integration with the CheckinMe AI server

The Laravel backend's `GeminiService::scanNid()`:

1. Stores the **original** uploaded image (no cleaned/annotated variants this phase).
2. Opens a `GeminiResult` audit row, `POST`s the image to `/gemini-ocr/upload`.
3. Records `status_code` + `response_data` (small — just `text`, `fields`, `model`,
   `timing`; no base64 image blobs).
4. Returns `$json['fields']`, which `GeminiPersonalInfoController::readIdCard()` maps
   to the personal-info response exactly as it did for CamDX.

Laravel config (`config/gemini.php` / `.env`):

```text
GEMINI_SERVICE_BASE_URL=<cloud-run-url>
GEMINI_SERVICE_TIMEOUT=60
GEMINI_SERVICE_MODEL=gemini-2.5-flash
GEMINI_SERVICE_ENDPOINTS_SCAN_NID=/gemini-ocr/upload
```

---

## 6. Model comparison: Gemini 2.5 Flash vs 2.5 Pro

Both models run through the **exact same** code and prompt; only the model name
changes (`GEMINI_MODEL`, or the `model` form field per request). So this is a clean
apples-to-apples comparison of the model itself.

### 6.1 What each model is for

| | **Gemini 2.5 Flash** | **Gemini 2.5 Pro** |
|---|---|---|
| Designed for | Speed & cost at scale | Maximum accuracy / hard reasoning |
| Latency | Lower (faster response) | Higher (noticeably slower) |
| Cost per call | Cheaper | More expensive |
| Best on | Clear, well-lit cards | Difficult cards: glare, skew, low-res, messy Khmer |

### 6.2 Evidence already baked into the codebase

- **The deployment defaults to Flash.** `.env.example`, `deploy.sh`
  (`GEMINI_MODEL=gemini-2.5-flash`), and the architecture doc all default to Flash —
  i.e. Flash was chosen as the day-to-day production model.
- **The timeouts were tuned around Pro being slow.** Both the Docker `--timeout 300`
  and `deploy.sh` comments explicitly call out that the long timeout exists so that
  "slow gemini-2.5-pro calls" survive. This is direct evidence that **Pro is
  materially slower** in practice.
- **`config.py` currently sets `gemini_model = "gemini-2.5-pro"` as the in-code
  default**, while the deploy/env files set Flash. ⚠️ *This is an inconsistency worth
  resolving* — pick one default so local runs and deployed runs behave the same. (See
  [Section 7](#7-findings--recommendations).)

### 6.3 Measured results from your test

{comparison_table(results)}

### 6.4 How to read the comparison

- **Flash** is the workhorse: fast, cheap, and good enough on the great majority of
  real submissions. For a user-facing flow where people wait for the result, the
  lower latency matters a lot.
- **Pro** earns its keep on the hard cases — poor lighting, heavy skew, faint or
  unusual Khmer text — where it tends to read more correctly, at the cost of speed
  and money.
- Because the **most sensitive fields are MRZ-verified in code**, the accuracy gap
  between the two narrows on exactly the fields that matter most for identity. That
  strengthens the case for Flash as the default.

### 6.5 Recommendation

- **Default to Gemini 2.5 Flash in production** (matches the existing deploy config):
  best balance of speed, cost, and accuracy for the typical card.
- **Keep Pro available as a fallback / escalation.** The `model` form field already
  lets the caller pick per request — so a card that fails the quality checks on Flash
  could be retried on Pro.
- **Resolve the default-model inconsistency** between `config.py` (Pro) and the
  deploy files (Flash) so behaviour is predictable everywhere.

---

## 7. Findings & recommendations

1. **Align the default model.** `app/core/config.py` defaults to `gemini-2.5-pro`,
   but `.env.example` and `deploy.sh` use `gemini-2.5-flash`. Make them consistent
   (recommend Flash) to avoid surprises between local and deployed runs.
2. **Record the measured numbers** from your Flash-vs-Pro test into the table in
   §6.3 so the recommendation is backed by data, not just defaults.
3. **Consider an automatic Flash→Pro escalation** for low-confidence results, using
   the existing per-request `model` override.
4. **Security:** the deploy currently allows unauthenticated access
   (`ALLOW_UNAUTH=true`) for dev simplicity. Before production, lock the Cloud Run
   service down (IAM / signed requests) since it processes personal ID data.

---

## 8. Glossary

- **OCR** — Optical Character Recognition: turning text in an image into machine text.
- **NID** — National ID card.
- **MRZ** — Machine-Readable Zone: the `<<<`-padded lines at the bottom of the card,
  designed to be read by machines and follow a fixed, verifiable format.
- **Stateless** — the service keeps no memory or database; each request is independent.
- **Latency** — how long the service takes to respond.
- **Cloud Run** — Google's service for running containers that scale automatically
  (including down to zero when idle, so you only pay per request).
"""


# ---------------------------------------------------------------------------
# HTML rendering (optional, professional output)
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gemini OCR Service — Report</title>
<style>
  body {{ font-family: -apple-system, system-ui, "Segoe UI", sans-serif;
    max-width: 860px; margin: 40px auto; padding: 0 24px; line-height: 1.65;
    color: #1a1a1a; }}
  h1 {{ font-size: 28px; border-bottom: 3px solid #3b6ef5; padding-bottom: 10px; }}
  h2 {{ font-size: 22px; margin-top: 40px; border-bottom: 1px solid #e2e2e2;
    padding-bottom: 6px; }}
  h3 {{ font-size: 17px; margin-top: 28px; color: #2c3e50; }}
  code {{ background: #f4f4f6; padding: 1px 5px; border-radius: 4px;
    font-size: 0.9em; }}
  pre {{ background: #1e2130; color: #e0e0e0; padding: 16px; border-radius: 8px;
    overflow-x: auto; font-size: 13px; line-height: 1.45; }}
  pre code {{ background: none; padding: 0; color: inherit; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 14px; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left;
    vertical-align: top; }}
  th {{ background: #f4f6fb; }}
  blockquote {{ border-left: 4px solid #3b6ef5; margin: 16px 0; padding: 8px 16px;
    background: #f6f8fd; color: #444; }}
  hr {{ border: none; border-top: 1px solid #e2e2e2; margin: 32px 0; }}
  em {{ color: #555; }}
</style>
</head>
<body>
{body}
</body>
</html>"""


def render_html(md_text: str) -> str | None:
    """Convert the markdown report to a styled HTML doc. Needs the `markdown` pkg."""
    try:
        import markdown
    except ImportError:
        return None
    body = markdown.markdown(
        md_text, extensions=["tables", "fenced_code", "toc"]
    )
    return _HTML_TEMPLATE.format(body=body)


# ---------------------------------------------------------------------------
# PDF rendering (pandoc -> HTML -> Chrome headless; embeds Khmer text)
# ---------------------------------------------------------------------------

_STYLE_HEADER = Path(__file__).resolve().parent / "report_style.html"

_CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome",
    "chromium",
    "chromium-browser",
]


def _find_chrome() -> str | None:
    import shutil
    for c in _CHROME_CANDIDATES:
        if Path(c).exists() or shutil.which(c):
            return c
    return None


def render_pdf(md_path: Path, pdf_path: Path) -> bool:
    """
    Convert the markdown report to PDF: pandoc builds styled HTML (with a
    Khmer-capable font stack from report_style.html), then headless Chrome
    prints it to PDF so Khmer Unicode renders as real, selectable text.

    Returns True on success, False (with a stderr hint) if a tool is missing.
    """
    import shutil
    import subprocess
    import tempfile

    if not shutil.which("pandoc"):
        print("PDF skipped — pandoc not found (brew install pandoc).", file=sys.stderr)
        return False
    chrome = _find_chrome()
    if not chrome:
        print("PDF skipped — no Chrome/Chromium/Edge found for HTML→PDF.", file=sys.stderr)
        return False

    with tempfile.TemporaryDirectory() as tmp:
        html_path = Path(tmp) / "report.html"
        cmd = [
            "pandoc", str(md_path), "-s", "--from", "gfm", "--to", "html5",
            "--metadata", "title=Gemini OCR Service — Report",
            "-o", str(html_path),
        ]
        if _STYLE_HEADER.exists():
            cmd += ["--include-in-header", str(_STYLE_HEADER)]
        subprocess.run(cmd, check=True)
        subprocess.run(
            [
                chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={pdf_path}", html_path.as_uri(),
            ],
            check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    return pdf_path.exists()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the Gemini OCR service report")
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run Flash + Pro over sample_imgs/ and inject live numbers (needs API key)",
    )
    parser.add_argument(
        "--models", nargs="+", default=["gemini-2.5-flash", "gemini-2.5-pro"],
        help="Models to benchmark (default: flash + pro)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of sample images used in the benchmark",
    )
    parser.add_argument("--html", action="store_true", help="Also emit a styled HTML file")
    parser.add_argument(
        "--pdf", action="store_true",
        help="Also emit a print-ready PDF (needs pandoc + Chrome; renders Khmer)",
    )
    parser.add_argument(
        "-o", "--out", default=str(REPO_ROOT),
        help="Output directory (default: repo root)",
    )
    parser.add_argument(
        "--name", default="gemini-ocr-service-report",
        help="Output file stem (default: gemini-ocr-service-report)",
    )
    args = parser.parse_args()

    results = None
    if args.benchmark:
        sys.path.insert(0, str(REPO_ROOT))  # so `gemini` package imports work
        images = _list_sample_images(args.limit)
        if not images:
            print(f"No images found in {SAMPLE_DIR}", file=sys.stderr)
            return 1
        print(f"Benchmarking {len(images)} image(s) across: {', '.join(args.models)}")
        results = {}
        for model in args.models:
            print(f"\n== {model} ==")
            results[model] = benchmark_model(model, images)
        print()

    report_md = build_report(results)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{args.name}.md"
    md_path.write_text(report_md, encoding="utf-8")
    print(f"Wrote {md_path}")

    if args.html:
        html = render_html(report_md)
        if html is None:
            print(
                "HTML skipped — install the renderer: pip install markdown",
                file=sys.stderr,
            )
        else:
            html_path = out_dir / f"{args.name}.html"
            html_path.write_text(html, encoding="utf-8")
            print(f"Wrote {html_path}")

    if args.pdf:
        pdf_path = out_dir / f"{args.name}.pdf"
        if render_pdf(md_path, pdf_path):
            print(f"Wrote {pdf_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
