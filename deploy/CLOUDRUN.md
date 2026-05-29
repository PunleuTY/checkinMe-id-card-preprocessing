# Deploy the Gemini OCR service to Google Cloud Run

Serverless, pay-per-request hosting for the FastAPI NID service so the Laravel
backend (`GeminiService::scanNid()`) can call it over HTTPS.

## What's in this pack

| File | Purpose |
|---|---|
| `Dockerfile` | Cloud Run-ready image. Binds `$PORT`, installs `libglib2.0-0` for opencv. |
| `.dockerignore` | Trims the build context (no `.venv`, `sample_imgs`, Laravel folders) for fast cold starts. |
| `requirements.txt` | **Now includes `google-genai`** — the prod endpoint needs it (was missing). |
| `deploy/deploy.sh` | One command: enable APIs → Secret Manager → deploy → smoke test. |
| `deploy/laravel.env.example` | The `.env` values for the Laravel side. |

## One-time prerequisites

```bash
# install gcloud (macOS): brew install --cask google-cloud-sdk
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

The Gemini API key is read automatically from the repo's `.env` (the
`GEMINI_API_KEY=` line), or you'll be prompted for it. It is stored in Secret
Manager and mounted as an env var at runtime — never baked into the image.

## Deploy

```bash
chmod +x deploy/deploy.sh
./deploy/deploy.sh
```

Override any setting via env vars, e.g. keep one instance warm (no cold starts):

```bash
MIN_INSTANCES=1 REGION=asia-southeast1 ./deploy/deploy.sh
```

When it finishes it prints the service URL and the exact Laravel `.env` lines.

## Wire up Laravel

Paste the printed values into your Laravel `.env` (see `deploy/laravel.env.example`):

```dotenv
GEMINI_SERVICE_BASE_URL=https://gemini-ocr-xxxxxxxxxx-as.a.run.app
GEMINI_SERVICE_TIMEOUT=60
GEMINI_SERVICE_MODEL=gemini-2.5-flash
GEMINI_SERVICE_ENDPOINTS_SCAN_NID=/gemini-ocr/upload
```

> `GeminiService::scanNid()` only hits the real service when
> `app()->environment() == 'production'`; otherwise it returns mock fields.
> Set `APP_ENV=production` (or bypass that guard) to test against live Cloud Run.

## Verify

```bash
URL=$(gcloud run services describe gemini-ocr --region asia-southeast1 --format='value(status.url)')
curl "$URL/health"                                              # {"status":"ok"}
curl -X POST "$URL/gemini-ocr/upload" \
  -F "file=@sample_imgs/id2.jpg" -F "model=gemini-2.5-flash"    # full round-trip
```

## Tuning (deploy.sh env vars)

| Var | Default | Notes |
|---|---|---|
| `MEMORY` | `1Gi` | opencv+numpy load at startup (~300 MB). Don't go below 512Mi. |
| `CONCURRENCY` | `8` | Requests are IO-bound on Gemini; one instance holds many cheaply. Matches gunicorn `--threads 8`. |
| `TIMEOUT` | `120` | Keep ≥ Laravel `GEMINI_SERVICE_TIMEOUT`. |
| `MIN_INSTANCES` | `0` | `0` = pay-per-request (cold start ~3–6s). `1` = always warm. |
| `MAX_INSTANCES` | `5` | Caps cost and protects your Gemini quota. |
| `ALLOW_UNAUTH` | `true` | Public URL — simplest for dev. See below to lock down. |

## Locking down for production (optional)

The dev deploy is public (`--allow-unauthenticated`), meaning anyone with the URL
can spend your Gemini key. To require IAM auth instead:

1. Redeploy with `ALLOW_UNAUTH=false ./deploy/deploy.sh`.
2. Grant your Laravel service account the invoker role:
   ```bash
   gcloud run services add-iam-policy-binding gemini-ocr --region asia-southeast1 \
     --member="serviceAccount:laravel-caller@PROJECT.iam.gserviceaccount.com" \
     --role="roles/run.invoker"
   ```
3. Have Laravel attach a Google OIDC identity token (audience = the service URL)
   on each request — `composer require google/auth`, then add an
   `Authorization: Bearer <id-token>` header in `GeminiService`.

## Cost

Cloud Run bills CPU+memory only while a request is in flight, per 100ms, and is
free when idle (`MIN_INSTANCES=0`). For bursty ID scans you'll likely stay within
the monthly free tier — the real cost driver is the Gemini API calls themselves.
