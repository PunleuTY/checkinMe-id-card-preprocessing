#!/usr/bin/env bash
#
# One-command deploy of the Gemini OCR FastAPI service to Google Cloud Run.
#
# Usage:
#   ./deploy/deploy.sh                 # uses defaults below + your gcloud project
#   PROJECT_ID=my-proj REGION=asia-southeast1 ./deploy/deploy.sh
#
# Prereqs: gcloud CLI installed and `gcloud auth login` done once.
# The Gemini API key is read from (in order): $GEMINI_API_KEY, the repo .env,
# or an interactive prompt — and stored in Secret Manager (never baked into the image).
#
set -euo pipefail

# ---- config (override any of these via environment variables) ----
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
REGION="${REGION:-asia-southeast1}"        # Singapore — closest region to Cambodia
SERVICE="${SERVICE:-gemini-ocr}"
SECRET_NAME="${SECRET_NAME:-gemini-api-key}"
GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.5-flash}"
MEMORY="${MEMORY:-1Gi}"
CPU="${CPU:-1}"
CONCURRENCY="${CONCURRENCY:-8}"
TIMEOUT="${TIMEOUT:-120}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"        # 0 = scale to zero (pay per request)
MAX_INSTANCES="${MAX_INSTANCES:-5}"
ALLOW_UNAUTH="${ALLOW_UNAUTH:-true}"       # true = public URL (simplest for dev)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

bold() { printf "\033[1m%s\033[0m\n" "$1"; }

if [ -z "$PROJECT_ID" ]; then
  echo "ERROR: no GCP project set. Run: gcloud config set project YOUR_PROJECT_ID"
  exit 1
fi

bold "==> Project: $PROJECT_ID   Region: $REGION   Service: $SERVICE"

# ---- 1. enable required APIs (idempotent) ----
bold "==> Enabling APIs (run, cloudbuild, secretmanager, artifactregistry)…"
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com \
  artifactregistry.googleapis.com \
  --project "$PROJECT_ID"

# ---- 2. ensure the Gemini API key secret exists ----
if ! gcloud secrets describe "$SECRET_NAME" --project "$PROJECT_ID" >/dev/null 2>&1; then
  bold "==> Secret '$SECRET_NAME' not found — creating it."
  KEY="${GEMINI_API_KEY:-}"
  if [ -z "$KEY" ] && [ -f "$REPO_DIR/.env" ]; then
    KEY="$(grep -E '^GEMINI_API_KEY=' "$REPO_DIR/.env" | head -1 | cut -d= -f2- | tr -d '\r' || true)"
  fi
  if [ -z "$KEY" ]; then
    read -r -s -p "Enter your GEMINI_API_KEY: " KEY; echo
  fi
  if [ -z "$KEY" ]; then echo "ERROR: empty API key."; exit 1; fi
  printf "%s" "$KEY" | gcloud secrets create "$SECRET_NAME" \
    --data-file=- --project "$PROJECT_ID"
else
  bold "==> Secret '$SECRET_NAME' already exists — leaving it as is."
  echo "    (To rotate: printf '%s' NEWKEY | gcloud secrets versions add $SECRET_NAME --data-file=-)"
fi

# ---- 3. grant the Cloud Run runtime SA access to the secret ----
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
bold "==> Granting secretAccessor to $RUNTIME_SA"
gcloud secrets add-iam-policy-binding "$SECRET_NAME" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/secretmanager.secretAccessor" \
  --project "$PROJECT_ID" >/dev/null

# ---- 4. deploy ----
AUTH_FLAG="--allow-unauthenticated"
[ "$ALLOW_UNAUTH" = "true" ] || AUTH_FLAG="--no-allow-unauthenticated"

bold "==> Deploying to Cloud Run from source ($REPO_DIR)…"
gcloud run deploy "$SERVICE" \
  --source "$REPO_DIR" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --memory "$MEMORY" \
  --cpu "$CPU" \
  --concurrency "$CONCURRENCY" \
  --timeout "$TIMEOUT" \
  --min-instances "$MIN_INSTANCES" \
  --max-instances "$MAX_INSTANCES" \
  --set-secrets "GEMINI_API_KEY=${SECRET_NAME}:latest" \
  --set-env-vars "GEMINI_MODEL=${GEMINI_MODEL}" \
  $AUTH_FLAG

# ---- 5. report + smoke test ----
URL="$(gcloud run services describe "$SERVICE" --region "$REGION" \
       --project "$PROJECT_ID" --format='value(status.url)')"
echo
bold "==> Deployed: $URL"
echo "Health check:"
curl -fsS "$URL/health" && echo
echo
bold "==> Put this in your Laravel .env:"
echo "GEMINI_SERVICE_BASE_URL=$URL"
echo "GEMINI_SERVICE_TIMEOUT=60"
echo "GEMINI_SERVICE_MODEL=$GEMINI_MODEL"
echo "GEMINI_SERVICE_ENDPOINTS_SCAN_NID=/gemini-ocr/upload"
