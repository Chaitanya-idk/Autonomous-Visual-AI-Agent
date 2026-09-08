#!/usr/bin/env bash
# ============================================================
# SAGE QLoRA Pipeline — Google Cloud Storage (GCS) Upload Script
# Uploads your local dataset and/or training outputs to GCS.
#
# Usage:
#   ./scripts/upload_to_gcs.sh <bucket-name> [--data-only | --outputs-only | --all]
#
# Modes:
#   --data-only     : Upload SAGE-sample + data/processed only
#   --outputs-only  : Upload checkpoints + outputs + logs only
#   --all (default) : Upload everything
# ============================================================

set -e

if [ -z "$1" ] && [ -z "$GCS_BUCKET" ]; then
    echo "Usage: ./scripts/upload_to_gcs.sh <bucket-name> [--data-only | --outputs-only | --all]"
    echo "Or set: export GCS_BUCKET=your-bucket-name"
    exit 1
fi

BUCKET="${1:-$GCS_BUCKET}"
BUCKET="${BUCKET#gs://}"  # strip prefix if provided
MODE="${2:---all}"

echo "============================================================"
echo "  SAGE QLoRA — GCS Upload"
echo "  Destination: gs://$BUCKET"
echo "  Mode: $MODE"
echo "============================================================"

# ── Dataset Upload ────────────────────────────────────────────────────────────

upload_data() {
    echo ""
    echo "[DATA] Uploading raw SAGE-sample Arrow files (~2 GB)..."
    if [ -d "SAGE-sample" ]; then
        gcloud storage cp -r "SAGE-sample/" "gs://$BUCKET/SAGE-sample/" \
            && echo "  [OK] SAGE-sample uploaded." \
            || echo "  [WARN] SAGE-sample upload failed."
    else
        echo "  [SKIP] SAGE-sample directory not found."
    fi

    echo ""
    echo "[DATA] Uploading extracted images (~1.8 GB)..."
    if [ -d "data/processed/images" ]; then
        gcloud storage cp -r "data/processed/images/" "gs://$BUCKET/data/processed/images/" \
            && echo "  [OK] Images uploaded." \
            || echo "  [WARN] Images upload failed."
    else
        echo "  [SKIP] data/processed/images not found."
    fi

    echo ""
    echo "[DATA] Uploading CSV splits and labels (~0.4 MB)..."
    for file in train.csv val.csv test.csv metadata.csv labels.json; do
        if [ -f "data/processed/$file" ]; then
            gcloud storage cp "data/processed/$file" "gs://$BUCKET/data/processed/" \
                && echo "  [OK] $file uploaded."
        else
            echo "  [SKIP] $file not found."
        fi
    done
}

# ── Training Outputs Upload ───────────────────────────────────────────────────

upload_outputs() {
    echo ""
    echo "[OUTPUTS] Uploading checkpoints (LoRA adapters)..."
    if [ -d "checkpoints" ] && [ "$(ls -A checkpoints)" ]; then
        gcloud storage cp -r "checkpoints/" "gs://$BUCKET/checkpoints/" \
            && echo "  [OK] Checkpoints uploaded."
    else
        echo "  [SKIP] checkpoints/ is empty or missing."
    fi

    echo ""
    echo "[OUTPUTS] Uploading evaluation outputs..."
    if [ -d "outputs" ] && [ "$(ls -A outputs)" ]; then
        gcloud storage cp -r "outputs/" "gs://$BUCKET/outputs/" \
            && echo "  [OK] Outputs uploaded."
    else
        echo "  [SKIP] outputs/ is empty or missing."
    fi

    echo ""
    echo "[OUTPUTS] Uploading training logs..."
    if [ -d "logs" ] && [ "$(ls -A logs)" ]; then
        gcloud storage cp -r "logs/" "gs://$BUCKET/logs/" \
            && echo "  [OK] Logs uploaded."
    else
        echo "  [SKIP] logs/ is empty or missing."
    fi
}

# ── Dispatch by mode ─────────────────────────────────────────────────────────

case "$MODE" in
    --data-only)
        upload_data
        ;;
    --outputs-only)
        upload_outputs
        ;;
    *)
        upload_data
        upload_outputs
        ;;
esac

echo ""
echo "============================================================"
echo "  GCS Upload Complete!"
echo "  View contents: gcloud storage ls gs://$BUCKET/"
echo "============================================================"
