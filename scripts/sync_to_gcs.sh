#!/usr/bin/env bash
# ============================================================
# SAGE QLoRA Pipeline — Google Cloud Storage (GCS) Sync Script
# Syncs checkpoints, training outputs, and logs to a GCS bucket
# ============================================================

set -e

if [ -z "$1" ] && [ -z "$GCS_BUCKET" ]; then
    echo "Usage: ./scripts/sync_to_gcs.sh <your-gcs-bucket-name>"
    echo "Or set environment variable: export GCS_BUCKET=your-bucket-name"
    exit 1
fi

BUCKET="${1:-$GCS_BUCKET}"
BUCKET="${BUCKET#gs://}"  # strip prefix if provided

echo "=== Syncing SAGE training artifacts to gs://$BUCKET ==="

# Sync checkpoints
if [ -d "checkpoints" ]; then
    echo "Syncing checkpoints..."
    gsutil -m rsync -r checkpoints "gs://$BUCKET/checkpoints/"
fi

# Sync outputs & evaluation metrics
if [ -d "outputs" ]; then
    echo "Syncing outputs..."
    gsutil -m rsync -r outputs "gs://$BUCKET/outputs/"
fi

# Sync logs
if [ -d "logs" ]; then
    echo "Syncing logs..."
    gsutil -m rsync -r logs "gs://$BUCKET/logs/"
fi

echo "=== GCS Sync Complete ==="
