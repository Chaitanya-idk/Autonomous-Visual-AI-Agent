#!/usr/bin/env bash
# ============================================================
# SAGE QLoRA Pipeline — GCP Setup Script
# Automated setup for GCP Compute Engine / Vertex AI GPU instances
#
# Usage:
#   export GCS_BUCKET="your-sage-bucket-name"
#   ./scripts/gcp_setup.sh
#
# Or with bucket as argument:
#   ./scripts/gcp_setup.sh your-sage-bucket-name
# ============================================================

set -e

BUCKET="${1:-$GCS_BUCKET}"

echo "============================================================"
echo "  SAGE QLoRA — Google Cloud Environment Setup"
echo "============================================================"

# 1. Verify NVIDIA GPU availability
echo ""
echo "[STEP 1] GPU Detection..."
if command -v nvidia-smi &> /dev/null; then
    echo "[OK] NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "[WARNING] nvidia-smi not found. Ensure GPU drivers are installed."
fi

# 2. Upgrade pip and install Python dependencies
echo ""
echo "[STEP 2] Installing Python dependencies..."
python3 -m pip install --upgrade pip --quiet
python3 -m pip install -r requirements.txt

echo "[OK] All dependencies installed."

# 3. Create required project directory structure
echo ""
echo "[STEP 3] Creating project directory structure..."
mkdir -p data/processed/images
mkdir -p SAGE-sample
mkdir -p checkpoints
mkdir -p outputs
mkdir -p logs

echo "[OK] Directory structure ready."

# 4. Pull dataset from Google Cloud Storage
if [ -z "$BUCKET" ]; then
    echo ""
    echo "[SKIP] GCS_BUCKET not set — skipping data download."
    echo "       To download data, run: export GCS_BUCKET=your-bucket-name && ./scripts/gcp_setup.sh"
else
    echo ""
    echo "[STEP 4] Downloading dataset from gs://$BUCKET ..."

    # 4a. Download raw SAGE-sample Arrow files (source dataset, ~2 GB)
    echo "  Downloading SAGE-sample (raw Arrow files, ~2 GB)..."
    gcloud storage cp -r "gs://$BUCKET/SAGE-sample/" . \
        && echo "  [OK] SAGE-sample downloaded." \
        || echo "  [INFO] SAGE-sample not found in bucket (may need to upload first)."

    # 4b. Download extracted images (~1.8 GB)
    # NOTE: Skip this if you plan to re-extract from SAGE-sample instead
    echo "  Downloading processed images (~1.8 GB)..."
    gcloud storage cp -r "gs://$BUCKET/data/processed/images/" data/processed/ \
        && echo "  [OK] Images downloaded." \
        || echo "  [INFO] Images not found in bucket."

    # 4c. Download CSV splits and labels (tiny, fast)
    echo "  Downloading CSV splits and labels.json..."
    for file in train.csv val.csv test.csv metadata.csv labels.json; do
        gcloud storage cp "gs://$BUCKET/data/processed/$file" data/processed/ 2>/dev/null \
            && echo "  [OK] $file downloaded." \
            || echo "  [INFO] $file not found in bucket."
    done

    # 4d. Download best checkpoint if it exists (optional, for inference / resumed training)
    if gcloud storage ls "gs://$BUCKET/checkpoints/" &>/dev/null; then
        echo ""
        echo "  Found checkpoints in GCS. Downloading..."
        gcloud storage cp -r "gs://$BUCKET/checkpoints/" . \
            && echo "  [OK] Checkpoints downloaded." \
            || echo "  [INFO] Could not download checkpoints."
    fi

    echo "[OK] GCS data sync complete."
fi

# 5. Verify pipeline
echo ""
echo "[STEP 5] Verifying pipeline imports..."
python3 -c "from src.dataset import SAGEDataset; from src.preprocessing import get_processor; print('[OK] Core imports verified.')"

echo ""
echo "============================================================"
echo "  GCP Setup Complete!"
echo "============================================================"
echo ""
echo "  Next steps:"
echo "    Verify full pipeline:  python3 -m src.dataset"
echo "    Run preflight test:    python3 -m src.train --preflight"
echo "    Start full training:   python3 -m src.train"
echo "    Sync outputs to GCS:   ./scripts/sync_to_gcs.sh $BUCKET"
echo ""
