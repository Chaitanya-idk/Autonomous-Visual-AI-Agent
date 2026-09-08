# SAGE — Crop Disease Visual Diagnosis

QLoRA fine-tuning of **Qwen2.5-VL-3B-Instruct** on the SAGE crop disease image dataset for automated visual diagnosis. Built for 100% offline local execution and Cloud migration (Google Cloud Platform / Compute Engine / Vertex AI).

## Architecture & Tech Stack

| Component | Detail |
|-----------|--------|
| Base model | `Qwen2.5-VL-3B-Instruct` (local / HF hub) |
| Quantisation | 4-bit NF4 via `bitsandbytes` |
| Fine-tuning | QLoRA — adapters on `q_proj, k_proj, v_proj, o_proj` |
| Task | Multi-class crop disease visual classification |
| Supported Hardware | Local GPU (RTX 3050 Ti 4GB+) or GCP VM (T4, L4, V100, A100) |

## Project Structure

```
dataset_loader/
├── configs/
│   └── config.yaml          # All hyperparameters (model, LoRA, visual tokens, training)
├── data/
│   ├── processed/           # ← gitignored, regeneratable
│   │   ├── images/          #   Extracted JPEGs (000001.jpg …)
│   │   ├── metadata.csv     #   Full image manifest
│   │   ├── train.csv        #   Training split
│   │   ├── val.csv          #   Validation split
│   │   └── test.csv         #   Test split
│   └── reports/             # ← gitignored, audit/split reports
├── scripts/
│   ├── gcp_setup.sh         # GCP VM / Vertex AI automated setup script
│   └── sync_to_gcs.sh       # Sync checkpoints and logs to Google Cloud Storage (GCS)
├── src/
│   ├── __init__.py
│   ├── inspect_dataset.py   # Step 1 — inspect raw HF dataset
│   ├── extract_dataset.py   # Step 2 — extract images + metadata.csv
│   ├── audit_dataset.py     # Step 3 — quality audit
│   ├── prepare_splits.py    # Step 4 — stratified train/val/test split
│   ├── preprocessing.py     # Qwen processor + resolution benchmarking
│   ├── prompts.py           # Conversation template builder
│   ├── dataset.py           # PyTorch Dataset + DataLoader
│   ├── model.py             # QLoRA model loading (4-bit NF4)
│   ├── benchmark_vram.py    # Visual token memory benchmarking module
│   ├── benchmark_memory.py  # Single-step memory benchmark
│   ├── train.py             # Sanity check, preflight test & full baseline training
│   ├── evaluate.py          # Test set evaluation & metrics calculation
│   ├── plot_results.py      # Plotting confusion matrices & error analysis
│   └── report.py            # Final summary report generator
├── requirements.txt         # Production dependencies for Local & GCP
├── .gitignore               # Comprehensive gitignore for Python, GCP, models & datasets
├── checkpoints/             # ← gitignored, saved LoRA adapters
├── outputs/                 # ← gitignored, evaluation predictions & figures
└── logs/                    # ← gitignored, training logs
```

## Quick Start (Local Setup)

### 1 — Environment & Installation

```bash
# Clone the repository
git clone https://github.com/Chaitanya-idk/Autonomous-Visual-AI-Agent.git
cd Autonomous-Visual-AI-Agent/dataset_loader

# Create conda environment or virtualenv
conda create -n ai python=3.12 -y
conda activate ai

# Install requirements
pip install -r requirements.txt
```

---

## ☁️ Google Cloud Platform (GCP) Migration Guide

### Option A — GCP Compute Engine VM (Recommended)

1. **Spin up a GPU VM Instance**:
   - Machine Type: `n1-standard-4` or `g2-standard-4`
   - GPU: 1x NVIDIA T4 (16 GB VRAM) or 1x NVIDIA L4 (24 GB VRAM)
   - OS Image: **Deep Learning VM Image (CUDA 12.1 / PyTorch 2.1)**

2. **Clone repo & Run Automated GCP Setup**:
   ```bash
   git clone https://github.com/Chaitanya-idk/Autonomous-Visual-AI-Agent.git
   cd Autonomous-Visual-AI-Agent/dataset_loader
   chmod +x scripts/*.sh
   ./scripts/gcp_setup.sh
   ```

3. **Google Cloud Storage (GCS) Integration**:
   - Upload dataset or checkpoints to GCS:
     ```bash
     export GCS_BUCKET="your-sage-bucket-name"
     ./scripts/sync_to_gcs.sh
     ```

### Option B — Vertex AI Workbench / Custom Container

- Ensure container environment specifies `transformers>=4.48.0`, `bitsandbytes>=0.43.0`, `peft>=0.10.0`.
- All outputs will auto-save to `checkpoints/` and `outputs/`.

---

## Training Pipeline Commands

### 1 — Dataset Processing & Pipeline Verification
```bash
python -m src.extract_dataset
python -m src.prepare_splits
python -m src.dataset
```

### 2 — Run 50-Step Preflight Check (Safety & Memory Verification)
```bash
python -m src.train --preflight
```

### 3 — Run Baseline QLoRA Training
```bash
python -m src.train
```

### 4 — Evaluate on Test Set & Plot Results
```bash
python -m src.evaluate
python -m src.plot_results
python -m src.report
```

## Key Technical Decisions

- **Resolution Bounded (32–64 tokens)**: `min_pixels=25088`, `max_pixels=50176` ensures peak VRAM stays safe (~3.83 GB) for low-memory GPUs, while scaling seamlessly on cloud GPUs (T4/L4/A100).
- **PagedAdamW8bit & FP16**: Reduced memory footprint for optimizer state.
- **`engine='python'` in CSV parsing**: Prevents Windows MKL / native C parser DLL conflicts.
