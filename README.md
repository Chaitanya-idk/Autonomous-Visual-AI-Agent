# SAGE — Crop Disease Visual Diagnosis

LoRA fine-tuning of **Qwen2.5-VL-3B-Instruct** on the [tirtho149/SAGE](https://huggingface.co/datasets/tirtho149/SAGE) dataset (21.4 GB, ~100K-1M images) for automated crop disease classification.

**Full precision (bfloat16) LoRA — no quantization — designed for GCP / Kaggle GPU instances.**

## Architecture

| Component | Detail |
|-----------|--------|
| Base model | `Qwen2.5-VL-3B-Instruct` |
| Fine-tuning | Full LoRA (r=32, α=64) on `q/k/v/o_proj` |
| Precision | bfloat16 (no quantization) |
| Dataset | tirtho149/SAGE — parquet format |
| Hardware | Kaggle T4 (16GB) / GCP L4 (24GB) / A100 |

## Repo Structure

```
├── configs/
│   └── config.yaml          # ← Edit this to configure paths, resolution, epochs
├── src/
│   ├── dataset.py           # Parquet → PyTorch Dataset (supports HF Hub + local)
│   ├── model.py             # Full LoRA model loading
│   ├── train.py             # Training loop + early stopping
│   ├── prompts.py           # Qwen2.5-VL conversation builder
│   └── utils.py             # Seeds, metrics
├── scripts/
│   ├── gcp_setup.sh         # GCP VM automated setup
│   ├── upload_to_gcs.sh     # Upload data to GCS bucket
│   └── sync_to_gcs.sh       # Sync outputs back to GCS
├── requirements.txt
└── .gitignore
```

## Kaggle Quick Start

### 1 — Add the SAGE dataset on Kaggle
1. Go to [tirtho149/SAGE on HuggingFace](https://huggingface.co/datasets/tirtho149/SAGE)
2. Download parquet files and add as a Kaggle dataset, **or** use the HuggingFace datasets API directly.

### 2 — Clone this repo into your Kaggle notebook
```python
!git clone https://github.com/Chaitanya-idk/Autonomous-Visual-AI-Agent.git
%cd Autonomous-Visual-AI-Agent/dataset_loader
!pip install -r requirements.txt
```

### 3 — Configure paths in `configs/config.yaml`
```yaml
dataset:
  use_hf_hub: false
  parquet_dir: "/kaggle/input/sage/data"   # ← path to your parquet files on Kaggle
  image_col: "image"                        # ← verify against your parquet schema
  disease_col: "disease"                    # ← verify against your parquet schema
```

### 4 — Run training
```bash
python -m src.train
```

---

## GCP Setup

```bash
export GCS_BUCKET="your-sage-bucket"
chmod +x scripts/*.sh
./scripts/gcp_setup.sh
python -m src.train
./scripts/sync_to_gcs.sh    # sync checkpoints after training
```

## Verify your parquet schema

Before running, check what columns your parquet files actually have:
```python
import pandas as pd
df = pd.read_parquet("/path/to/one/file.parquet", engine="pyarrow")
print(df.columns.tolist())
print(df.iloc[0])  # inspect first row + image column format
```

Update `configs/config.yaml` → `dataset.*_col` fields to match.

## Key Design Notes

- **No image extraction needed** — images are decoded directly from parquet bytes at runtime.
- **Early stopping** at patience=7 prevents overfitting on noisy agricultural images.
- **LoRA adapter checkpoints** are ~50–200 MB regardless of model size (base model NOT included).
- To produce a self-contained deployment model: set `save_mode: merged` in config (saves ~7 GB).
