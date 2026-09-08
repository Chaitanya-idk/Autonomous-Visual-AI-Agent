# SAGE — Crop Disease Visual Diagnosis

QLoRA fine-tuning of **Qwen2.5-VL-3B-Instruct** on the SAGE crop disease image dataset for automated visual diagnosis.

## Architecture

| Component | Detail |
|-----------|--------|
| Base model | `Qwen2.5-VL-3B-Instruct` (local) |
| Quantisation | 4-bit NF4 via `bitsandbytes` |
| Fine-tuning | QLoRA — LoRA on `q_proj, k_proj, v_proj, o_proj` |
| Task | Multi-class crop disease classification from images |
| Hardware | NVIDIA RTX 3050 Ti (4 GB VRAM) |

## Project Structure

```
dataset_loader/
├── configs/
│   └── config.yaml          # All hyperparameters (model, LoRA, training)
├── data/
│   ├── processed/           # ← gitignored, regeneratable
│   │   ├── images/          #   Extracted JPEGs (000001.jpg …)
│   │   ├── metadata.csv     #   Full image manifest
│   │   ├── train.csv        #   Training split
│   │   ├── val.csv          #   Validation split
│   │   └── test.csv         #   Test split
│   └── reports/             # ← gitignored, audit/split reports
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
│   └── utils.py             # Seeds, metrics, vocab helpers
├── checkpoints/             # ← gitignored, saved LoRA adapters
├── outputs/                 # ← gitignored, inference results
└── logs/                    # ← gitignored, training logs
```

> **Note:** The `SAGE-sample/` raw dataset (~2.2 GB HuggingFace arrow files) is **gitignored**.  
> Place it at `dataset_loader/SAGE-sample/` before running the pipeline.

## Quick Start

### 1 — Set up environment

```bash
conda activate ai
```

### 2 — Regenerate processed data (if needed)

```bash
# From dataset_loader/
python -m src.inspect_dataset
python -m src.extract_dataset
python -m src.audit_dataset
python -m src.prepare_splits
```

### 3 — Verify the dataset pipeline

```bash
python -m src.dataset
```

Expected: `[DATASET TEST] ALL CHECKS PASSED` with **Active training target tokens > 0**.

### 4 — Train

```bash
python -m src.train
```

### 5 — Evaluate

```bash
python -m src.evaluate
```

## Key Design Decisions

- **`engine='python'`** in `pd.read_csv` — avoids a Windows MKL DLL conflict when `torch` is loaded before pandas' C extension.
- **Pandas imported before torch** at module level in `dataset.py` — same reason.
- **Batch size = 1** enforced — RTX 3050 Ti with 4 GB VRAM cannot handle larger batches with a 3B VLM even at 4-bit.
- **Label masking** — only disease-name tokens (assistant turn) are unmasked (`-100` everywhere else) so the model is not penalised for the prompt or visual tokens.
- **Resolution = LOW** (min_pixels=256²·4, max_pixels=512²·4) — the benchmark showed this safely fits in 4 GB VRAM.
