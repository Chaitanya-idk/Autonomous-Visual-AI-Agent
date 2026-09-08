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

## Quick Start & Reproducibility

### 1 — Environment Setup & Specifications

```bash
conda activate ai
```

- **Python:** `3.12.13`
- **PyTorch:** `2.5.1+cu121`
- **Transformers:** `5.5.4`
- **PEFT:** `0.20.0`
- **CUDA:** `12.1`
- **GPU:** NVIDIA GeForce RTX 3050 Ti Laptop GPU (~4 GB VRAM)
- **Random Seed:** `42` (configured in `configs/config.yaml`)

### 2 — Data Preparation (Optional / If Raw SAGE-sample is present)

```bash
python -m src.inspect_dataset
python -m src.extract_dataset
python -m src.audit_dataset
python -m src.prepare_splits
```

### 3 — Pipeline Verification

```bash
python -m src.dataset
```
Expected: `[DATASET TEST] ALL CHECKS PASSED` with **Active training target tokens: 8**.

### 4 — Sanity Test (16 Samples Overfit Check)

```bash
python -m src.train --sanity
```
Expected: `SANITY TEST: Result: PASS` confirming forward pass, finite loss, non-zero LoRA gradients, frozen base model, and parameter update.

### 5 — Baseline Training

```bash
python -m src.train
```
Runs 1 epoch with batch size 1 and gradient accumulation 8. Saves best adapter to `checkpoints/001_baseline_qlora/best_checkpoint`.

### 6 — Evaluation on Test Set

```bash
python -m src.evaluate
```
Computes overall accuracy, micro/macro/weighted metrics, per-class breakdown, and saves predictions to `outputs/001_baseline_qlora/test_predictions.csv`.

### 7 — Visualization & Error Analysis

```bash
python -m src.plot_results
```
Generates confusion matrix, focused confusion matrix, error analysis CSV, and loss/accuracy/F1 training curves under `outputs/001_baseline_qlora/figures/`.

## Key Design Decisions

- **`engine='python'`** in `pd.read_csv` — avoids a Windows MKL DLL conflict when `torch` is loaded before pandas' C extension.
- **Pandas imported before torch** at module level in `dataset.py` — avoids native-level parser crashes.
- **Batch size = 1** enforced — RTX 3050 Ti with 4 GB VRAM cannot handle larger batches with a 3B VLM even at 4-bit.
- **Label masking** — only disease-name tokens (assistant turn) are unmasked (`-100` everywhere else) so the model is not penalised for the prompt or visual tokens.
- **Resolution = LOW** (min_pixels=100352, max_pixels=200704) — bounded token budget safely fits within 4 GB VRAM.

