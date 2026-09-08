"""
Automated Baseline Report Generator for SAGE Crop Disease Diagnosis.
Produces outputs/001_baseline_qlora/baseline_report.md summarizing
dataset, model, training parameters, validation selection, test metrics,
and error patterns.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def generate_baseline_report(run_id: str = "001_baseline_qlora") -> Path:
    out_dir = PROJECT_ROOT / "outputs" / run_id
    report_file = out_dir / "baseline_report.md"

    cfg_file = out_dir / "config.json"
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        from src.train import load_config
        config = load_config()

    # Load dataset statistics
    train_df = pd.read_csv(PROJECT_ROOT / config["paths"]["train_csv"], engine="python")
    val_df = pd.read_csv(PROJECT_ROOT / config["paths"]["val_csv"], engine="python")
    test_df = pd.read_csv(PROJECT_ROOT / config["paths"]["test_csv"], engine="python")
    with open(PROJECT_ROOT / config["paths"]["labels_json"], "r", encoding="utf-8") as f:
        labels_data = json.load(f)

    total_images = len(train_df) + len(val_df) + len(test_df)
    num_classes = len(labels_data["label2id"])
    crops_dist = train_df["crop"].value_counts().to_dict()

    # Load test metrics if available
    test_metrics_file = out_dir / "test_metrics.json"
    test_metrics = {}
    if test_metrics_file.exists():
        with open(test_metrics_file, "r", encoding="utf-8") as f:
            test_metrics = json.load(f)

    # Load run summary if available
    run_summary_file = out_dir / "run_summary.json"
    run_summary = {}
    if run_summary_file.exists():
        with open(run_summary_file, "r", encoding="utf-8") as f:
            run_summary = json.load(f)

    overall = test_metrics.get("overall_metrics", {})

    report_content = f"""# SAGE Crop Disease Diagnosis — QLoRA Baseline Report
**Run ID:** `{run_id}`  
**Model:** `Qwen2.5-VL-3B-Instruct` (4-bit NF4 QLoRA)  

---

## 1. Dataset Overview
- **Total Processed Images:** {total_images}
- **Training Set:** {len(train_df)} samples ({len(train_df['disease'].unique())} unique classes)
- **Validation Set:** {len(val_df)} samples ({len(val_df['disease'].unique())} unique classes)
- **Test Set:** {len(test_df)} samples ({len(test_df['disease'].unique())} unique classes)
- **Total Authoritative Disease Classes:** {num_classes}
- **Crop Distribution (Train):** {crops_dist}
- **Class Imbalance:** Significant long-tail distribution with rare classes having fewer than 3 samples. Stratified splitting preserves class presence across splits.

## 2. Model & Quantization Architecture
- **Base Architecture:** `Qwen2.5-VL-3B-Instruct`
- **Quantization:** 4-bit NormalFloat4 (NF4) with double quantization
- **Compute Dtype:** `torch.float16`
- **LoRA Targets:** `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **LoRA Rank (r):** {config.get('lora', {}).get('r', 16)} | **Alpha:** {config.get('lora', {}).get('lora_alpha', 32)} | **Dropout:** {config.get('lora', {}).get('lora_dropout', 0.05)}
- **Trainable Status:** Base model completely frozen; only LoRA adapters are updated (< 0.5% trainable parameters).

## 3. Training Configuration
- **Batch Size:** {config.get('training', {}).get('batch_size', 1)}
- **Gradient Accumulation Steps:** {config.get('training', {}).get('gradient_accumulation_steps', 8)} (Effective batch size: {config.get('training', {}).get('batch_size', 1) * config.get('training', {}).get('gradient_accumulation_steps', 8)})
- **Learning Rate:** {config.get('training', {}).get('learning_rate', 2e-4)}
- **Warmup Ratio:** {config.get('training', {}).get('warmup_ratio', 0.05)}
- **Visual Resolution Budget:** min_pixels={config.get('preprocessing', {}).get('min_pixels', 100352)}, max_pixels={config.get('preprocessing', {}).get('max_pixels', 200704)}
- **Precision:** FP16 with AMP Autocast
- **Gradient Checkpointing:** {config.get('training', {}).get('gradient_checkpointing', True)}
- **Random Seed:** {config.get('training', {}).get('seed', 42)}

## 4. Validation & Checkpoint Selection
- **Selection Metric:** Primary model selection based on **Validation Macro F1**
- **Best Epoch:** {run_summary.get('best_epoch', 'N/A')}
- **Best Validation Macro F1:** {run_summary.get('best_val_macro_f1', 'N/A')}
- **Checkpoint Location:** `{run_summary.get('best_checkpoint_path', f'checkpoints/{run_id}/best_checkpoint')}`

## 5. Final Test Performance
| Metric | Value |
|---|---|
| **Accuracy** | {overall.get('accuracy', 'Pending Evaluation')} |
| **Macro Precision** | {overall.get('macro_precision', 'Pending Evaluation')} |
| **Macro Recall** | {overall.get('macro_recall', 'Pending Evaluation')} |
| **Macro F1** | {overall.get('macro_f1', 'Pending Evaluation')} |
| **Micro F1** | {overall.get('micro_f1', 'Pending Evaluation')} |
| **Weighted F1** | {overall.get('weighted_f1', 'Pending Evaluation')} |

## 6. Error Analysis & Key Observations
- **Zero-Recall Classes:** {test_metrics.get('classes_with_zero_recall_count', 'N/A')} classes in test set had zero correct predictions.
- **Never-Predicted Classes:** {test_metrics.get('classes_never_predicted_count', 'N/A')} classes were never generated by the model.
- **Unknown Predictions:** {test_metrics.get('unknown_predictions_count', 'N/A')} model outputs could not be matched to the authoritative vocabulary.
- **Figures:** Detailed confusion matrices and error CSVs saved under `{out_dir / 'figures'}`.

## 7. Limitations & Resource Constraints
1. **Multi-Class Sparsity:** 294 disease classes with limited support per class creates high macro F1 sensitivity to individual misclassifications.
2. **Generative Diagnosis:** Generating disease tokens requires post-hoc mapping to vocabulary; hallucinations or slight phrasing deviations are flagged as unknown.
3. **GPU Constraints:** Hardware budget of ~4GB VRAM restricts batch size to 1 and enforces low visual resolution presets.
"""

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[OK] Saved baseline report to: {report_file}")
    return report_file


if __name__ == "__main__":
    generate_baseline_report()
