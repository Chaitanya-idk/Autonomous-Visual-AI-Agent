"""
Final Evaluation Pipeline for SAGE Crop Disease Diagnosis.
Evaluates the best trained QLoRA checkpoint on the test set.
Calculates overall and per-class metrics, audits vocabulary integrity,
and saves granular machine-readable predictions.
"""

import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import Counter
from sklearn.metrics import classification_report

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_labels_vocab, compute_classification_metrics
from src.preprocessing import get_processor
from src.dataset import SAGEDataset, sage_collate_fn
from src.model import load_trained_lora_model
from src.train import match_prediction_to_vocab, load_config


def verify_vocabulary_integrity(
    labels_path: Path,
    train_csv: Path,
    val_csv: Path,
    test_csv: Path
) -> Dict[str, Any]:
    """
    PHASE 9 — Label Vocabulary Integrity
    Ensures train, val, and test labels strictly belong to the single authoritative labels.json.
    """
    if not labels_path.exists():
        raise FileNotFoundError(f"Authoritative labels file missing: {labels_path}")

    with open(labels_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    label2id = vocab_data["label2id"]
    vocab_classes = set(label2id.keys())

    train_df = pd.read_csv(train_csv, engine="python")
    val_df = pd.read_csv(val_csv, engine="python")
    test_df = pd.read_csv(test_csv, engine="python")

    train_labels = set(train_df["disease"].unique())
    val_labels = set(val_df["disease"].unique())
    test_labels = set(test_df["disease"].unique())

    mismatches = {
        "train_unseen": list(train_labels - vocab_classes),
        "val_unseen": list(val_labels - vocab_classes),
        "test_unseen": list(test_labels - vocab_classes),
    }

    has_mismatch = any(len(v) > 0 for v in mismatches.values())
    if has_mismatch:
        print("[CRITICAL ERROR] Label vocabulary mismatch detected:")
        for k, v in mismatches.items():
            if len(v) > 0:
                print(f"  {k}: {v}")
        raise ValueError("Vocabulary integrity check failed. Halting evaluation.")

    print(f"[OK] Label vocabulary integrity verified across all splits ({len(vocab_classes)} classes in vocabulary).")
    return {
        "vocab_size": len(vocab_classes),
        "train_classes": len(train_labels),
        "val_classes": len(val_labels),
        "test_classes": len(test_labels),
        "status": "PASS"
    }


def evaluate_test_set(
    checkpoint_path: str,
    config: Optional[Dict[str, Any]] = None,
    output_subdir: str = "001_baseline_qlora"
):
    """
    PHASE 6, 7, 8 — Evaluation on test.csv
    Loads LoRA checkpoint, runs inference on all test images,
    computes full metric suites and exports predictions.
    """
    if config is None:
        config = load_config()

    labels_path = PROJECT_ROOT / config["paths"]["labels_json"]
    train_csv = PROJECT_ROOT / config["paths"]["train_csv"]
    val_csv = PROJECT_ROOT / config["paths"]["val_csv"]
    test_csv = PROJECT_ROOT / config["paths"]["test_csv"]

    # Phase 9: Vocabulary check
    verify_vocabulary_integrity(labels_path, train_csv, val_csv, test_csv)

    label2id, id2label = load_labels_vocab(labels_path)

    out_dir = PROJECT_ROOT / config["paths"]["output_dir"] / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading LoRA model from checkpoint: {checkpoint_path}")
    model = load_trained_lora_model(
        adapter_path=checkpoint_path,
        base_model_name_or_path=config["model"]["name_or_path"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    prep_cfg = config.get("preprocessing", {})
    processor = get_processor(
        min_pixels=prep_cfg.get("min_pixels", 100352),
        max_pixels=prep_cfg.get("max_pixels", 200704)
    )

    test_dataset = SAGEDataset(str(test_csv), processor=processor, is_training=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=sage_collate_fn
    )

    print(f"Running inference on {len(test_dataset)} test samples...")

    prediction_records = []
    y_true = []
    y_pred = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
            image_grid_thw = batch["image_grid_thw"].to(device)

            image_id = batch["image_id"]
            crop = batch["crop"]
            true_disease = batch["disease"]
            true_label_id = batch["label_id"]

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=32,
                do_sample=False
            )

            prompt_len = input_ids.shape[1]
            new_tokens = generated_ids[0, prompt_len:]
            raw_text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            pred_label, pred_id, status = match_prediction_to_vocab(raw_text, label2id)
            is_correct = (pred_label == true_disease)

            record = {
                "image_id": image_id,
                "crop": crop,
                "true_label": true_disease,
                "true_label_id": true_label_id,
                "predicted_label": pred_label,
                "predicted_label_id": pred_id,
                "correct": is_correct,
                "raw_model_output": raw_text,
                "prediction_status": status
            }
            prediction_records.append(record)
            y_true.append(true_label_id)
            y_pred.append(pred_id)

            if i % 25 == 0 or i == len(test_dataset):
                print(f"  Processed {i}/{len(test_dataset)} test images...")

    pred_df = pd.DataFrame(prediction_records)
    predictions_csv = out_dir / "test_predictions.csv"
    pred_df.to_csv(predictions_csv, index=False)
    print(f"[OK] Saved test predictions to: {predictions_csv}")

    # Compute overall metrics
    overall_metrics = compute_classification_metrics(y_true, y_pred)

    # Per-class metrics
    present_class_ids = sorted(list(set(y_true) | set(y_pred)))
    # filter out -1 for unknown
    eval_class_ids = [cid for cid in present_class_ids if cid >= 0]
    eval_class_names = [id2label[cid] for cid in eval_class_ids]

    cls_report = classification_report(
        y_true,
        y_pred,
        labels=eval_class_ids,
        target_names=eval_class_names,
        output_dict=True,
        zero_division=0
    )

    per_class_rows = []
    zero_recall_classes = []
    zero_precision_classes = []
    never_predicted_classes = []
    small_support_classes = []

    for cid, cname in zip(eval_class_ids, eval_class_names):
        entry = cls_report.get(cname, {})
        prec = entry.get("precision", 0.0)
        rec = entry.get("recall", 0.0)
        f1 = entry.get("f1-score", 0.0)
        supp = entry.get("support", 0)

        per_class_rows.append({
            "class_id": cid,
            "disease": cname,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "support": supp
        })

        if supp > 0 and rec == 0.0:
            zero_recall_classes.append(cname)
        if prec == 0.0 and (y_pred.count(cid) > 0 or supp > 0):
            zero_precision_classes.append(cname)
        if y_pred.count(cid) == 0:
            never_predicted_classes.append(cname)
        if supp > 0 and supp < 3:
            small_support_classes.append({"disease": cname, "support": supp})

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_csv = out_dir / "per_class_metrics.csv"
    per_class_df.to_csv(per_class_csv, index=False)
    print(f"[OK] Saved per-class metrics to: {per_class_csv}")

    analysis_summary = {
        "overall_metrics": overall_metrics,
        "classes_with_zero_recall_count": len(zero_recall_classes),
        "classes_with_zero_recall": zero_recall_classes,
        "classes_with_zero_precision_count": len(zero_precision_classes),
        "classes_with_zero_precision": zero_precision_classes,
        "classes_never_predicted_count": len(never_predicted_classes),
        "classes_never_predicted": never_predicted_classes,
        "classes_with_small_support_count": len(small_support_classes),
        "classes_with_small_support": small_support_classes,
        "total_test_samples": len(test_dataset),
        "unknown_predictions_count": int((pred_df["prediction_status"] == "unknown").sum())
    }

    metrics_json = out_dir / "test_metrics.json"
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(analysis_summary, f, indent=2)
    print(f"[OK] Saved test summary metrics to: {metrics_json}")

    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Accuracy:              {overall_metrics['accuracy']:.4f}")
    print(f"Macro Precision:       {overall_metrics['macro_precision']:.4f}")
    print(f"Macro Recall:          {overall_metrics['macro_recall']:.4f}")
    print(f"Macro F1:              {overall_metrics['macro_f1']:.4f}")
    print(f"Micro F1:              {overall_metrics['micro_f1']:.4f}")
    print(f"Weighted F1:           {overall_metrics['weighted_f1']:.4f}")
    print(f"Zero-recall classes:   {len(zero_recall_classes)}")
    print(f"Never-predicted:       {len(never_predicted_classes)}")
    print(f"Unknown predictions:   {analysis_summary['unknown_predictions_count']}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained QLoRA on SAGE test set")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/001_baseline_qlora/best_checkpoint",
        help="Path to saved LoRA adapter checkpoint"
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path

    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint path does not exist: {ckpt_path}")
        print("Run training first (python -m src.train) before running evaluation.")
        sys.exit(1)

    evaluate_test_set(str(ckpt_path))
