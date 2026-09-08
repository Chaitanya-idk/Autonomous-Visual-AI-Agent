"""
Visualization and Error Analysis Pipeline for SAGE Crop Disease Diagnosis.
Generates full and focused confusion matrices, error analysis tables,
and training/validation curves.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict
from sklearn.metrics import confusion_matrix
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_labels_vocab


def plot_confusion_matrices(
    predictions_df: pd.DataFrame,
    figures_dir: Path,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    top_n_focused: int = 15
):
    """
    PHASE 10 — Generates full and focused confusion matrices.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    y_true = predictions_df["true_label"].tolist()
    y_pred = predictions_df["predicted_label"].tolist()

    # Full confusion matrix (sorted by canonical vocabulary order)
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    # 1. Full Confusion Matrix
    plt.figure(figsize=(24, 20))
    sns.heatmap(
        cm,
        xticklabels=unique_labels,
        yticklabels=unique_labels,
        cmap="Blues",
        cbar=True,
        annot=False
    )
    plt.title("Full Confusion Matrix — Test Set", fontsize=16)
    plt.xlabel("Predicted Disease", fontsize=12)
    plt.ylabel("True Disease", fontsize=12)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    full_cm_path = figures_dir / "confusion_matrix.png"
    plt.savefig(full_cm_path, dpi=200)
    plt.close()
    print(f"[OK] Saved full confusion matrix to: {full_cm_path}")

    # 2. Focused Confusion Matrix (top N most frequent / most confused classes in test set)
    true_counts = Counter(y_true)
    top_classes = [c for c, _ in true_counts.most_common(top_n_focused)]

    focused_df = predictions_df[
        predictions_df["true_label"].isin(top_classes) &
        predictions_df["predicted_label"].isin(top_classes)
    ]

    if len(focused_df) > 0:
        cm_focused = confusion_matrix(
            focused_df["true_label"],
            focused_df["predicted_label"],
            labels=top_classes
        )

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm_focused,
            xticklabels=top_classes,
            yticklabels=top_classes,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            cbar=True
        )
        plt.title(f"Focused Confusion Matrix (Top {top_n_focused} Most Frequent Test Classes)", fontsize=14)
        plt.xlabel("Predicted Disease", fontsize=11)
        plt.ylabel("True Disease", fontsize=11)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        focused_cm_path = figures_dir / "confusion_matrix_focused.png"
        plt.savefig(focused_cm_path, dpi=200)
        plt.close()
        print(f"[OK] Saved focused confusion matrix to: {focused_cm_path}")


def perform_error_analysis(
    predictions_df: pd.DataFrame,
    output_dir: Path
) -> pd.DataFrame:
    """
    PHASE 11 — Error Analysis
    Saves granular error CSV and prints summary breakdown.
    """
    errors_df = predictions_df[~predictions_df["correct"]].copy()
    error_csv = output_dir / "error_analysis.csv"

    export_cols = ["image_id", "crop", "true_label", "predicted_label", "correct", "raw_model_output"]
    errors_df[export_cols].to_csv(error_csv, index=False)
    print(f"[OK] Saved error analysis to: {error_csv} ({len(errors_df)} misclassifications)")

    # Grouped breakdowns
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS BREAKDOWN")
    print("=" * 60)

    # Top confused pairs
    confusion_pairs = Counter(zip(errors_df["true_label"], errors_df["predicted_label"]))
    print("\nTop 5 Most Common Confusions (True -> Predicted):")
    for (t, p), count in confusion_pairs.most_common(5):
        print(f"  {t} -> {p}: {count} instance(s)")

    # Errors by crop
    crop_errors = Counter(errors_df["crop"])
    print("\nMisclassifications by Crop:")
    for crop, count in crop_errors.most_common(5):
        print(f"  {crop}: {count} error(s)")

    # Diseases never correctly recognized
    all_true = set(predictions_df["true_label"])
    correctly_predicted_true = set(predictions_df[predictions_df["correct"]]["true_label"])
    never_correct = all_true - correctly_predicted_true
    print(f"\nDiseases Never Correctly Recognized in Test Set ({len(never_correct)} classes):")
    for d in sorted(list(never_correct))[:10]:
        print(f"  - {d}")
    if len(never_correct) > 10:
        print(f"  ... and {len(never_correct) - 10} more")

    # Diseases with 100% recall
    perfect_recall = []
    for d in all_true:
        sub = predictions_df[predictions_df["true_label"] == d]
        if (sub["correct"]).all():
            perfect_recall.append(d)
    print(f"\nDiseases with Perfect Recall in Test Set ({len(perfect_recall)} classes):")
    for d in sorted(perfect_recall)[:10]:
        print(f"  + {d}")

    print("=" * 60)
    return errors_df


def plot_training_curves(training_log_path: Path, figures_dir: Path):
    """
    PHASE 12 — Generates training and validation curves.
    """
    if not training_log_path.exists():
        print(f"[SKIP] Training log file not found: {training_log_path}")
        return

    figures_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(training_log_path)
    epochs = df["epoch"].tolist()

    # 1. Training & Validation Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df["train_loss"], marker="o", label="Training Loss", color="royalblue")
    if "val_loss" in df.columns:
        plt.plot(epochs, df["val_loss"], marker="s", label="Validation Loss", color="darkorange")
    plt.title("Training and Validation Loss Curve", fontsize=13)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Cross-Entropy Loss", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    loss_path = figures_dir / "training_loss.png"
    plt.savefig(loss_path, dpi=200)
    plt.close()
    print(f"[OK] Saved loss curve to: {loss_path}")

    # 2. Validation Accuracy
    if "accuracy" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, df["accuracy"], marker="^", label="Validation Accuracy", color="seagreen")
        plt.title("Validation Accuracy Curve", fontsize=13)
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Accuracy", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        acc_path = figures_dir / "validation_accuracy.png"
        plt.savefig(acc_path, dpi=200)
        plt.close()
        print(f"[OK] Saved accuracy curve to: {acc_path}")

    # 3. Validation Macro F1 (primary selection metric)
    if "macro_f1" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, df["macro_f1"], marker="D", label="Validation Macro F1", color="crimson")
        best_epoch = df.loc[df["macro_f1"].idxmax(), "epoch"]
        best_f1 = df["macro_f1"].max()
        plt.axvline(x=best_epoch, color="gray", linestyle=":", label=f"Best Checkpoint (Epoch {best_epoch})")
        plt.title(f"Validation Macro F1 Curve (Best: {best_f1:.4f} at Epoch {best_epoch})", fontsize=13)
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Macro F1", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        f1_path = figures_dir / "validation_macro_f1.png"
        plt.savefig(f1_path, dpi=200)
        plt.close()
        print(f"[OK] Saved macro F1 curve to: {f1_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate plots and error analysis for SAGE QLoRA")
    parser.add_argument("--run-id", type=str, default="001_baseline_qlora", help="Run directory identifier")
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / "outputs" / args.run_id
    figures_dir = run_dir / "figures"
    predictions_csv = run_dir / "test_predictions.csv"
    training_log_csv = run_dir / "training_log.csv"

    if not predictions_csv.exists():
        print(f"[ERROR] Predictions file not found: {predictions_csv}")
        print("Run evaluation first (python -m src.evaluate) before plotting results.")
        sys.exit(1)

    label2id, id2label = load_labels_vocab()
    pred_df = pd.read_csv(predictions_csv)

    print("\n" + "=" * 60)
    print("PLOTTING RESULTS & RUNNING ERROR ANALYSIS")
    print("=" * 60)

    plot_confusion_matrices(pred_df, figures_dir, label2id, id2label)
    perform_error_analysis(pred_df, run_dir)
    plot_training_curves(training_log_csv, figures_dir)

    print("\n[OK] All plots and error analysis artifacts generated under:")
    print(f"     {figures_dir}")


if __name__ == "__main__":
    main()
