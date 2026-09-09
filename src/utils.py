"""Utilities for SAGE crop-disease LoRA training and evaluation."""

import json
import random
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
LABELS_PATH = PROCESSED_DIR / "labels.json"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def save_labels_vocab(label2id: Dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    id2label = {int(v): k for k, v in label2id.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"label2id": label2id, "id2label": id2label},
            f,
            indent=2,
            ensure_ascii=False,
        )


def load_labels_vocab(
    labels_path: Optional[Path] = None,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    path = labels_path or LABELS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Authoritative labels file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    label2id = {str(k): int(v) for k, v in data["label2id"].items()}
    id2label = {int(k): str(v) for k, v in data["id2label"].items()}
    return label2id, id2label


def normalize_label_text(text: str) -> str:
    """Normalize harmless formatting differences without fuzzy matching."""
    text = str(text or "").strip()
    text = text.replace("\r", "\n")
    text = text.split("\n", 1)[0].strip()
    text = text.strip(" \t\"'`.,:;!?()[]{}")
    text = " ".join(text.split())
    text = text.replace("-", "_")
    text = text.replace(" ", "_")
    return text.lower()


def match_prediction_to_vocab(
    raw: str,
    label2id: Dict[str, int],
) -> Tuple[str, int, str]:
    """
    Match model output to the authoritative vocabulary.

    Accepted differences:
      - case
      - spaces vs underscores
      - hyphens vs underscores
      - harmless leading/trailing punctuation

    We intentionally do NOT use fuzzy matching because that could turn an
    incorrect disease into an artificially correct metric result.
    """
    raw = str(raw or "").strip()
    if not raw:
        return "Unknown", -1, "unknown"

    exact = raw.strip().strip("\"'").strip()
    if exact in label2id:
        return exact, int(label2id[exact]), "valid"

    lower_map = {str(label).lower(): label for label in label2id}
    if exact.lower() in lower_map:
        canonical = lower_map[exact.lower()]
        return canonical, int(label2id[canonical]), "valid"

    normalized_map = {
        normalize_label_text(label): label
        for label in label2id
    }
    normalized = normalize_label_text(raw)
    if normalized in normalized_map:
        canonical = normalized_map[normalized]
        return canonical, int(label2id[canonical]), "valid_normalized"

    # Some models answer with a short sentence. Accept only an exact
    # vocabulary phrase occurring as a standalone normalized answer.
    normalized_raw = normalize_label_text(raw)
    for normalized_label, canonical in normalized_map.items():
        if normalized_raw == normalized_label:
            return canonical, int(label2id[canonical]), "valid_normalized"

    return "Unknown", -1, "unknown"


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    labels_list: Optional[List[int]] = None,
) -> Dict[str, float]:
    if not y_true:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "weighted_precision": 0.0,
            "weighted_recall": 0.0,
            "weighted_f1": 0.0,
        }

    acc = accuracy_score(y_true, y_pred)

    # For classification metrics we explicitly evaluate the real disease
    # vocabulary. -1 (Unknown) remains a wrong prediction but is not treated
    # as a legitimate disease class.
    if labels_list is None:
        labels_list = sorted(set(y_true))

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
        labels=labels_list,
    )
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
        labels=labels_list,
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
        labels=labels_list,
    )

    return {
        "accuracy": float(acc),
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1": float(f1_macro),
        "micro_precision": float(prec_micro),
        "micro_recall": float(rec_micro),
        "micro_f1": float(f1_micro),
        "weighted_precision": float(prec_weighted),
        "weighted_recall": float(rec_weighted),
        "weighted_f1": float(f1_weighted),
    }


def get_device_info() -> Dict[str, str]:
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        info["vram_gb"] = round(vram_bytes / (1024 ** 3), 2)
    return info
