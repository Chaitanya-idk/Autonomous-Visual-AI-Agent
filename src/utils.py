"""
Utilities for SAGE crop-disease LoRA training and evaluation.

The evaluation path is deliberately strict:
- ground-truth labels can never silently become -1
- label IDs remain stable once assigned
- model outputs are normalized deterministically
- unknown predictions are counted explicitly
- an empty evaluation set is an error, not a fake 0.0 result
"""

import json
import random
import re
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


# =============================================================================
# LABEL VOCABULARY
# =============================================================================

def save_labels_vocab(
    label2id: Dict[str, int],
    path: Path,
) -> None:
    """
    Save label2id WITHOUT renumbering existing classes.

    This is critical for resumable training: if new classes are discovered
    later, existing IDs must remain exactly the same.
    """
    if not label2id:
        raise ValueError("Cannot save an empty label vocabulary.")

    cleaned = {}
    for label, idx in label2id.items():
        label = str(label).strip()
        idx = int(idx)

        if not label:
            raise ValueError("Label vocabulary contains an empty label.")
        if idx < 0:
            raise ValueError(f"Negative class ID for {label!r}: {idx}")

        if label in cleaned and cleaned[label] != idx:
            raise ValueError(
                f"Conflicting IDs for label {label!r}: "
                f"{cleaned[label]} vs {idx}"
            )

        cleaned[label] = idx

    ids = list(cleaned.values())
    if len(ids) != len(set(ids)):
        raise ValueError(
            "Two different disease labels share the same class ID."
        )

    expected = set(range(len(cleaned)))
    actual = set(ids)
    if actual != expected:
        raise ValueError(
            "Label IDs must be contiguous from 0..N-1. "
            f"Expected {sorted(expected)}, got {sorted(actual)}"
        )

    id2label = {
        str(idx): label
        for label, idx in sorted(
            cleaned.items(),
            key=lambda x: x[1],
        )
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label2id": cleaned,
                "id2label": id2label,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


def load_labels_vocab(
    labels_path: Optional[Path] = None,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    path = Path(labels_path or LABELS_PATH)

    if not path.exists():
        raise FileNotFoundError(
            f"Authoritative labels file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "label2id" not in data:
        raise ValueError(
            f"Invalid labels file: missing 'label2id': {path}"
        )

    label2id = {
        str(k): int(v)
        for k, v in data["label2id"].items()
    }

    if not label2id:
        raise ValueError(
            f"Labels file contains an empty vocabulary: {path}"
        )

    ids = list(label2id.values())

    if any(idx < 0 for idx in ids):
        raise ValueError("labels.json contains a negative class ID.")

    if len(ids) != len(set(ids)):
        raise ValueError(
            "labels.json contains duplicate class IDs."
        )

    expected = set(range(len(label2id)))
    actual = set(ids)

    if actual != expected:
        raise ValueError(
            "Invalid label vocabulary. IDs must be contiguous "
            f"0..{len(label2id)-1}, got {sorted(actual)}"
        )

    id2label = {
        idx: label
        for label, idx in label2id.items()
    }

    return label2id, id2label


def extend_labels_vocab(
    label2id: Dict[str, int],
    labels: List[str],
) -> bool:
    """
    Add previously unseen labels while preserving every existing ID.

    Returns True when the vocabulary changed.
    """
    changed = False

    next_id = (
        max(label2id.values()) + 1
        if label2id
        else 0
    )

    for raw_label in sorted(set(labels)):
        label = str(raw_label).strip()

        if not label:
            continue

        if label not in label2id:
            label2id[label] = next_id
            next_id += 1
            changed = True

    return changed


# =============================================================================
# LABEL NORMALIZATION / PREDICTION MATCHING
# =============================================================================

def normalize_label_text(text: str) -> str:
    """
    Normalize harmless formatting differences.

    This is deterministic matching, not fuzzy matching.
    """
    text = str(text or "").strip()

    if not text:
        return ""

    # Remove special whitespace.
    text = text.replace("\r", "\n")

    # Keep the first non-empty line. Qwen should answer with one label.
    lines = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]
    if not lines:
        return ""

    text = lines[0]

    # Remove common wrappers.
    text = text.strip(
        " \t\"'`.,:;!?()[]{}<>"
    )

    # Remove common conversational prefixes.
    prefixes = [
        "the disease is",
        "the disease appears to be",
        "the disease appears as",
        "the disease is:",
        "disease:",
        "answer:",
        "prediction:",
        "predicted disease:",
        "the answer is",
    ]

    lowered = text.lower()

    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if lowered.startswith(prefix):
                text = text[len(prefix):].strip()
                lowered = text.lower()
                changed = True
                break

    # Normalize separators.
    text = text.replace("-", "_")
    text = text.replace("/", "_")
    text = text.replace("\\", "_")
    text = " ".join(text.split())
    text = text.replace(" ", "_")

    # Remove punctuation except underscore.
    text = re.sub(r"[^\w_]", "", text)

    # Collapse repeated underscores.
    text = re.sub(r"_+", "_", text)

    return text.strip("_").lower()


def build_normalized_vocab(
    label2id: Dict[str, int],
) -> Dict[str, str]:
    """
    Return normalized_label -> canonical label.
    """
    normalized_vocab: Dict[str, str] = {}

    for label in label2id:
        normalized = normalize_label_text(label)

        if not normalized:
            raise ValueError(
                f"Label {label!r} becomes empty after normalization."
            )

        if normalized in normalized_vocab:
            previous = normalized_vocab[normalized]
            if previous != label:
                raise ValueError(
                    "Two labels normalize to the same value: "
                    f"{previous!r} and {label!r} -> {normalized!r}"
                )

        normalized_vocab[normalized] = label

    return normalized_vocab


def match_prediction_to_vocab(
    raw: str,
    label2id: Dict[str, int],
) -> Tuple[str, int, str]:
    """
    Match a generated answer to the official vocabulary.

    Returns:
        canonical_label, class_id, status

    status:
        valid
        valid_normalized
        unknown
    """
    raw = str(raw or "").strip()

    if not raw:
        return "Unknown", -1, "unknown"

    normalized_vocab = build_normalized_vocab(label2id)
    normalized_prediction = normalize_label_text(raw)

    if not normalized_prediction:
        return "Unknown", -1, "unknown"

    # Exact normalized match.
    if normalized_prediction in normalized_vocab:
        canonical = normalized_vocab[normalized_prediction]
        return (
            canonical,
            int(label2id[canonical]),
            "valid_normalized",
        )

    # Sometimes generation includes a small wrapper around the label.
    # Only accept a deterministic complete-label containment.
    for normalized_label, canonical in normalized_vocab.items():
        if (
            normalized_prediction == normalized_label
            or normalized_prediction.startswith(
                normalized_label + "_"
            )
            or normalized_prediction.endswith(
                "_" + normalized_label
            )
        ):
            return (
                canonical,
                int(label2id[canonical]),
                "valid_normalized",
            )

    return "Unknown", -1, "unknown"


# =============================================================================
# METRICS
# =============================================================================

def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    labels_list: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Empty evaluation data is an error.
    Ground-truth IDs may never be negative.
    Prediction ID -1 is retained as an explicit incorrect/unknown prediction.
    """
    if not y_true:
        raise RuntimeError(
            "No evaluation samples were supplied. "
            "Refusing to report fake zero metrics."
        )

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, "
            f"y_pred={len(y_pred)}"
        )

    if any(int(x) < 0 for x in y_true):
        raise ValueError(
            "Ground-truth labels contain a negative class ID. "
            "Ground-truth labels must never be -1."
        )

    if labels_list is None:
        labels_list = sorted(set(int(x) for x in y_true))

    labels_list = sorted(
        set(
            int(x)
            for x in labels_list
            if int(x) >= 0
        )
    )

    if not labels_list:
        raise RuntimeError(
            "No valid class IDs were supplied."
        )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
            labels=labels_list,
        )
    )

    precision_micro, recall_micro, f1_micro, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
            labels=labels_list,
        )
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
            labels=labels_list,
        )
    )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "micro_precision": float(precision_micro),
        "micro_recall": float(recall_micro),
        "micro_f1": float(f1_micro),
        "weighted_precision": float(precision_weighted),
        "weighted_recall": float(recall_weighted),
        "weighted_f1": float(f1_weighted),
    }


def get_device_info() -> Dict[str, str]:
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        ),
        "torch_version": torch.__version__,
    }

    if torch.cuda.is_available():
        vram_bytes = (
            torch.cuda
            .get_device_properties(0)
            .total_memory
        )
        info["vram_gb"] = round(
            vram_bytes / (1024 ** 3),
            2,
        )

    return info
