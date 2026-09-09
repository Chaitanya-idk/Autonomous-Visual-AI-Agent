"""Utilities for reproducibility, label vocabularies, and evaluation metrics."""

import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
LABELS_PATH = PROCESSED_DIR / "labels.json"


def set_seed(seed: int = 42):
    """Set Python/NumPy/PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Reproducibility is preferable for this experiment. cuDNN may choose
        # slower kernels, but the run is much easier to reproduce.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_labels_vocab(
    labels_path: Optional[Path] = None,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Load the authoritative label2id/id2label JSON file."""
    path = Path(labels_path or LABELS_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Authoritative labels file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    label2id = {str(k): int(v) for k, v in data["label2id"].items()}
    id2label = {int(k): str(v) for k, v in data["id2label"].items()}

    if len(label2id) != len(id2label):
        raise ValueError(
            f"labels.json is inconsistent: {len(label2id)} label2id entries "
            f"vs {len(id2label)} id2label entries."
        )

    return label2id, id2label


def save_labels_vocab(
    labels_path: Path,
    label2id: Dict[str, int],
):
    """Write one deterministic authoritative vocabulary."""
    labels_path = Path(labels_path)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    normalized = {str(k): int(v) for k, v in label2id.items()}
    id2label = {str(v): k for k, v in normalized.items()}

    with open(labels_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "label2id": normalized,
                "id2label": id2label,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )


def normalize_label(text: str) -> str:
    """Normalize formatting without changing the semantic label."""
    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", str(text)).strip()

    # Remove common Markdown/code formatting and surrounding quotes.
    text = text.strip("` \t\r\n\"'")

    # Normalize separators so spaces/underscores/hyphens are equivalent.
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    # Remove punctuation at the edges and lowercase for comparison.
    text = text.strip(" .,:;!?()[]{}")
    return text.casefold()


def match_prediction(
    raw: str,
    label2id: Dict[str, int],
):
    """
    Map a generated answer to an official class.

    Matching is deliberately deterministic and conservative:
      1. exact normalized match;
      2. if the model wrapped the answer in a sentence, find an official
         normalized label as a complete phrase;
      3. otherwise return Unknown.

    We do NOT use fuzzy edit distance because that can turn genuinely wrong
    generations into artificially correct metrics.
    """
    if raw is None:
        return "Unknown", -1, "unknown"

    raw = str(raw).replace("\r", "\n")

    # The model should answer with one label. Prefer the first non-empty line.
    lines = [line.strip() for line in raw.split("\n") if line.strip()]
    candidate = lines[0] if lines else ""

    normalized_to_label = {
        normalize_label(label): label
        for label in label2id
    }

    norm_candidate = normalize_label(candidate)
    if norm_candidate in normalized_to_label:
        label = normalized_to_label[norm_candidate]
        return label, label2id[label], "valid_exact"

    # Handle harmless answer wrappers such as:
    # "The disease is Tomato Early Blight."
    # Choose the longest matching label to avoid shorter labels winning.
    normalized_raw = normalize_label(raw)
    matches = []
    for norm_label, label in normalized_to_label.items():
        if not norm_label:
            continue
        if re.search(
            rf"(?<!\w){re.escape(norm_label)}(?!\w)",
            normalized_raw,
        ):
            matches.append((len(norm_label), label))

    if matches:
        _, label = max(matches, key=lambda item: item[0])
        return label, label2id[label], "valid_embedded"

    return "Unknown", -1, "unknown"


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    labels_list: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Unknown predictions use -1. Accuracy counts them as incorrect. For macro,
    micro, and weighted class metrics, only official labels in labels_list are
    evaluated, so an Unknown prediction does not become a fake extra class.
    """
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
            "unknown_predictions": 0,
            "unknown_rate": 0.0,
        }

    if labels_list is None:
        labels_list = sorted(set(y_true))

    acc = accuracy_score(y_true, y_pred)

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

    unknown_count = sum(pred < 0 for pred in y_pred)

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
        "unknown_predictions": int(unknown_count),
        "unknown_rate": float(unknown_count / len(y_pred)),
    }


def compute_per_class_metrics(
    y_true: List[int],
    y_pred: List[int],
    id2label: Dict[int, str],
) -> Dict[str, Dict[str, Any]]:
    """Return precision/recall/F1/support for every official disease class."""
    labels = sorted(id2label.keys())
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    report = {}
    for idx, label_id in enumerate(labels):
        report[id2label[label_id]] = {
            "label_id": int(label_id),
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
    return report


def get_device_info() -> Dict[str, Any]:
    """Return basic GPU/PyTorch information."""
    info: Dict[str, Any] = {
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
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        info["vram_gb"] = round(vram_bytes / (1024 ** 3), 2)
    return info
