"""
Common utilities for SAGE crop disease QLoRA pipeline.
Handles seed setting, labels.json loading, metrics calculation, and logging.
"""

import os
import json
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Common directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
LABELS_PATH = PROCESSED_DIR / "labels.json"

def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_labels_vocab(labels_path: Optional[Path] = None) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Loads the single authoritative label vocabulary from labels.json.
    Returns:
        label2id: Dict mapping disease label string to integer ID
        id2label: Dict mapping integer ID to disease label string
    """
    path = labels_path or LABELS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Authoritative labels file not found: {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    label2id = data["label2id"]
    id2label = {int(k): v for k, v in data["id2label"].items()}
    return label2id, id2label

def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    labels_list: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Computes comprehensive multi-class classification metrics:
    Accuracy, Micro/Macro/Weighted Precision, Recall, and F1.
    """
    acc = accuracy_score(y_true, y_pred)
    
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0, labels=labels_list
    )
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0, labels=labels_list
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0, labels=labels_list
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
    """Returns GPU and PyTorch device information."""
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
