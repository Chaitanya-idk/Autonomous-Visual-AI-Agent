"""
Full generation-based evaluation for the SAGE crop-disease model.

This script evaluates the held-out validation shard without giving the model
its ground-truth disease. It loads the saved LoRA adapter and evaluates the
entire validation shard by default.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import torch
import yaml
from tqdm.auto import tqdm
from transformers import AutoProcessor
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import SAGEDataset, sage_collate_fn
from src.model import load_trained_lora_model
from src.utils import (
    compute_classification_metrics,
    load_labels_vocab,
    match_prediction_to_vocab,
)
from src.train import download_shard


def load_config(path: str = None):
    cfg_path = Path(path) if path else PROJECT_ROOT / "configs" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate(cfg: Dict[str, Any], checkpoint: str, max_samples: int = None):
    ds_cfg = cfg["dataset"]
    common = dict(
        image_col=ds_cfg.get("image_col", "image"),
        disease_col=ds_cfg.get("disease_col", "disease"),
        crop_col=ds_cfg.get("crop_col", "crop"),
        label_id_col=ds_cfg.get("label_id_col", ""),
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Full evaluation requires a CUDA GPU in this configuration.")

    dtype_name = cfg["model"].get("torch_dtype", "float16").lower()
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    device = torch.device("cuda")

    processor = AutoProcessor.from_pretrained(
        cfg["model"]["name_or_path"],
        local_files_only=cfg["model"].get("local_files_only", False),
        use_fast=False,
    )
    prep_cfg = cfg.get("preprocessing", {})
    if prep_cfg.get("min_pixels"):
        processor.image_processor.min_pixels = prep_cfg["min_pixels"]
    if prep_cfg.get("max_pixels"):
        processor.image_processor.max_pixels = prep_cfg["max_pixels"]

    labels_path = Path(cfg["paths"].get("labels_json", "/kaggle/working/labels.json"))
    label2id, id2label = load_labels_vocab(labels_path)

    adapter_path = Path(checkpoint)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {adapter_path}")

    model = load_trained_lora_model(
        adapter_path=str(adapter_path),
        base_model_name_or_path=cfg["model"]["name_or_path"],
        torch_dtype=cfg["model"].get("torch_dtype", "float16"),
        local_files_only=cfg["model"].get("local_files_only", False),
        merge_weights=False,
    )
    model.eval()

    chunk_dir = Path(ds_cfg.get("chunk_dir", "/kaggle/working/data_cache"))
    val_dir = chunk_dir / "val"
    val_dir.mkdir(parents=True, exist_ok=True)

    val_shard = int(ds_cfg.get("total_shards", 48)) - 1
    token = __import__("os").environ.get("HF_TOKEN")
    download_shard(ds_cfg["hf_repo_id"], val_shard, val_dir, token=token)

    val_ds = SAGEDataset(
        parquet_dir=str(val_dir),
        processor=processor,
        is_training=False,
        label2id=label2id,
        **common,
    )

    if max_samples is not None:
        max_samples = min(int(max_samples), len(val_ds))
        indices = list(range(max_samples))
        from torch.utils.data import Subset
        val_ds = Subset(val_ds, indices)

    loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=sage_collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    y_true: List[int] = []
    y_pred: List[int] = []
    rows: List[Dict[str, Any]] = []
    unknown = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Full evaluation", unit="image", dynamic_ncols=True):
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            pv = batch["pixel_values"].to(device=device, dtype=dtype)
            thw = batch["image_grid_thw"].to(device)

            true_id = int(batch["label_id"][0].item())
            true_label = batch["disease"][0]

            generated = model.generate(
                input_ids=ids,
                attention_mask=attn,
                pixel_values=pv,
                image_grid_thw=thw,
                max_new_tokens=32,
                do_sample=False,
            )
            prompt_len = ids.shape[1]
            raw = processor.tokenizer.decode(
                generated[0, prompt_len:],
                skip_special_tokens=True,
            ).strip()

            pred_label, pred_id, status = match_prediction_to_vocab(raw, label2id)
            if status == "unknown":
                unknown += 1

            y_true.append(true_id)
            y_pred.append(pred_id)
            rows.append({
                "true_label": true_label,
                "true_id": true_id,
                "predicted_label": pred_label,
                "predicted_id": pred_id,
                "status": status,
                "correct": bool(true_id == pred_id),
                "raw_model_output": raw,
            })

    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        labels_list=list(label2id.values()),
    )
    metrics.update({
        "num_samples": len(y_true),
        "unknown_predictions": unknown,
        "unknown_rate": unknown / max(len(y_true), 1),
        "checkpoint": str(adapter_path),
        "validation_shard": val_shard,
    })

    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "final_evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / "predictions.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else ["true_label"])
        writer.writeheader()
        writer.writerows(rows)

    print("\n========== FINAL EVALUATION ==========")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"Metrics -> {output_dir / 'final_evaluation_metrics.json'}")
    print(f"Predictions -> {output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    evaluate(load_config(args.config), args.checkpoint, args.max_samples)
