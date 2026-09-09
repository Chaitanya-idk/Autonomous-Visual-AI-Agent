"""
Final, full validation evaluation for a trained SAGE LoRA adapter.

This script evaluates the entire fixed validation shard (not the small
per-epoch validation cap) and writes:
  outputs/final_evaluation_metrics.json
  outputs/per_class_metrics.json
  outputs/predictions.csv

Usage:
    python -m src.evaluate
    python -m src.evaluate --adapter /kaggle/working/checkpoints/best_checkpoint
"""

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import SAGEDataset, sage_collate_fn
from src.model import load_trained_lora_model
from src.utils import (
    load_labels_vocab,
    match_prediction,
    compute_classification_metrics,
    compute_per_class_metrics,
)
from src.train import load_config


def make_collate_fn(processor):
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0
    return partial(sage_collate_fn, pad_token_id=int(pad_token_id))


def main(cfg: Dict[str, Any], adapter_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = cfg["model"]
    dtype_name = str(model_cfg.get("torch_dtype", "float16")).lower()
    dtype = torch.bfloat16 if dtype_name in {"bfloat16", "bf16"} else torch.float16
    if device.type == "cpu":
        dtype = torch.float32

    processor = AutoProcessor.from_pretrained(
        model_cfg["name_or_path"],
        local_files_only=bool(model_cfg.get("local_files_only", False)),
        use_fast=False,
    )

    prep_cfg = cfg.get("preprocessing", {})
    if prep_cfg.get("min_pixels"):
        processor.image_processor.min_pixels = prep_cfg["min_pixels"]
    if prep_cfg.get("max_pixels"):
        processor.image_processor.max_pixels = prep_cfg["max_pixels"]

    labels_path = Path(cfg["paths"].get("labels_json", "data/labels.json"))
    label2id, id2label = load_labels_vocab(labels_path)

    adapter = Path(adapter_path)
    if not adapter.exists():
        raise FileNotFoundError(f"Adapter checkpoint not found: {adapter}")

    model = load_trained_lora_model(
        adapter_path=str(adapter),
        base_model_name_or_path=model_cfg["name_or_path"],
        torch_dtype=model_cfg.get("torch_dtype", "float16"),
        local_files_only=bool(model_cfg.get("local_files_only", False)),
        merge_weights=False,
        is_trainable=False,
    )
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    ds_cfg = cfg["dataset"]
    chunk_dir = Path(ds_cfg.get("chunk_dir", "/kaggle/working/data_cache"))
    total_shards = int(ds_cfg.get("total_shards", 48))
    val_dir = chunk_dir / "val"
    val_shard_idx = total_shards - 1
    val_file = val_dir / f"train-{val_shard_idx:05d}.parquet"

    # The training script normally leaves the fixed validation shard in place.
    if not val_file.exists():
        from src.train import download_shard
        token = __import__("os").environ.get("HF_TOKEN")
        download_shard(
            ds_cfg.get("hf_repo_id", "tirtho149/SAGE"),
            val_shard_idx,
            val_dir,
            token=token,
        )

    common = dict(
        processor=processor,
        image_col=ds_cfg.get("image_col", "image"),
        disease_col=ds_cfg.get("disease_col", "disease"),
        crop_col=ds_cfg.get("crop_col", "crop"),
        label_id_col=ds_cfg.get("label_id_col", ""),
        label2id=label2id,
    )

    # Full validation is generation-only here. No ground-truth answer enters
    # the model input.
    val_ds = SAGEDataset(
        parquet_dir=str(val_dir),
        is_training=False,
        **common,
    )
    loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=make_collate_fn(processor),
        num_workers=0,
        pin_memory=True,
    )

    y_true: List[int] = []
    y_pred: List[int] = []
    rows: List[Dict[str, Any]] = []

    print(
        f"[Final Evaluation] shard={val_shard_idx:02d} | "
        f"samples={len(val_ds)} | classes={len(label2id)}"
    )

    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), desc="Final evaluation", unit="image"):
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device, dtype=dtype)
            image_grid_thw = batch["image_grid_thw"].to(device)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=attn,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=32,
                do_sample=False,
                num_beams=1,
            )

            new_tokens = generated_ids[0, ids.shape[1]:]
            raw_text = processor.tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            ).strip()

            predicted_label, pred_id, status = match_prediction(
                raw_text,
                label2id,
            )
            true_id = int(batch["label_id"][0].item())
            true_label = batch["disease"][0]

            y_true.append(true_id)
            y_pred.append(pred_id)
            rows.append(
                {
                    "true_label": true_label,
                    "true_id": true_id,
                    "predicted_label": predicted_label,
                    "predicted_id": pred_id,
                    "status": status,
                    "correct": bool(true_id == pred_id),
                    "raw_model_output": raw_text,
                }
            )

    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        labels_list=sorted(label2id.values()),
    )
    per_class = compute_per_class_metrics(y_true, y_pred, id2label)

    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "final_evaluation_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    with open(output_dir / "per_class_metrics.json", "w", encoding="utf-8") as file:
        json.dump(per_class, file, indent=2, ensure_ascii=False)

    pd.DataFrame(rows).to_csv(
        output_dir / "predictions.csv",
        index=False,
    )

    print("\n===== FINAL EVALUATION =====")
    for key in (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "unknown_predictions",
        "unknown_rate",
    ):
        print(f"{key}: {metrics[key]}")

    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--adapter", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    default_adapter = (
        Path(cfg["paths"]["checkpoint_dir"]) / "best_checkpoint"
    )
    main(cfg, args.adapter or str(default_adapter))
