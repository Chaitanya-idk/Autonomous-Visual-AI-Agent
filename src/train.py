"""
Training pipeline for SAGE crop disease LoRA fine-tuning.
Supports HuggingFace Hub streaming (no 21GB download needed).
Full precision bfloat16/float16 LoRA — no quantization.
"""

import os
import sys
import json
import time
import yaml
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import set_seed, compute_classification_metrics
from src.dataset import StreamingSAGEDataset, SAGEDataset, sage_collate_fn
from src.model import get_qwen_lora_model


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    cfg_file = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "config.yaml"
    with open(cfg_file, "r") as f:
        return yaml.safe_load(f)


# ── Prediction matching ───────────────────────────────────────────────────────

def match_prediction(raw: str, label2id: Dict[str, int]):
    cleaned = raw.strip().strip('"\'').split("\n")[0].strip()
    if cleaned in label2id:
        return cleaned, label2id[cleaned], "valid"
    lower_map = {k.lower(): k for k in label2id}
    if cleaned.lower() in lower_map:
        k = lower_map[cleaned.lower()]
        return k, label2id[k], "valid"
    return "Unknown", -1, "unknown"


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_datasets(cfg: Dict[str, Any], processor):
    ds_cfg = cfg["dataset"]
    common = dict(
        processor=processor,
        image_col=ds_cfg.get("image_col", "image"),
        disease_col=ds_cfg.get("disease_col", "disease"),
        crop_col=ds_cfg.get("crop_col", "crop"),
        label_id_col=ds_cfg.get("label_id_col", ""),
    )

    if ds_cfg.get("streaming", False):
        # ── Streaming mode: reads from HuggingFace Hub live ──────────────────
        print("[Train] Using STREAMING mode (tirtho149/SAGE via HF Hub)...")

        # First scan a small sample to build the label vocabulary
        scan_ds = StreamingSAGEDataset(
            repo_id=ds_cfg["hf_repo_id"], split="train",
            max_samples=5000, is_training=False, **common
        )
        label2id = scan_ds._build_label2id_from_streaming(scan_samples=5000)

        train_ds = StreamingSAGEDataset(
            repo_id=ds_cfg["hf_repo_id"], split="train",
            max_samples=ds_cfg.get("train_size", 50000),
            is_training=True, label2id=label2id,
            seed=ds_cfg.get("seed", 42), **common
        )
        val_ds = StreamingSAGEDataset(
            repo_id=ds_cfg["hf_repo_id"], split="train",  # use a held-out portion
            max_samples=ds_cfg.get("val_size", 5000),
            is_training=True, label2id=label2id,
            seed=ds_cfg.get("seed", 42) + 1, **common  # different seed → different shuffle
        )
        return train_ds, val_ds, label2id

    else:
        # ── Local parquet mode ────────────────────────────────────────────────
        parquet_dir = ds_cfg.get("parquet_dir", "data/parquet")
        if not Path(parquet_dir).is_absolute():
            parquet_dir = str(PROJECT_ROOT / parquet_dir)
        print(f"[Train] Loading local parquet from {parquet_dir}...")

        from torch.utils.data import random_split
        full_ds = SAGEDataset(parquet_dir=parquet_dir, is_training=True, **common)
        label2id = full_ds.label2id
        n = len(full_ds)
        val_n   = int(n * ds_cfg.get("val_split", 0.15))
        train_n = n - val_n
        train_ds, val_ds = random_split(
            full_ds, [train_n, val_n],
            generator=torch.Generator().manual_seed(ds_cfg.get("seed", 42))
        )
        return train_ds, val_ds, label2id


# ── Validation ────────────────────────────────────────────────────────────────

def validate(model, processor, val_loader, label2id, device, dtype, max_batches=200):
    """Run validation — capped at max_batches for streaming mode."""
    model.eval()
    total_loss, count = 0.0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            ids  = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            pv   = batch["pixel_values"].to(device, dtype=dtype)
            thw  = batch["image_grid_thw"].to(device)
            lbl  = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                out = model(input_ids=ids, attention_mask=attn,
                            pixel_values=pv, image_grid_thw=thw, labels=lbl)
            total_loss += out.loss.item()
            count += 1

            # Generate prediction
            gen_ids = model.generate(
                input_ids=ids, attention_mask=attn,
                pixel_values=pv, image_grid_thw=thw,
                max_new_tokens=32, do_sample=False
            )
            raw = processor.tokenizer.decode(
                gen_ids[0, ids.shape[1]:], skip_special_tokens=True).strip()
            _, pred_id, _ = match_prediction(raw, label2id)

            y_true.extend(batch["label_id"].tolist())
            y_pred.append(pred_id)

    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["val_loss"] = total_loss / max(1, count)
    return metrics


# ── Training loop ─────────────────────────────────────────────────────────────

def train(cfg: Dict[str, Any]):
    train_cfg = cfg["training"]
    set_seed(train_cfg.get("seed", 42))

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = train_cfg.get("bf16", True)
    dtype    = torch.bfloat16 if use_bf16 else torch.float16

    # Output directories
    ckpt_dir   = Path(cfg["paths"]["checkpoint_dir"])
    output_dir = Path(cfg["paths"]["output_dir"])
    log_dir    = Path(cfg["paths"]["log_dir"])
    for d in [ckpt_dir, output_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Processor — use_fast=False avoids the Qwen2VL fast-processor breaking-change warning
    prep_cfg  = cfg.get("preprocessing", {})
    processor = AutoProcessor.from_pretrained(
        cfg["model"]["name_or_path"],
        local_files_only=cfg["model"].get("local_files_only", False),
        use_fast=False,
    )
    if prep_cfg.get("min_pixels"):
        processor.image_processor.min_pixels = prep_cfg["min_pixels"]
    if prep_cfg.get("max_pixels"):
        processor.image_processor.max_pixels = prep_cfg["max_pixels"]

    # Datasets
    train_ds, val_ds, label2id = build_datasets(cfg, processor)

    # Save label vocab
    labels_path = Path(cfg["paths"].get("labels_json", "data/labels.json"))
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_path, "w") as f:
        json.dump({"label2id": label2id,
                   "id2label": {v: k for k, v in label2id.items()}}, f, indent=2)
    print(f"[Train] {len(label2id)} disease classes. Labels saved → {labels_path}")

    batch_size = train_cfg.get("batch_size", 2)
    # Note: streaming IterableDataset needs shuffle=False (shuffled inside HF)
    is_streaming = cfg["dataset"].get("streaming", False)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=(not is_streaming),
                              collate_fn=sage_collate_fn,
                              num_workers=0,        # 0 required for IterableDataset
                              pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              collate_fn=sage_collate_fn, num_workers=0)

    # Model
    lora_cfg = cfg["lora"]
    model = get_qwen_lora_model(
        model_name_or_path=cfg["model"]["name_or_path"],
        lora_r=lora_cfg.get("r", 32),
        lora_alpha=lora_cfg.get("lora_alpha", 64),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        torch_dtype=cfg["model"].get("torch_dtype", "bfloat16"),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        local_files_only=cfg["model"].get("local_files_only", False),
        is_trainable=True,
    )

    # Optimizer + scheduler
    grad_accum  = train_cfg.get("gradient_accumulation_steps", 16)
    num_epochs  = train_cfg.get("num_epochs", 10)
    patience    = train_cfg.get("early_stopping_patience", 7)
    lr          = float(train_cfg.get("learning_rate", 1e-4))

    # Estimate steps (for streaming: use train_size / batch_size)
    if is_streaming:
        steps_per_epoch = ds_cfg_steps = cfg["dataset"].get("train_size", 50000) // batch_size
    else:
        steps_per_epoch = len(train_loader)
    total_steps  = (steps_per_epoch // grad_accum) * num_epochs
    warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.05))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=float(train_cfg.get("weight_decay", 0.01))
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=max(1, total_steps)
    )

    print(f"\n{'='*60}")
    print(f"STARTING TRAINING | epochs={num_epochs} | patience={patience} | batch={batch_size}")
    print(f"Mode: {'STREAMING' if is_streaming else 'LOCAL PARQUET'}")
    print(f"{'='*60}\n")

    best_f1    = -1.0
    best_epoch = -1
    no_improve = 0
    logs       = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        step       = 0
        t0         = time.time()
        optimizer.zero_grad()

        for batch in train_loader:
            ids  = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            pv   = batch["pixel_values"].to(device, dtype=dtype)
            thw  = batch["image_grid_thw"].to(device)
            lbl  = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                out  = model(input_ids=ids, attention_mask=attn,
                             pixel_values=pv, image_grid_thw=thw, labels=lbl)
                loss = out.loss / grad_accum

            loss.backward()
            step       += 1
            epoch_loss += loss.item() * grad_accum

            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    train_cfg.get("max_grad_norm", 1.0)
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                opt_step = step // grad_accum
                if opt_step % 20 == 0:
                    print(f"  Ep {epoch} | step {step} | "
                          f"loss {epoch_loss/step:.4f} | lr {scheduler.get_last_lr()[0]:.2e}",
                          flush=True)

        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(step, 1)
        print(f"\nEpoch {epoch} done in {elapsed:.0f}s | avg_loss={avg_loss:.4f}")

        # Validation (cap at 200 batches for streaming)
        val_metrics = validate(model, processor, val_loader, label2id,
                               device, dtype, max_batches=200)
        macro_f1 = val_metrics["macro_f1"]
        print(f"[VAL] loss={val_metrics['val_loss']:.4f} | "
              f"acc={val_metrics['accuracy']:.4f} | macro_f1={macro_f1:.4f}")

        logs.append({"epoch": epoch, "train_loss": avg_loss, "time_sec": elapsed, **val_metrics})
        pd.DataFrame(logs).to_csv(log_dir / "training_log.csv", index=False)

        # Checkpoint
        if macro_f1 > best_f1:
            best_f1    = macro_f1
            best_epoch = epoch
            no_improve = 0
            best_ckpt  = str(ckpt_dir / "best_checkpoint")
            model.save_pretrained(best_ckpt)
            print(f"  [SAVED] Best → {best_ckpt}", flush=True)
        else:
            no_improve += 1
            print(f"  [EarlyStopping] {no_improve}/{patience} epochs without improvement.")
            if no_improve >= patience:
                print(f"  Stopping early at epoch {epoch}.")
                break

        model.save_pretrained(str(ckpt_dir / f"epoch_{epoch}"))

    # Summary
    summary = {"best_epoch": best_epoch, "best_val_macro_f1": best_f1,
               "checkpoint": str(ckpt_dir / "best_checkpoint")}
    with open(output_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}\nDONE | Best Epoch: {best_epoch} | Best Macro F1: {best_f1:.4f}\n{'='*60}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    train(load_config(args.config))
