"""
Training pipeline for SAGE crop disease LoRA fine-tuning.
Loads from parquet files or HuggingFace Hub. Full precision (no quantization).
Supports early stopping, batch_size > 1, and bfloat16 autocast.
"""

import os
import sys
import gc
import json
import time
import yaml
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader, random_split
from transformers import get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import set_seed, compute_classification_metrics
from src.dataset import SAGEDataset, sage_collate_fn
from src.model import get_qwen_lora_model, load_trained_lora_model


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    cfg_file = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "config.yaml"
    with open(cfg_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Prediction matching ───────────────────────────────────────────────────────

def match_prediction(raw: str, label2id: Dict[str, int]):
    cleaned = raw.strip().strip('"\'').split("\n")[0].strip()
    if cleaned in label2id:
        return cleaned, label2id[cleaned], "valid"
    lower = {k.lower(): k for k in label2id}
    if cleaned.lower() in lower:
        k = lower[cleaned.lower()]
        return k, label2id[k], "valid"
    return "Unknown", -1, "unknown"


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_datasets(cfg: Dict[str, Any], processor):
    ds_cfg = cfg["dataset"]
    prep_cfg = cfg.get("preprocessing", {})

    common_kwargs = dict(
        processor=processor,
        image_col=ds_cfg.get("image_col", "image"),
        disease_col=ds_cfg.get("disease_col", "disease"),
        crop_col=ds_cfg.get("crop_col", "crop"),
        label_id_col=ds_cfg.get("label_id_col", "label_id"),
    )

    if ds_cfg.get("use_hf_hub", False):
        from datasets import load_dataset
        print(f"[Dataset] Loading from HuggingFace Hub: {ds_cfg['hf_repo_id']}")
        hf = load_dataset(ds_cfg["hf_repo_id"])
        train_hf = hf.get("train", hf[list(hf.keys())[0]])
        val_hf   = hf.get("validation", None)
        test_hf  = hf.get("test", None)

        full_ds = SAGEDataset(hf_dataset=train_hf, is_training=True, **common_kwargs)
        label2id = full_ds.label2id

        if val_hf:
            val_ds  = SAGEDataset(hf_dataset=val_hf,  is_training=True,  label2id=label2id, **common_kwargs)
            test_ds = SAGEDataset(hf_dataset=test_hf, is_training=False, label2id=label2id, **common_kwargs) if test_hf else None
            train_ds = full_ds
        else:
            # Split locally
            n = len(full_ds)
            val_n  = int(n * ds_cfg.get("val_split", 0.15))
            test_n = int(n * ds_cfg.get("test_split", 0.10))
            train_n = n - val_n - test_n
            train_ds, val_ds, test_ds = random_split(
                full_ds, [train_n, val_n, test_n],
                generator=torch.Generator().manual_seed(ds_cfg.get("seed", 42))
            )
    else:
        parquet_dir = ds_cfg.get("parquet_dir", "data/parquet")
        if not Path(parquet_dir).is_absolute():
            parquet_dir = str(PROJECT_ROOT / parquet_dir)

        full_ds = SAGEDataset(parquet_dir=parquet_dir, is_training=True, **common_kwargs)
        label2id = full_ds.label2id

        n = len(full_ds)
        val_n  = int(n * ds_cfg.get("val_split", 0.15))
        test_n = int(n * ds_cfg.get("test_split", 0.10))
        train_n = n - val_n - test_n
        train_ds, val_ds_raw, test_ds_raw = random_split(
            full_ds, [train_n, val_n, test_n],
            generator=torch.Generator().manual_seed(ds_cfg.get("seed", 42))
        )
        # Wrap subsets for eval (is_training=False for gen validation)
        val_ds  = val_ds_raw
        test_ds = test_ds_raw

    return train_ds, val_ds, test_ds, label2id


# ── Validation ────────────────────────────────────────────────────────────────

def validate(model, processor, val_loader, label2id, device, dtype):
    model.eval()
    total_loss, count = 0.0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in val_loader:
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

            # Generative prediction for F1
            gen_ids = model.generate(
                input_ids=ids, attention_mask=attn,
                pixel_values=pv, image_grid_thw=thw,
                max_new_tokens=32, do_sample=False
            )
            prompt_len = ids.shape[1]
            raw = processor.tokenizer.decode(
                gen_ids[0, prompt_len:], skip_special_tokens=True
            ).strip()
            _, pred_id, _ = match_prediction(raw, label2id)
            true_ids = batch["label_id"].tolist()
            y_true.extend(true_ids)
            y_pred.append(pred_id)

    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["val_loss"] = total_loss / max(1, count)
    return metrics


# ── Main training ─────────────────────────────────────────────────────────────

def train(cfg: Dict[str, Any]):
    train_cfg = cfg["training"]
    set_seed(train_cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = train_cfg.get("bf16", True)
    dtype    = torch.bfloat16 if use_bf16 else torch.float16

    # Directories
    run_id = "001_sage_lora"
    ckpt_dir   = PROJECT_ROOT / cfg["paths"]["checkpoint_dir"] / run_id
    output_dir = PROJECT_ROOT / cfg["paths"]["output_dir"]    / run_id
    log_dir    = PROJECT_ROOT / cfg["paths"]["log_dir"]       / run_id
    for d in [ckpt_dir, output_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Processor
    from transformers import AutoProcessor
    prep_cfg = cfg.get("preprocessing", {})
    processor = AutoProcessor.from_pretrained(
        cfg["model"]["name_or_path"],
        local_files_only=cfg["model"].get("local_files_only", False),
    )
    min_px = prep_cfg.get("min_pixels")
    max_px = prep_cfg.get("max_pixels")
    if min_px:
        processor.image_processor.min_pixels = min_px
    if max_px:
        processor.image_processor.max_pixels = max_px

    # Data
    train_ds, val_ds, test_ds, label2id = build_datasets(cfg, processor)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} samples")

    # Save label vocab
    labels_path = PROJECT_ROOT / cfg["paths"].get("labels_json", "data/labels.json")
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_path, "w") as f:
        json.dump({"label2id": label2id, "id2label": {v: k for k, v in label2id.items()}}, f, indent=2)

    batch_size = train_cfg.get("batch_size", 4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=sage_collate_fn, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              collate_fn=sage_collate_fn, num_workers=1)

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

    # Optimizer & scheduler
    grad_accum   = train_cfg.get("gradient_accumulation_steps", 8)
    num_epochs   = train_cfg.get("num_epochs", 10)
    patience     = train_cfg.get("early_stopping_patience", 7)
    lr           = float(train_cfg.get("learning_rate", 1e-4))
    total_steps  = (len(train_loader) // grad_accum) * num_epochs
    warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.05))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=float(train_cfg.get("weight_decay", 0.01))
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=max(1, total_steps)
    )

    print(f"\n{'='*60}\nSTARTING TRAINING — {num_epochs} epochs | patience={patience} | batch={batch_size}\n{'='*60}")

    best_f1     = -1.0
    best_epoch  = -1
    no_improve  = 0
    logs        = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
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
            epoch_loss += loss.item() * grad_accum

            if step % grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    train_cfg.get("max_grad_norm", 1.0)
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                opt_step = step // grad_accum
                if opt_step % 20 == 0:
                    print(f"  Epoch {epoch} | Step {step}/{len(train_loader)} | "
                          f"Loss: {epoch_loss/step:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        avg_train_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0
        print(f"\nEpoch {epoch} done in {elapsed:.0f}s. Avg train loss: {avg_train_loss:.4f}")

        # Validation
        val_metrics = validate(model, processor, val_loader, label2id, device, dtype)
        macro_f1 = val_metrics["macro_f1"]
        print(f"[VAL] loss={val_metrics['val_loss']:.4f} | acc={val_metrics['accuracy']:.4f} | macro_f1={macro_f1:.4f}")

        log = {"epoch": epoch, "train_loss": avg_train_loss,
               "time_sec": elapsed, **val_metrics}
        logs.append(log)
        pd.DataFrame(logs).to_csv(log_dir / "training_log.csv", index=False)

        # Best checkpoint
        if macro_f1 > best_f1:
            best_f1    = macro_f1
            best_epoch = epoch
            no_improve = 0
            ckpt_path = str(ckpt_dir / "best_checkpoint")
            model.save_pretrained(ckpt_path)
            print(f"  [SAVED] Best checkpoint → {ckpt_path}")
        else:
            no_improve += 1
            print(f"  [EarlyStopping] No improvement for {no_improve}/{patience} epochs.")
            if no_improve >= patience:
                print(f"  [EarlyStopping] Stopping at epoch {epoch}.")
                break

        # Per-epoch checkpoint
        model.save_pretrained(str(ckpt_dir / f"epoch_{epoch}"))

    # Final summary
    with open(output_dir / "run_summary.json", "w") as f:
        json.dump({"best_epoch": best_epoch, "best_val_macro_f1": best_f1,
                   "checkpoint": str(ckpt_dir / "best_checkpoint")}, f, indent=2)

    print(f"\n{'='*60}\nTRAINING COMPLETE\nBest Epoch: {best_epoch} | Best Val Macro F1: {best_f1:.4f}\n{'='*60}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)
