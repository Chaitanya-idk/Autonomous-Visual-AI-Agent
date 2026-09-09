"""
Training pipeline for SAGE crop-disease LoRA fine-tuning.

Chunked mode is the primary Kaggle workflow:
    - keep validation shard 47 on disk
    - download two training Parquet shards at a time
    - train the SAME in-memory LoRA model on that chunk
    - delete the chunk
    - continue with the next chunk
    - after all training shards have been consumed, run validation
    - repeat the complete training-shard pass for every epoch

For the default 48-shard SAGE layout this means:
    Epoch 1: [0,1] -> [2,3] -> ... -> [44,45] -> [46] -> validate [47]
    Epoch 2: [0,1] -> [2,3] -> ... -> [44,45] -> [46] -> validate [47]
    Epoch 3: [0,1] -> [2,3] -> ... -> [44,45] -> [46] -> validate [47]

The base model is loaded exactly once. The LoRA weights are NOT reset between
chunks or epochs.
"""

import gc
import json
import math
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import requests
import torch
import yaml
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AutoProcessor, get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    set_seed,
    compute_classification_metrics,
    save_labels_vocab,
    normalize_label_text,
    match_prediction_to_vocab,
)
from src.dataset import StreamingSAGEDataset, SAGEDataset, sage_collate_fn
from src.model import get_qwen_lora_model


# ─────────────────────────────────────────────────────────────────────────────
# Hugging Face shard download
# ─────────────────────────────────────────────────────────────────────────────

def download_shard(
    repo_id: str,
    shard_idx: int,
    dest_dir: Path,
    token: Optional[str] = None,
) -> Path:
    """Download one SAGE Parquet shard directly to dest_dir."""
    filename = f"train-{shard_idx:05d}.parquet"
    dest_path = dest_dir / filename

    if dest_path.exists() and dest_path.stat().st_size > 1000:
        print(f"  [Cache] Shard {shard_idx:02d} already exists; reusing it.")
        return dest_path

    dest_dir.mkdir(parents=True, exist_ok=True)
    temp_path = dest_dir / f"{filename}.tmp"
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/data/{filename}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    with requests.get(url, headers=headers, stream=True, timeout=120) as response:
        response.raise_for_status()
        total_bytes = int(response.headers.get("content-length", 0))

        with open(temp_path, "wb") as f, tqdm(
            total=total_bytes if total_bytes > 0 else None,
            unit="B",
            unit_scale=True,
            desc=f"  Download shard {shard_idx:02d}",
            leave=False,
        ) as pbar:
            for data in response.iter_content(chunk_size=2 * 1024 * 1024):
                if data:
                    f.write(data)
                    pbar.update(len(data))

    temp_path.replace(dest_path)
    return dest_path


def delete_shard_files(directory: Path, shard_group: List[int]) -> None:
    """Delete only the training shard files belonging to shard_group."""
    for shard_idx in shard_group:
        path = directory / f"train-{shard_idx:05d}.parquet"
        if path.exists():
            try:
                path.unlink()
            except OSError as exc:
                print(f"  [Warning] Could not delete {path}: {exc}")


def clear_training_chunk(directory: Path) -> None:
    """Remove any stale training parquet/temp files before a fresh chunk."""
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("train-*.parquet"):
        try:
            path.unlink()
        except OSError:
            pass
    for path in directory.glob("*.tmp"):
        try:
            path.unlink()
        except OSError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    cfg_file = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "config.yaml"
    with open(cfg_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Label vocabulary
# ─────────────────────────────────────────────────────────────────────────────

def load_or_create_label_vocab(
    labels_path: Path,
    validation_dir: Path,
    processor,
    common: Dict[str, Any],
) -> Dict[str, int]:
    """
    Load the existing authoritative vocabulary if available.

    For a fresh chunked run, the fixed validation shard is used to create the
    vocabulary. The resulting labels.json is then reused by every train chunk,
    validation pass, and final evaluation. It is never rebuilt per chunk.
    """
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        label2id = data["label2id"]
        print(f"[Labels] Loaded {len(label2id)} classes from {labels_path}")
        return label2id

    temp_ds = SAGEDataset(
        parquet_dir=str(validation_dir),
        processor=processor,
        is_training=False,
        **common,
    )
    label2id = temp_ds.label2id

    labels_path.parent.mkdir(parents=True, exist_ok=True)
    save_labels_vocab(label2id, labels_path)
    print(f"[Labels] Created {len(label2id)} classes from validation shard -> {labels_path}")
    return label2id


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def _autocast_context(dtype: torch.dtype):
    """Return a CUDA autocast context when CUDA is available, otherwise no-op."""
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=dtype)
    from contextlib import nullcontext
    return nullcontext()


def validate(
    model,
    processor,
    val_loss_loader,
    val_gen_loader,
    label2id: Dict[str, int],
    device: torch.device,
    dtype: torch.dtype,
    max_batches: Optional[int] = 200,
    epoch: Optional[int] = None,
) -> Dict[str, float]:
    """
    Validate without leaking the answer into generation.

    val_loss_loader contains teacher-forced samples with labels and is used
    only for validation loss.

    val_gen_loader contains prompt-only samples and is used for model.generate()
    and classification metrics.
    """
    model.eval()

    total_loss = 0.0
    loss_count = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    unknown_count = 0

    loss_total = len(val_loss_loader)
    if max_batches is not None:
        loss_total = min(loss_total, max_batches)

    gen_total = len(val_gen_loader)
    if max_batches is not None:
        gen_total = min(gen_total, max_batches)

    print(f"\n  [Validation] Epoch {epoch if epoch is not None else ''}".strip())

    # 1) Teacher-forced validation loss.
    loss_pbar = tqdm(
        enumerate(val_loss_loader),
        total=loss_total,
        desc="  Val loss",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )
    with torch.no_grad():
        for i, batch in loss_pbar:
            if max_batches is not None and i >= max_batches:
                break

            ids = batch["input_ids"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)
            pv = batch["pixel_values"].to(device=device, dtype=dtype, non_blocking=True)
            thw = batch["image_grid_thw"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with _autocast_context(dtype):
                out = model(
                    input_ids=ids,
                    attention_mask=attn,
                    pixel_values=pv,
                    image_grid_thw=thw,
                    labels=labels,
                )

            total_loss += float(out.loss.detach().float().item())
            loss_count += 1
            loss_pbar.set_postfix(val_loss=f"{total_loss / max(loss_count, 1):.4f}")

    avg_val_loss = total_loss / max(loss_count, 1)

    # 2) Generation-based classification evaluation.
    gen_pbar = tqdm(
        enumerate(val_gen_loader),
        total=gen_total,
        desc="  Val generate",
        unit="image",
        dynamic_ncols=True,
        leave=False,
    )
    with torch.no_grad():
        for i, batch in gen_pbar:
            if max_batches is not None and i >= max_batches:
                break

            ids = batch["input_ids"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)
            pv = batch["pixel_values"].to(device=device, dtype=dtype, non_blocking=True)
            thw = batch["image_grid_thw"].to(device, non_blocking=True)

            true_id = int(batch["label_id"][0].item())

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=attn,
                pixel_values=pv,
                image_grid_thw=thw,
                max_new_tokens=32,
                do_sample=False,
            )

            prompt_len = ids.shape[1]
            new_tokens = generated_ids[0, prompt_len:]
            raw_text = processor.tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            ).strip()

            _, pred_id, status = match_prediction_to_vocab(raw_text, label2id)

            if true_id >= 0:
                y_true.append(true_id)
                y_pred.append(pred_id)
            if status == "unknown":
                unknown_count += 1

            running_acc = (
                sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true)
                if y_true else 0.0
            )
            gen_pbar.set_postfix(acc=f"{running_acc:.2%}", unknown=unknown_count)

    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        labels_list=list(label2id.values()),
    )
    metrics["val_loss"] = float(avg_val_loss)
    metrics["unknown_predictions"] = int(unknown_count)
    metrics["unknown_rate"] = float(unknown_count / max(len(y_true), 1))
    metrics["num_eval_samples"] = int(len(y_true))

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    checkpoint_dir: Path,
    state: Dict[str, Any],
) -> None:
    """Save adapter weights plus optimizer/scheduler/training state."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(checkpoint_dir))
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
    with open(checkpoint_dir / "trainer_state.json", "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def optimizer_step(
    model,
    optimizer,
    scheduler,
    train_cfg: Dict[str, Any],
) -> None:
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    torch.nn.utils.clip_grad_norm_(
        trainable_params,
        float(train_cfg.get("max_grad_norm", 1.0)),
    )
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


# ─────────────────────────────────────────────────────────────────────────────
# Non-chunked dataset builder (kept for compatibility)
# ─────────────────────────────────────────────────────────────────────────────

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
        print("[Train] Using STREAMING mode (HF Hub)...")
        scan_ds = StreamingSAGEDataset(
            repo_id=ds_cfg["hf_repo_id"],
            split="train",
            max_samples=5000,
            is_training=False,
            **common,
        )
        label2id = scan_ds._build_label2id_from_streaming(scan_samples=5000)

        train_ds = StreamingSAGEDataset(
            repo_id=ds_cfg["hf_repo_id"],
            split="train",
            max_samples=ds_cfg.get("train_size", 50000),
            is_training=True,
            label2id=label2id,
            seed=ds_cfg.get("seed", 42),
            **common,
        )
        val_ds = StreamingSAGEDataset(
            repo_id=ds_cfg["hf_repo_id"],
            split="train",
            max_samples=ds_cfg.get("val_size", 5000),
            is_training=True,
            label2id=label2id,
            seed=ds_cfg.get("seed", 42) + 1,
            **common,
        )
        return train_ds, val_ds, label2id

    parquet_dir = Path(ds_cfg.get("parquet_dir", "data/parquet"))
    if not parquet_dir.is_absolute():
        parquet_dir = PROJECT_ROOT / parquet_dir

    from torch.utils.data import random_split
    full_ds = SAGEDataset(parquet_dir=str(parquet_dir), is_training=True, **common)
    label2id = full_ds.label2id
    n = len(full_ds)
    val_n = int(n * ds_cfg.get("val_split", 0.15))
    train_n = n - val_n
    train_ds, val_ds = random_split(
        full_ds,
        [train_n, val_n],
        generator=torch.Generator().manual_seed(ds_cfg.get("seed", 42)),
    )
    return train_ds, val_ds, label2id


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: Dict[str, Any]) -> None:
    train_cfg = cfg["training"]
    ds_cfg = cfg["dataset"]
    set_seed(int(train_cfg.get("seed", 42)))

    if not torch.cuda.is_available():
        raise RuntimeError("This training configuration is intended for a CUDA GPU (Kaggle T4/P100/A100).")

    device = torch.device("cuda")
    model_dtype_name = cfg["model"].get("torch_dtype", "float16").lower()
    dtype = torch.bfloat16 if model_dtype_name == "bfloat16" else torch.float16

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    output_dir = Path(cfg["paths"]["output_dir"])
    log_dir = Path(cfg["paths"]["log_dir"])
    for d in (ckpt_dir, output_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    prep_cfg = cfg.get("preprocessing", {})
    processor = AutoProcessor.from_pretrained(
        cfg["model"]["name_or_path"],
        local_files_only=cfg["model"].get("local_files_only", False),
        use_fast=False,
    )
    if prep_cfg.get("min_pixels"):
        processor.image_processor.min_pixels = prep_cfg["min_pixels"]
    if prep_cfg.get("max_pixels"):
        processor.image_processor.max_pixels = prep_cfg["max_pixels"]

    ds_mode = ds_cfg.get(
        "mode",
        "chunked" if ds_cfg.get("chunked", False)
        else ("streaming" if ds_cfg.get("streaming", False) else "local"),
    )
    is_chunked = ds_mode == "chunked"
    is_streaming = ds_mode == "streaming"
    batch_size = int(train_cfg.get("batch_size", 4))
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 8))
    num_workers = int(train_cfg.get("num_workers", 2))
    num_epochs = int(train_cfg.get("num_epochs", 3))
    patience = int(train_cfg.get("early_stopping_patience", 2))
    lr = float(train_cfg.get("learning_rate", 1e-4))

    common = dict(
        image_col=ds_cfg.get("image_col", "image"),
        disease_col=ds_cfg.get("disease_col", "disease"),
        crop_col=ds_cfg.get("crop_col", "crop"),
        label_id_col=ds_cfg.get("label_id_col", ""),
    )

    labels_path = Path(cfg["paths"].get("labels_json", "/kaggle/working/labels.json"))

    # ── Prepare data access ───────────────────────────────────────────────────
    if is_chunked:
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("[Warning] HF_TOKEN is not set. Public Hub access may still work, but private/gated access will fail.")

        repo_id = ds_cfg.get("hf_repo_id", "tirtho149/SAGE")
        total_shards = int(ds_cfg.get("total_shards", 48))
        chunk_size = int(ds_cfg.get("chunk_size", 2))
        if chunk_size < 1:
            raise ValueError("dataset.chunk_size must be >= 1")
        if total_shards < 2:
            raise ValueError("dataset.total_shards must be >= 2 because one shard is reserved for validation")

        chunk_dir = Path(ds_cfg.get("chunk_dir", "/kaggle/working/data_cache"))
        train_dir = chunk_dir / "train"
        val_dir = chunk_dir / "val"
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        # Shard 47 is held out. It stays on disk for the entire run so that it
        # can be reused for validation after every epoch.
        val_shard_idx = total_shards - 1
        print(f"\n[Chunked Mode] Validation shard = {val_shard_idx:02d}")
        download_shard(repo_id, val_shard_idx, val_dir, token=token)

        label2id = load_or_create_label_vocab(
            labels_path,
            val_dir,
            processor,
            common,
        )

        # IMPORTANT: two validation datasets with the same held-out shard.
        # Loss loader has labels; generation loader has prompt only.
        val_loss_ds = SAGEDataset(
            parquet_dir=str(val_dir),
            processor=processor,
            is_training=True,
            label2id=label2id,
            **common,
        )
        val_gen_ds = SAGEDataset(
            parquet_dir=str(val_dir),
            processor=processor,
            is_training=False,
            label2id=label2id,
            **common,
        )
        val_loss_loader = DataLoader(
            val_loss_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=sage_collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_gen_loader = DataLoader(
            val_gen_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=sage_collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )
        print(
            f"[Chunked Mode] {len(label2id)} disease classes | "
            f"validation shard {val_shard_idx:02d} rows={len(val_gen_ds)}"
        )

        train_shards = list(range(val_shard_idx))
        chunks = [
            train_shards[i:i + chunk_size]
            for i in range(0, len(train_shards), chunk_size)
        ]

        max_chunks = ds_cfg.get("max_chunks_per_epoch")
        if max_chunks is not None:
            max_chunks = int(max_chunks)
            if max_chunks > 0:
                chunks = chunks[:max_chunks]

        if not chunks:
            raise RuntimeError("No training chunks configured. Check total_shards/chunk_size/max_chunks_per_epoch.")

        print(f"[Chunked Mode] Training shards: {train_shards[0]}..{train_shards[-1]} ({len(train_shards)} shards)")
        print(f"[Chunked Mode] Chunks per epoch: {len(chunks)}")
        print(f"[Chunked Mode] Chunk plan: {chunks}")

        # Used only to size the LR scheduler before the first chunk is loaded.
        # SAGE shards are approximately this many rows; exact per-chunk batch
        # counts are tracked during training. The scheduler is deliberately
        # given a slightly conservative estimate so it does not finish early.
        estimated_rows_per_shard = int(ds_cfg.get("estimated_rows_per_shard", 15900))
        estimated_optimizer_steps_per_epoch = 0
        for shard_group in chunks:
            estimated_rows = len(shard_group) * estimated_rows_per_shard
            estimated_batches = math.ceil(estimated_rows / batch_size)
            # We force a step at the end of every chunk so that no gradients
            # remain stranded while the Parquet files are flushed.
            estimated_optimizer_steps_per_epoch += math.ceil(estimated_batches / grad_accum)
        steps_per_epoch_estimate = estimated_optimizer_steps_per_epoch
    else:
        train_ds, val_ds, label2id = build_datasets(cfg, processor)
        save_labels_vocab(label2id, labels_path)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=not is_streaming,
            collate_fn=sage_collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loss_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=sage_collate_fn,
            num_workers=num_workers,
        )
        val_gen_loader = val_loss_loader
        steps_per_epoch_estimate = (
            math.ceil(len(train_ds) / batch_size) if is_streaming else len(train_loader)
        )

    # ── Load base model ONCE ──────────────────────────────────────────────────
    # This object survives every chunk and every epoch. We never reload the
    # base model inside the chunk loop.
    lora_cfg = cfg["lora"]
    model = get_qwen_lora_model(
        model_name_or_path=cfg["model"]["name_or_path"],
        lora_r=int(lora_cfg.get("r", 32)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 64)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        torch_dtype=cfg["model"].get("torch_dtype", "float16"),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        local_files_only=bool(cfg["model"].get("local_files_only", False)),
        is_trainable=True,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    # Use a scheduler large enough for the COMPLETE pass through all training
    # shards. This is intentionally based on all chunks, not max_chunks=1.
    total_optimizer_steps = max(1, steps_per_epoch_estimate * num_epochs)
    warmup_steps = int(total_optimizer_steps * float(train_cfg.get("warmup_ratio", 0.05)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    print("\n" + "=" * 72)
    print(
        f"STARTING TRAINING | epochs={num_epochs} | patience={patience} | "
        f"batch={batch_size} | grad_accum={grad_accum} | workers={num_workers}"
    )
    print(f"Mode: {ds_mode.upper()} {'(FULL rolling-window pass)' if is_chunked else ''}")
    if is_chunked:
        print(f"Training chunks per epoch: {len(chunks)}")
        print(f"Training shards per epoch: {sum(len(c) for c in chunks)}")
        print("Base model load count: 1 (model persists across every chunk/epoch)")
    print("=" * 72 + "\n")

    best_f1 = -1.0
    best_epoch = -1
    no_improve = 0
    logs: List[Dict[str, Any]] = []
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        epoch_optimizer_steps = 0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        if is_chunked:
            # ─────────────────────────────────────────────────────────────────
            # THE ROLLING WINDOW
            # ─────────────────────────────────────────────────────────────────
            # Every chunk is trained by the SAME model object. Only the local
            # Parquet files are replaced.
            for chunk_idx, shard_group in enumerate(chunks, 1):
                print(
                    f"\n>>> [Epoch {epoch}/{num_epochs}] "
                    f"Chunk {chunk_idx}/{len(chunks)} | Shards {shard_group} <<<",
                    flush=True,
                )

                clear_training_chunk(train_dir)
                for shard_idx in shard_group:
                    download_shard(repo_id, shard_idx, train_dir, token=token)

                chunk_ds = SAGEDataset(
                    parquet_dir=str(train_dir),
                    processor=processor,
                    is_training=True,
                    label2id=label2id,
                    **common,
                )
                chunk_loader = DataLoader(
                    chunk_ds,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=sage_collate_fn,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=False,
                )

                chunk_rows = len(chunk_ds)
                chunk_batches = len(chunk_loader)
                print(f"  [Chunk] rows={chunk_rows} | batches={chunk_batches}")

                pbar = tqdm(
                    chunk_loader,
                    desc=f"Ep {epoch} Ch {chunk_idx}/{len(chunks)}",
                    unit="step",
                    dynamic_ncols=True,
                    leave=True,
                )

                for batch_idx, batch in enumerate(pbar, 1):
                    ids = batch["input_ids"].to(device, non_blocking=True)
                    attn = batch["attention_mask"].to(device, non_blocking=True)
                    pv = batch["pixel_values"].to(device=device, dtype=dtype, non_blocking=True)
                    thw = batch["image_grid_thw"].to(device, non_blocking=True)
                    lbl = batch["labels"].to(device, non_blocking=True)

                    with _autocast_context(dtype):
                        out = model(
                            input_ids=ids,
                            attention_mask=attn,
                            pixel_values=pv,
                            image_grid_thw=thw,
                            labels=lbl,
                        )
                        loss = out.loss / grad_accum

                    loss.backward()
                    epoch_batches += 1
                    epoch_loss += float(out.loss.detach().float().item())

                    should_step = (
                        batch_idx % grad_accum == 0
                        or batch_idx == chunk_batches
                    )

                    if should_step:
                        # IMPORTANT: only scale by the actual number of batches
                        # in a partial accumulation group. The normal case is
                        # exactly grad_accum batches.
                        if batch_idx % grad_accum != 0:
                            remainder = batch_idx % grad_accum
                            scale = grad_accum / remainder
                            for p in trainable_params:
                                if p.grad is not None:
                                    p.grad.mul_(scale)

                        optimizer_step(model, optimizer, scheduler, train_cfg)
                        epoch_optimizer_steps += 1
                        global_step += 1

                    elapsed = time.time() - t0
                    items = epoch_batches * batch_size
                    postfix = {
                        "loss": f"{epoch_loss / max(epoch_batches, 1):.4f}",
                        "items/s": f"{items / max(elapsed, 0.001):.1f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    }
                    if torch.cuda.is_available():
                        postfix["vram"] = f"{torch.cuda.memory_allocated() / (1024**3):.1f}GB"
                    pbar.set_postfix(postfix)

                # Drop all references before deleting the Parquet files. This
                # keeps DataFrame memory and worker handles from accumulating.
                del pbar
                del chunk_loader
                del chunk_ds
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                delete_shard_files(train_dir, shard_group)
                print(
                    f"[Disk Cleared] Deleted shards {shard_group}. "
                    f"Only validation shard {val_shard_idx:02d} remains cached.",
                    flush=True,
                )

                # Save progress after every rolling window. The model weights,
                # optimizer, scheduler and current position are all preserved.
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    ckpt_dir / "latest_checkpoint",
                    {
                        "epoch": epoch,
                        "completed_chunk": chunk_idx,
                        "total_chunks": len(chunks),
                        "completed_shards": shard_group,
                        "global_optimizer_step": global_step,
                    },
                )
                print("  [Checkpoint] Rolling-window progress saved.", flush=True)

        else:
            pbar = tqdm(
                train_loader,
                total=steps_per_epoch_estimate,
                desc=f"Epoch {epoch}/{num_epochs}",
                unit="step",
                dynamic_ncols=True,
                leave=True,
            )
            total_batches = len(train_loader)
            for batch_idx, batch in enumerate(pbar, 1):
                ids = batch["input_ids"].to(device, non_blocking=True)
                attn = batch["attention_mask"].to(device, non_blocking=True)
                pv = batch["pixel_values"].to(device=device, dtype=dtype, non_blocking=True)
                thw = batch["image_grid_thw"].to(device, non_blocking=True)
                lbl = batch["labels"].to(device, non_blocking=True)

                with _autocast_context(dtype):
                    out = model(
                        input_ids=ids,
                        attention_mask=attn,
                        pixel_values=pv,
                        image_grid_thw=thw,
                        labels=lbl,
                    )
                    loss = out.loss / grad_accum

                loss.backward()
                epoch_batches += 1
                epoch_loss += float(out.loss.detach().float().item())

                if batch_idx % grad_accum == 0 or batch_idx == total_batches:
                    if batch_idx % grad_accum != 0:
                        remainder = batch_idx % grad_accum
                        scale = grad_accum / remainder
                        for p in trainable_params:
                            if p.grad is not None:
                                p.grad.mul_(scale)
                    optimizer_step(model, optimizer, scheduler, train_cfg)
                    epoch_optimizer_steps += 1
                    global_step += 1

                elapsed = time.time() - t0
                pbar.set_postfix(
                    loss=f"{epoch_loss / max(epoch_batches, 1):.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(epoch_batches, 1)
        total_items = epoch_batches * batch_size
        throughput = total_items / max(elapsed, 0.001)

        print(
            f"\n[Epoch {epoch} Done] {epoch_batches} batches | "
            f"{epoch_optimizer_steps} optimizer steps | "
            f"{total_items} items in {elapsed:.1f}s ({throughput:.2f} items/s) | "
            f"avg_loss={avg_loss:.4f}"
        )

        # ── Validation after ALL training chunks ─────────────────────────────
        val_max_batches = cfg.get("evaluation", {}).get("val_max_batches", 200)
        val_metrics = validate(
            model,
            processor,
            val_loss_loader,
            val_gen_loader,
            label2id,
            device,
            dtype,
            max_batches=val_max_batches,
            epoch=epoch,
        )

        macro_f1 = val_metrics["macro_f1"]
        print(
            f"[VAL Epoch {epoch}] "
            f"loss={val_metrics['val_loss']:.4f} | "
            f"acc={val_metrics['accuracy']:.4f} | "
            f"macro_f1={macro_f1:.4f} | "
            f"unknown={val_metrics['unknown_rate']:.2%} | "
            f"samples={val_metrics['num_eval_samples']}\n"
        )

        log_row = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_batches": epoch_batches,
            "optimizer_steps": epoch_optimizer_steps,
            "time_sec": elapsed,
            **val_metrics,
        }
        logs.append(log_row)
        pd.DataFrame(logs).to_csv(log_dir / "training_log.csv", index=False)

        # ── Best checkpoint ──────────────────────────────────────────────────
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                ckpt_dir / "best_checkpoint",
                {
                    "epoch": epoch,
                    "global_optimizer_step": global_step,
                    "best_val_macro_f1": best_f1,
                },
            )
            print(f"  [SAVED] Best checkpoint -> {ckpt_dir / 'best_checkpoint'}")
        else:
            no_improve += 1
            print(f"  [EarlyStopping] {no_improve}/{patience} epochs without improvement.")
            if no_improve >= patience:
                print(f"  Stopping early at epoch {epoch}.")
                break

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            ckpt_dir / f"epoch_{epoch}",
            {
                "epoch": epoch,
                "global_optimizer_step": global_step,
                "val_metrics": val_metrics,
            },
        )

    summary = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "checkpoint": str(ckpt_dir / "best_checkpoint"),
        "training_mode": ds_mode,
        "num_epochs_requested": num_epochs,
        "training_shards": list(range(ds_cfg.get("total_shards", 48) - 1)) if is_chunked else None,
        "validation_shard": ds_cfg.get("total_shards", 48) - 1 if is_chunked else None,
        "chunks_per_epoch": len(chunks) if is_chunked else None,
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"\n{'=' * 72}\n"
        f"DONE | Best Epoch: {best_epoch} | Best Macro F1: {best_f1:.4f}\n"
        f"{'=' * 72}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    train(load_config(args.config))
