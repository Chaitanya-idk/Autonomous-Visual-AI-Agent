"""
Training pipeline for SAGE crop-disease LoRA fine-tuning.

Current Kaggle design:
  - Base model is loaded from the Kaggle model input.
  - SAGE is accessed from Hugging Face using HF_TOKEN.
  - In chunked mode, only the active Parquet training shards are downloaded to
    /kaggle/working/data_cache, then deleted after use.
  - A fixed validation shard is kept separately.
  - Validation loss and generation metrics use separate datasets so the
    ground-truth answer is NEVER supplied to model.generate().
"""

import os
import sys
import json
import time
import yaml
import argparse
import re
import unicodedata
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from functools import partial

import pandas as pd
import requests
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    set_seed,
    save_labels_vocab,
    compute_classification_metrics,
    match_prediction,
)
from src.dataset import StreamingSAGEDataset, SAGEDataset, sage_collate_fn
from src.model import get_qwen_lora_model


# -----------------------------------------------------------------------------
# Hugging Face shard downloader
# -----------------------------------------------------------------------------

def download_shard(
    repo_id: str,
    shard_idx: int,
    dest_dir: Path,
    token: Optional[str] = None,
) -> Path:
    """Download one Parquet shard directly into the requested cache directory."""
    filename = f"train-{shard_idx:05d}.parquet"
    dest_dir = Path(dest_dir)
    dest_path = dest_dir / filename

    if dest_path.exists() and dest_path.stat().st_size > 1000:
        return dest_path

    dest_dir.mkdir(parents=True, exist_ok=True)
    temp_path = dest_dir / f"{filename}.tmp"

    url = (
        f"https://huggingface.co/datasets/{repo_id}/resolve/main/"
        f"data/{filename}"
    )
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    print(f"  [HF] Downloading shard {shard_idx:02d}...", flush=True)
    response = requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=120,
    )
    response.raise_for_status()

    total_bytes = int(response.headers.get("content-length", 0))

    try:
        with open(temp_path, "wb") as file:
            with tqdm(
                total=total_bytes or None,
                unit="B",
                unit_scale=True,
                desc=f"  Download shard {shard_idx:02d}",
                leave=False,
            ) as pbar:
                for chunk in response.iter_content(
                    chunk_size=2 * 1024 * 1024
                ):
                    if chunk:
                        file.write(chunk)
                        pbar.update(len(chunk))

        temp_path.replace(dest_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

    return dest_path


def delete_shards(shard_group: List[int], train_dir: Path):
    """Delete only the temporary training shards."""
    for shard_idx in shard_group:
        path = train_dir / f"train-{shard_idx:05d}.parquet"
        if path.exists():
            path.unlink()


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    cfg_file = (
        Path(config_path)
        if config_path
        else PROJECT_ROOT / "configs" / "config.yaml"
    )
    with open(cfg_file, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


# -----------------------------------------------------------------------------
# Runtime helpers
# -----------------------------------------------------------------------------

def autocast_context(device: torch.device, dtype: torch.dtype):
    """Use CUDA AMP on GPU and a no-op context on CPU."""
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def move_batch_to_device(batch, device, dtype):
    """Move the tensors used by Qwen2.5-VL to the target device."""
    ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    pixel_values = batch["pixel_values"].to(device, dtype=dtype)
    image_grid_thw = batch["image_grid_thw"].to(device)
    return ids, attn, pixel_values, image_grid_thw


def save_json(path: Path, data: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    ckpt_dir: Path,
    name: str,
    state: Dict[str, Any],
):
    """
    Save LoRA weights plus optimizer/scheduler/trainer state.

    This is a real training checkpoint rather than adapter weights alone.
    """
    path = ckpt_dir / name
    path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(path))
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    torch.save(scheduler.state_dict(), path / "scheduler.pt")
    save_json(path / "trainer_state.json", state)

    print(f"  [Checkpoint] Saved complete checkpoint → {path}", flush=True)
    return path


# -----------------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------------

def make_collate_fn(processor):
    """Bind the tokenizer's actual pad token instead of assuming pad_id=0."""
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0
    return partial(sage_collate_fn, pad_token_id=int(pad_token_id))


def build_datasets(cfg: Dict[str, Any], processor):
    """Build train/validation datasets for non-chunked modes."""
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
            max_samples=ds_cfg.get("label_scan_samples", 5000),
            is_training=False,
            seed=ds_cfg.get("seed", 42),
            **common,
        )
        label2id = scan_ds._build_label2id_from_streaming(
            scan_samples=ds_cfg.get("label_scan_samples", 5000)
        )

        train_size = int(ds_cfg.get("train_size", 3200))
        val_size = int(ds_cfg.get("val_size", 400))
        seed = int(ds_cfg.get("seed", 42))

        # Build two disjoint views of the same deterministic shuffled stream.
        # The previous implementation created two independent shuffled streams,
        # which could overlap and contaminate validation.
        from datasets import load_dataset

        base_stream = load_dataset(
            ds_cfg["hf_repo_id"],
            split="train",
            streaming=True,
        ).shuffle(seed=seed, buffer_size=1000)

        train_rows = base_stream.take(train_size)
        val_rows = base_stream.skip(train_size).take(val_size)

        class _WrappedStreaming(StreamingSAGEDataset):
            def __init__(self, hf_rows, **kwargs):
                self._hf_ds = hf_rows
                self.repo_id = kwargs["repo_id"]
                self.processor = kwargs["processor"]
                self.is_training = kwargs["is_training"]
                self.image_col = kwargs["image_col"]
                self.disease_col = kwargs["disease_col"]
                self.crop_col = kwargs["crop_col"] or None
                self.label_id_col = kwargs["label_id_col"] or None
                self.max_samples = kwargs["max_samples"]
                self.label2id = kwargs["label2id"]

        stream_common = dict(
            repo_id=ds_cfg["hf_repo_id"],
            processor=processor,
            image_col=common["image_col"],
            disease_col=common["disease_col"],
            crop_col=common["crop_col"],
            label_id_col=common["label_id_col"],
            label2id=label2id,
        )

        train_ds = _WrappedStreaming(
            train_rows,
            **stream_common,
            is_training=True,
            max_samples=train_size,
        )
        val_loss_ds = _WrappedStreaming(
            val_rows,
            **stream_common,
            is_training=True,
            max_samples=val_size,
        )

        # Re-create the deterministic stream for generation because the
        # original IterableDataset is consumed after one pass.
        gen_stream = (
            load_dataset(
                ds_cfg["hf_repo_id"],
                split="train",
                streaming=True,
            )
            .shuffle(seed=seed, buffer_size=1000)
            .skip(train_size)
            .take(val_size)
        )
        val_gen_ds = _WrappedStreaming(
            gen_stream,
            **stream_common,
            is_training=False,
            max_samples=val_size,
        )

        return train_ds, val_loss_ds, val_gen_ds, label2id

    # Local mode: use one deterministic split of row indices, then create two
    # independent dataset objects so validation can have both formats.
    parquet_dir = ds_cfg.get("parquet_dir", "data/parquet")
    if not Path(parquet_dir).is_absolute():
        parquet_dir = str(PROJECT_ROOT / parquet_dir)

    print(f"[Train] Loading local parquet from {parquet_dir}...")
    from torch.utils.data import random_split

    full_train = SAGEDataset(
        parquet_dir=parquet_dir,
        processor=processor,
        is_training=True,
        **common,
    )
    label2id = full_train.label2id
    n = len(full_train)
    val_n = max(1, int(n * ds_cfg.get("val_split", 0.15)))
    train_n = n - val_n

    train_subset, val_subset = random_split(
        range(n),
        [train_n, val_n],
        generator=torch.Generator().manual_seed(ds_cfg.get("seed", 42)),
    )

    # Dataset objects read the same Parquet source but use different modes.
    train_ds_full = SAGEDataset(
        parquet_dir=parquet_dir,
        processor=processor,
        is_training=True,
        label2id=label2id,
        **common,
    )
    val_loss_full = SAGEDataset(
        parquet_dir=parquet_dir,
        processor=processor,
        is_training=True,
        label2id=label2id,
        **common,
    )
    val_gen_full = SAGEDataset(
        parquet_dir=parquet_dir,
        processor=processor,
        is_training=False,
        label2id=label2id,
        **common,
    )

    from torch.utils.data import Subset
    train_ds = Subset(train_ds_full, train_subset.indices)
    val_loss_ds = Subset(val_loss_full, val_subset.indices)
    val_gen_ds = Subset(val_gen_full, val_subset.indices)

    return train_ds, val_loss_ds, val_gen_ds, label2id


def prepare_chunked_validation(
    cfg: Dict[str, Any],
    processor,
    label2id: Dict[str, int],
    val_dir: Path,
    collate_fn,
):
    """Create separate supervised-loss and generation validation loaders."""
    ds_cfg = cfg["dataset"]
    common = dict(
        processor=processor,
        image_col=ds_cfg.get("image_col", "image"),
        disease_col=ds_cfg.get("disease_col", "disease"),
        crop_col=ds_cfg.get("crop_col", "crop"),
        label_id_col=ds_cfg.get("label_id_col", ""),
    )

    # Loss dataset contains the answer so model(...) can calculate CE loss.
    val_loss_ds = SAGEDataset(
        parquet_dir=str(val_dir),
        is_training=True,
        label2id=label2id,
        **common,
    )

    # Generation dataset contains ONLY image + prompt. This prevents label
    # leakage and is the only dataset allowed to feed model.generate().
    val_gen_ds = SAGEDataset(
        parquet_dir=str(val_dir),
        is_training=False,
        label2id=label2id,
        **common,
    )

    val_loss_loader = DataLoader(
        val_loss_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    val_gen_loader = DataLoader(
        val_gen_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    print(
        f"[Validation] loss samples={len(val_loss_ds)} | "
        f"generation samples={len(val_gen_ds)}"
    )
    return val_loss_loader, val_gen_loader


# -----------------------------------------------------------------------------
# Validation / evaluation
# -----------------------------------------------------------------------------

def validate(
    model,
    processor,
    val_loss_loader,
    val_gen_loader,
    label2id,
    device,
    dtype,
    max_batches=200,
    epoch=None,
):
    """
    Validate the model in two independent passes.

    Pass 1: supervised validation loss, where the answer is present only in
            labels and never used for generation.
    Pass 2: true autoregressive generation, where the model receives image +
            prompt and must produce the disease itself.
    """
    model.eval()

    total_loss = 0.0
    loss_count = 0

    # ------------------------------------------------------------------
    # 1. Validation loss
    # ------------------------------------------------------------------
    loss_pbar = tqdm(
        val_loss_loader,
        total=max_batches,
        desc=(f"  Val loss [Epoch {epoch}]" if epoch else "  Val loss"),
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )

    with torch.no_grad():
        for i, batch in enumerate(loss_pbar):
            if i >= max_batches:
                break

            ids, attn, pixel_values, image_grid_thw = move_batch_to_device(
                batch, device, dtype
            )
            labels = batch["labels"].to(device)

            with autocast_context(device, dtype):
                output = model(
                    input_ids=ids,
                    attention_mask=attn,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    labels=labels,
                )

            if output.loss is None or not torch.isfinite(output.loss):
                continue

            total_loss += float(output.loss.item())
            loss_count += 1
            loss_pbar.set_postfix(
                val_loss=f"{total_loss / max(loss_count, 1):.4f}"
            )

    avg_val_loss = total_loss / max(loss_count, 1)

    # Generation can use KV cache; training had it disabled for gradient
    # checkpointing. Turn it back on for the inference-style validation pass.
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    # ------------------------------------------------------------------
    # 2. Generative classification
    # ------------------------------------------------------------------
    y_true: List[int] = []
    y_pred: List[int] = []
    statuses: Dict[str, int] = {}

    gen_pbar = tqdm(
        val_gen_loader,
        total=max_batches,
        desc=(
            f"  Val generate [Epoch {epoch}]"
            if epoch
            else "  Val generate"
        ),
        unit="image",
        dynamic_ncols=True,
        leave=False,
    )

    with torch.no_grad():
        for i, batch in enumerate(gen_pbar):
            if i >= max_batches:
                break

            ids, attn, pixel_values, image_grid_thw = move_batch_to_device(
                batch, device, dtype
            )

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=attn,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=32,
                do_sample=False,
                num_beams=1,
            )

            # Validation generation uses batch_size=1. The generated tensor
            # contains the prompt followed by newly generated tokens.
            prompt_len = ids.shape[1]
            new_tokens = generated_ids[0, prompt_len:]
            raw_text = processor.tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            ).strip()

            _, pred_id, status = match_prediction(raw_text, label2id)
            true_id = int(batch["label_id"][0].item())

            if true_id < 0:
                continue

            y_true.append(true_id)
            y_pred.append(pred_id)
            statuses[status] = statuses.get(status, 0) + 1

            running_acc = sum(
                yt == yp for yt, yp in zip(y_true, y_pred)
            ) / max(len(y_true), 1)
            gen_pbar.set_postfix(acc=f"{running_acc:.2%}")

    official_ids = sorted(label2id.values())
    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        labels_list=official_ids,
    )
    metrics["val_loss"] = float(avg_val_loss)
    metrics["validation_samples"] = int(len(y_true))
    metrics["prediction_valid_exact"] = int(statuses.get("valid_exact", 0))
    metrics["prediction_valid_embedded"] = int(
        statuses.get("valid_embedded", 0)
    )
    metrics["prediction_unknown"] = int(statuses.get("unknown", 0))

    # Put the model back into the training-safe configuration for the next
    # epoch.
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    return metrics


# -----------------------------------------------------------------------------
# Training primitives
# -----------------------------------------------------------------------------

def optimizer_step(
    model,
    optimizer,
    scheduler,
    train_cfg,
):
    """Clip gradients and perform one optimizer/scheduler step."""
    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]

    if trainable_params:
        torch.nn.utils.clip_grad_norm_(
            trainable_params,
            float(train_cfg.get("max_grad_norm", 1.0)),
        )

    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def flush_partial_accumulation(
    model,
    optimizer,
    scheduler,
    train_cfg,
    accumulated_batches: int,
    grad_accum: int,
):
    """
    Finish a final partial accumulation group correctly.

    Each training loss is divided by grad_accum. If only N < grad_accum
    batches remain, scale the accumulated gradients by grad_accum/N before the
    optimizer step so the final group has the same average-gradient scale.
    """
    if accumulated_batches <= 0:
        return False

    if accumulated_batches < grad_accum:
        scale = grad_accum / accumulated_batches
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.grad is not None:
                parameter.grad.mul_(scale)

    optimizer_step(model, optimizer, scheduler, train_cfg)
    return True


def train_batch(
    model,
    batch,
    device,
    dtype,
    grad_accum,
):
    """Run one forward/backward pass and return the unscaled loss."""
    ids, attn, pixel_values, image_grid_thw = move_batch_to_device(
        batch, device, dtype
    )
    labels = batch["labels"].to(device)

    with autocast_context(device, dtype):
        output = model(
            input_ids=ids,
            attention_mask=attn,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
        )
        loss = output.loss

    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite training loss: {loss.item()}")

    (loss / grad_accum).backward()
    return float(loss.item())


# -----------------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------------

def train(cfg: Dict[str, Any]):
    train_cfg = cfg["training"]
    set_seed(int(train_cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = bool(train_cfg.get("bf16", False))
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    if device.type == "cpu":
        dtype = torch.float32

    # Output directories
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    output_dir = Path(cfg["paths"]["output_dir"])
    log_dir = Path(cfg["paths"]["log_dir"])
    for directory in (ckpt_dir, output_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    save_json(output_dir / "config.json", cfg)

    # Processor
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

    collate_fn = make_collate_fn(processor)

    # Dataset mode
    ds_cfg = cfg["dataset"]
    ds_mode = ds_cfg.get(
        "mode",
        "chunked"
        if ds_cfg.get("chunked", False)
        else ("streaming" if ds_cfg.get("streaming", False) else "local"),
    )
    is_chunked = ds_mode == "chunked"
    is_streaming = ds_mode == "streaming"
    batch_size = int(train_cfg.get("batch_size", 2))

    labels_path = Path(cfg["paths"].get("labels_json", "data/labels.json"))

    # These are populated for every mode.
    train_loader = None
    val_loss_loader = None
    val_gen_loader = None

    if is_chunked:
        token = os.environ.get("HF_TOKEN")
        if not token:
            print(
                "[Warning] HF_TOKEN is not set. The dataset may still be public, "
                "but authenticated access is recommended."
            )

        repo_id = ds_cfg.get("hf_repo_id", "tirtho149/SAGE")
        total_shards = int(ds_cfg.get("total_shards", 48))
        chunk_size = int(ds_cfg.get("chunk_size", 2))
        chunk_dir = Path(
            ds_cfg.get("chunk_dir", "/kaggle/working/data_cache")
        )
        val_dir = chunk_dir / "val"
        train_dir = chunk_dir / "train"
        val_dir.mkdir(parents=True, exist_ok=True)
        train_dir.mkdir(parents=True, exist_ok=True)

        # Fixed validation shard. It is deliberately excluded from training.
        val_shard_idx = total_shards - 1
        print(
            f"\n[Chunked Mode] Downloading fixed validation shard "
            f"{val_shard_idx:02d}..."
        )
        download_shard(repo_id, val_shard_idx, val_dir, token=token)

        if labels_path.exists():
            with open(labels_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            label2id = {str(k): int(v) for k, v in data["label2id"].items()}
            print(
                f"[Labels] Loaded existing vocabulary: "
                f"{len(label2id)} classes → {labels_path}"
            )
        else:
            # The first run uses the fixed validation shard to establish a
            # deterministic vocabulary. Later runs reuse labels.json.
            temp_ds = SAGEDataset(
                parquet_dir=str(val_dir),
                processor=processor,
                is_training=False,
                **{
                    "image_col": ds_cfg.get("image_col", "image"),
                    "disease_col": ds_cfg.get("disease_col", "disease"),
                    "crop_col": ds_cfg.get("crop_col", "crop"),
                    "label_id_col": ds_cfg.get("label_id_col", ""),
                },
            )
            label2id = temp_ds.label2id
            save_labels_vocab(labels_path, label2id)
            print(
                f"[Labels] Created vocabulary from validation shard: "
                f"{len(label2id)} classes → {labels_path}"
            )

        val_loss_loader, val_gen_loader = prepare_chunked_validation(
            cfg,
            processor,
            label2id,
            val_dir,
            collate_fn,
        )
        print(
            f"[Chunked Mode] Validation shard={val_shard_idx:02d} | "
            f"classes={len(label2id)}"
        )

        # Training shards exclude the fixed validation shard.
        train_shards = list(range(val_shard_idx))
        chunks = [
            train_shards[i:i + chunk_size]
            for i in range(0, len(train_shards), chunk_size)
        ]
        max_chunks = ds_cfg.get("max_chunks_per_epoch")
        if max_chunks:
            chunks = chunks[: int(max_chunks)]

        if not chunks:
            raise ValueError("No training chunks are configured.")

        # SAGE shards are approximately 15.9k rows each. This is only used to
        # size the LR scheduler before the rolling shards are downloaded. The
        # actual batch count is measured from each Parquet chunk.
        estimated_rows_per_shard = int(
            ds_cfg.get("estimated_rows_per_shard", 15900)
        )
        estimated_batches = max(
            1,
            sum(
                min(len(group), chunk_size) * estimated_rows_per_shard
                // batch_size
                for group in chunks
            ),
        )
        steps_per_epoch = estimated_batches

    else:
        (
            train_ds,
            val_loss_ds,
            val_gen_ds,
            label2id,
        ) = build_datasets(cfg, processor)

        save_labels_vocab(labels_path, label2id)
        print(
            f"[Train] {len(label2id)} disease classes. "
            f"Labels saved → {labels_path}"
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=not is_streaming,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
        )
        val_loss_loader = DataLoader(
            val_loss_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
        )
        val_gen_loader = DataLoader(
            val_gen_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
        )

        steps_per_epoch = (
            int(ds_cfg.get("train_size", 3200)) // batch_size
            if is_streaming
            else len(train_loader)
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    lora_cfg = cfg["lora"]
    model = get_qwen_lora_model(
        model_name_or_path=cfg["model"]["name_or_path"],
        lora_r=int(lora_cfg.get("r", 32)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 64)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
        torch_dtype=cfg["model"].get("torch_dtype", "float16"),
        gradient_checkpointing=bool(
            train_cfg.get("gradient_checkpointing", True)
        ),
        local_files_only=bool(
            cfg["model"].get("local_files_only", False)
        ),
        is_trainable=True,
    )

    # Make sure generation uses the model's configured pad/eos IDs.
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id
    if getattr(model.config, "eos_token_id", None) is None:
        model.config.eos_token_id = processor.tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # Optimizer + scheduler
    # ------------------------------------------------------------------
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 16))
    num_epochs = int(train_cfg.get("num_epochs", 10))
    patience = int(train_cfg.get("early_stopping_patience", 7))
    lr = float(train_cfg.get("learning_rate", 1e-4))

    optimizer_steps_per_epoch = max(
        1,
        (steps_per_epoch + grad_accum - 1) // grad_accum,
    )
    total_steps = optimizer_steps_per_epoch * num_epochs
    warmup_steps = int(
        total_steps * float(train_cfg.get("warmup_ratio", 0.05))
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(1, total_steps),
    )

    print(f"\n{'=' * 60}")
    print(
        f"STARTING TRAINING | epochs={num_epochs} | "
        f"patience={patience} | batch={batch_size} | "
        f"grad_accum={grad_accum}"
    )
    print(
        f"Mode: {ds_mode.upper()} "
        f"{'(Auto-flush rolling window)' if is_chunked else ''}"
    )
    print(
        f"Estimated optimizer steps/epoch: {optimizer_steps_per_epoch} | "
        f"total scheduler steps: {total_steps}"
    )
    print(f"{'=' * 60}\n")

    best_f1 = -1.0
    best_epoch = -1
    no_improve = 0
    logs: List[Dict[str, Any]] = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        step = 0
        accumulated_batches = 0
        optimizer_steps_done = 0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        def process_loader(loader, total=None, desc="Training"):
            nonlocal epoch_loss, step, accumulated_batches, optimizer_steps_done

            pbar = tqdm(
                loader,
                total=total,
                desc=desc,
                unit="step",
                dynamic_ncols=True,
                leave=True,
            )

            for batch in pbar:
                loss_value = train_batch(
                    model,
                    batch,
                    device,
                    dtype,
                    grad_accum,
                )
                step += 1
                accumulated_batches += 1
                epoch_loss += loss_value

                if accumulated_batches == grad_accum:
                    optimizer_step(
                        model,
                        optimizer,
                        scheduler,
                        train_cfg,
                    )
                    accumulated_batches = 0
                    optimizer_steps_done += 1

                elapsed_sec = time.time() - t0
                items_per_sec = (step * batch_size) / max(elapsed_sec, 0.001)
                postfix = {
                    "loss": f"{epoch_loss / max(step, 1):.4f}",
                    "items/s": f"{items_per_sec:.1f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.1e}",
                }
                if torch.cuda.is_available():
                    postfix["vram"] = (
                        f"{torch.cuda.memory_allocated() / (1024 ** 3):.1f}GB"
                    )
                pbar.set_postfix(postfix)

            pbar.close()

        if is_chunked:
            for chunk_idx, shard_group in enumerate(chunks, 1):
                print(
                    f"\n>>> [Epoch {epoch}/{num_epochs}] "
                    f"Downloading Chunk {chunk_idx}/{len(chunks)} "
                    f"(Shards: {shard_group}) <<<",
                    flush=True,
                )

                for shard_idx in shard_group:
                    download_shard(
                        repo_id,
                        shard_idx,
                        train_dir,
                        token=token,
                    )

                common = dict(
                    processor=processor,
                    image_col=ds_cfg.get("image_col", "image"),
                    disease_col=ds_cfg.get("disease_col", "disease"),
                    crop_col=ds_cfg.get("crop_col", "crop"),
                    label_id_col=ds_cfg.get("label_id_col", ""),
                )
                chunk_ds = SAGEDataset(
                    parquet_dir=str(train_dir),
                    processor=processor,
                    is_training=True,
                    label2id=label2id,
                    **{
                        "image_col": common["image_col"],
                        "disease_col": common["disease_col"],
                        "crop_col": common["crop_col"],
                        "label_id_col": common["label_id_col"],
                    },
                )
                chunk_loader = DataLoader(
                    chunk_ds,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=collate_fn,
                    num_workers=2,
                    pin_memory=True,
                )

                print(
                    f"[Chunk] {len(chunk_ds)} rows | "
                    f"{len(chunk_loader)} training batches"
                )
                process_loader(
                    chunk_loader,
                    total=len(chunk_loader),
                    desc=f"Ep {epoch} Ch {chunk_idx}/{len(chunks)}",
                )

                delete_shards(shard_group, train_dir)
                print(
                    f"[Disk Cleared] Flushed shards {shard_group}. "
                    f"Disk usage stays bounded.",
                    flush=True,
                )

                # Only save a chunk checkpoint when no partial accumulation is
                # pending. Otherwise resuming would lose those gradients.
                if accumulated_batches == 0:
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        ckpt_dir,
                        "latest_checkpoint",
                        {
                            "epoch": epoch,
                            "completed_chunk": chunk_idx,
                            "optimizer_steps": optimizer_steps_done,
                            "global_batches": step,
                            "best_f1": best_f1,
                        },
                    )

        else:
            process_loader(
                train_loader,
                total=steps_per_epoch,
                desc=f"Epoch {epoch}/{num_epochs}",
            )

        # Correctly perform a final optimizer step for a partial accumulation
        # group instead of silently discarding its gradients.
        if accumulated_batches:
            did_step = flush_partial_accumulation(
                model,
                optimizer,
                scheduler,
                train_cfg,
                accumulated_batches,
                grad_accum,
            )
            if did_step:
                optimizer_steps_done += 1
            accumulated_batches = 0

        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(step, 1)
        total_items = step * batch_size
        overall_throughput = total_items / max(elapsed, 0.001)

        print(
            f"\n[Epoch {epoch} Done] {step} steps | "
            f"{total_items} items in {elapsed:.1f}s "
            f"({overall_throughput:.1f} items/s) | "
            f"optimizer_steps={optimizer_steps_done} | "
            f"avg_loss={avg_loss:.4f}"
        )

        # Validation after each epoch.
        val_metrics = validate(
            model,
            processor,
            val_loss_loader,
            val_gen_loader,
            label2id,
            device,
            dtype,
            max_batches=int(
                train_cfg.get("validation_max_batches", 200)
            ),
            epoch=epoch,
        )

        macro_f1 = float(val_metrics["macro_f1"])
        print(
            f"[VAL Epoch {epoch}] "
            f"loss={val_metrics['val_loss']:.4f} | "
            f"acc={val_metrics['accuracy']:.4f} | "
            f"macro_f1={macro_f1:.4f} | "
            f"unknown={val_metrics['unknown_rate']:.2%}"
        )

        log_row = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "time_sec": elapsed,
            "optimizer_steps": optimizer_steps_done,
            **val_metrics,
        }
        logs.append(log_row)
        pd.DataFrame(logs).to_csv(
            log_dir / "training_log.csv",
            index=False,
        )

        # Best checkpoint: save model + optimizer + scheduler + state.
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                ckpt_dir,
                "best_checkpoint",
                {
                    "epoch": epoch,
                    "completed_chunk": len(chunks) if is_chunked else None,
                    "optimizer_steps": optimizer_steps_done,
                    "global_batches": step,
                    "best_f1": best_f1,
                },
            )
        else:
            no_improve += 1
            print(
                f"  [EarlyStopping] {no_improve}/{patience} "
                f"epochs without improvement."
            )
            if no_improve >= patience:
                print(f"  Stopping early at epoch {epoch}.")
                break

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            ckpt_dir,
            f"epoch_{epoch}",
            {
                "epoch": epoch,
                "optimizer_steps": optimizer_steps_done,
                "global_batches": step,
                "best_f1": best_f1,
            },
        )

    summary = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "checkpoint": str(ckpt_dir / "best_checkpoint"),
        "labels_path": str(labels_path),
        "dataset_mode": ds_mode,
        "validation_is_generation_safe": True,
    }
    save_json(output_dir / "run_summary.json", summary)

    print(
        f"\n{'=' * 60}\n"
        f"DONE | Best Epoch: {best_epoch} | "
        f"Best Macro F1: {best_f1:.4f}\n"
        f"{'=' * 60}"
    )


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    train(load_config(args.config))
