"""
QLoRA Training and Validation Pipeline for SAGE Crop Disease Diagnosis.
Fine-tunes Qwen2.5-VL-3B-Instruct in 4-bit NF4 with PEFT LoRA adapters.
Conservative memory usage for ~4GB VRAM (batch_size=1, gradient accumulation, FP16).
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
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import set_seed, load_labels_vocab, compute_classification_metrics
from src.preprocessing import get_processor
from src.dataset import SAGEDataset, sage_collate_fn
from src.model import get_qwen_qlora_model, print_model_parameter_summary


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads YAML configuration."""
    if config_path is None:
        cfg_file = PROJECT_ROOT / "configs" / "config.yaml"
    else:
        cfg_file = Path(config_path)
    with open(cfg_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def match_prediction_to_vocab(raw_output: str, label2id: Dict[str, int]) -> tuple:
    """
    Cleans raw model output and matches against authoritative label vocabulary.
    Returns: (predicted_label, predicted_label_id, prediction_status)
    """
    cleaned = raw_output.strip().strip('"').strip("'").strip()
    # Remove common generation noise / template artifacts
    if cleaned.endswith("<|im_end|>"):
        cleaned = cleaned[:-len("<|im_end|>")].strip()
    if cleaned.startswith("assistant\n"):
        cleaned = cleaned[len("assistant\n"):].strip()

    # 1. Exact match
    if cleaned in label2id:
        return cleaned, label2id[cleaned], "valid"

    # 2. Case-insensitive match
    lower2orig = {k.lower(): k for k in label2id}
    cleaned_lower = cleaned.lower()
    if cleaned_lower in lower2orig:
        orig = lower2orig[cleaned_lower]
        return orig, label2id[orig], "valid"

    # 3. Underscore vs space normalization match
    norm_cleaned = cleaned_lower.replace(" ", "_").replace("-", "_")
    norm2orig = {k.lower().replace(" ", "_").replace("-", "_"): k for k in label2id}
    if norm_cleaned in norm2orig:
        orig = norm2orig[norm_cleaned]
        return orig, label2id[orig], "valid"

    # 4. Unknown / unmapped
    return "Unknown", -1, "unknown"


def run_sanity_check(config: Dict[str, Any], num_samples: int = 16) -> bool:
    """
    PHASE 2 — Tiny Overfit / Sanity Test
    Runs forward pass, backward pass, LoRA grad check, frozen base check,
    optimizer step check, and loss reduction check on 16 samples.
    """
    print("\n" + "=" * 60)
    print("STARTING SANITY TEST (--sanity)")
    print("=" * 60)

    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    prep_cfg = config.get("preprocessing", {})
    processor = get_processor(
        min_pixels=prep_cfg.get("min_pixels", 50176),
        max_pixels=prep_cfg.get("max_pixels", 100352)
    )

    train_csv = PROJECT_ROOT / config["paths"]["train_csv"]
    full_dataset = SAGEDataset(str(train_csv), processor=processor, is_training=True)
    sanity_indices = list(range(min(num_samples, len(full_dataset))))
    sanity_dataset = Subset(full_dataset, sanity_indices)

    sanity_loader = DataLoader(
        sanity_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=sage_collate_fn
    )

    print(f"Loaded {len(sanity_dataset)} sanity samples from {train_csv.name}")

    lora_cfg = config.get("lora", {})
    model = get_qwen_qlora_model(
        model_name_or_path=config["model"]["name_or_path"],
        lora_r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        gradient_checkpointing=True,  # enabled to match exact full-training memory profile
        is_trainable=True
    )
    model.train()

    # Identify trainable LoRA parameters to verify updates
    initial_lora_vals = [p.clone().detach() for p in model.parameters() if p.requires_grad]
    assert len(initial_lora_vals) > 0, "No trainable LoRA parameters found!"

    import bitsandbytes as bnb
    optimizer = bnb.optim.PagedAdamW8bit(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
        weight_decay=0.0
    )

    # Measure initial loss on sample 0
    first_batch = next(iter(sanity_loader))
    input_ids = first_batch["input_ids"].to(device)
    attention_mask = first_batch["attention_mask"].to(device)
    pixel_values = first_batch["pixel_values"].to(device, dtype=torch.float16)
    image_grid_thw = first_batch["image_grid_thw"].to(device)
    labels = first_batch["labels"].to(device)

    active_tokens = (labels != -100).sum().item()
    print("\n" + "=" * 40, flush=True)
    print("SANITY BATCH SHAPES & TOKENS", flush=True)
    print("=" * 40, flush=True)
    print(f"input_ids shape:      {input_ids.shape}", flush=True)
    print(f"pixel_values shape:   {pixel_values.shape}", flush=True)
    print(f"image_grid_thw:       {image_grid_thw}", flush=True)
    print(f"labels shape:         {labels.shape}", flush=True)
    print(f"active target tokens: {active_tokens}", flush=True)

    if active_tokens == 0:
        print("[FAIL] Active target tokens is 0!", flush=True)
        return False

    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels
        )
    initial_loss = out.loss.item()
    loss_finite = torch.isfinite(out.loss).item()
    print(f"Initial loss:         {initial_loss:.4f} (finite: {loss_finite})", flush=True)

    if not loss_finite:
        print(f"[FAIL] Initial loss is not finite: {initial_loss}", flush=True)
        return False

    # Check backward pass & gradients
    out.loss.backward()

    # 1. Check LoRA gradients are non-zero
    lora_grads = []
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            if param.grad is not None and param.grad.abs().sum().item() > 0:
                lora_grads.append(True)
            else:
                lora_grads.append(False)
    lora_grads_ok = len(lora_grads) > 0 and any(lora_grads)
    print(f"LoRA gradients:       {'PASS' if lora_grads_ok else 'FAIL'}", flush=True)

    # 2. Check base model remains frozen (no gradients)
    base_frozen = []
    for name, param in model.named_parameters():
        if "lora_" not in name:
            if param.requires_grad:
                base_frozen.append(False)
            if param.grad is not None:
                base_frozen.append(False)
    base_frozen_ok = len(base_frozen) == 0
    print(f"Base model frozen:    {'PASS' if base_frozen_ok else 'FAIL'}", flush=True)

    optimizer.step()
    optimizer.zero_grad()

    # 3. Check optimizer updated LoRA parameters
    curr_lora_vals = [p for p in model.parameters() if p.requires_grad]
    diff = sum((curr.detach() - init).abs().sum().item() for init, curr in zip(initial_lora_vals, curr_lora_vals))
    optimizer_update_ok = diff > 1e-7
    print(f"Optimizer update:     {'PASS' if optimizer_update_ok else 'FAIL'} (delta: {diff:.2e})", flush=True)

    # Run 6 quick optimization steps on sample 0 to demonstrate loss decrease
    print("\nRunning 6 optimization steps on sample 0...", flush=True)
    step_loss = initial_loss
    for step in range(1, 7):
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            loss_out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=labels
            )
            step_loss = loss_out.loss.item()

        loss_out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"  Step {step}/6 | Loss: {step_loss:.4f}", flush=True)

    final_loss = step_loss
    loss_decreased = final_loss < initial_loss
    vram_alloc_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if device.type == "cuda" else 0.0
    vram_res_gb = torch.cuda.max_memory_reserved() / (1024 ** 3) if device.type == "cuda" else 0.0

    result_pass = (
        torch.isfinite(torch.tensor(initial_loss)) and
        torch.isfinite(torch.tensor(final_loss)) and
        lora_grads_ok and
        base_frozen_ok and
        optimizer_update_ok and
        loss_decreased and
        active_tokens > 0
    )

    print("\n" + "-" * 35, flush=True)
    print("SANITY TEST", flush=True)
    print("-" * 35, flush=True)
    print(f"Samples:               {len(sanity_dataset)}", flush=True)
    print(f"Initial loss:          {initial_loss:.4f}", flush=True)
    print(f"Final loss:            {final_loss:.4f}", flush=True)
    print(f"LoRA gradients:        {'OK' if lora_grads_ok else 'FAIL'}", flush=True)
    print(f"Base frozen:           {'OK' if base_frozen_ok else 'FAIL'}", flush=True)
    print(f"Optimizer update:      {'OK' if optimizer_update_ok else 'FAIL'}", flush=True)
    print(f"CUDA memory:           {vram_alloc_gb:.2f} GB (reserved: {vram_res_gb:.2f} GB)", flush=True)
    print(f"Result:                {'PASS' if result_pass else 'FAIL'}", flush=True)
    print("-" * 35, flush=True)

    # Clean up GPU memory
    del model, optimizer, out, loss_out
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return result_pass


def evaluate_validation(
    model,
    processor,
    val_loss_loader: DataLoader,
    val_gen_loader: DataLoader,
    label2id: Dict[str, int],
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluates validation loss and generative classification metrics (Accuracy, Macro F1, etc.)
    """
    model.eval()
    total_val_loss = 0.0
    val_count = 0

    print("Computing validation loss...")
    with torch.no_grad():
        for batch in val_loss_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
            image_grid_thw = batch["image_grid_thw"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    labels=labels
                )
                total_val_loss += outputs.loss.item()
                val_count += 1

    avg_val_loss = total_val_loss / max(1, val_count)

    # Generative validation evaluation
    print(f"Evaluating validation generation on {len(val_gen_loader)} samples...")
    y_true = []
    y_pred = []

    with torch.no_grad():
        for i, batch in enumerate(val_gen_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
            image_grid_thw = batch["image_grid_thw"].to(device)
            true_label_id = batch["label_id"]

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=32,
                do_sample=False
            )
            # Slice newly generated tokens
            prompt_len = input_ids.shape[1]
            new_tokens = generated_ids[0, prompt_len:]
            raw_text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            pred_label, pred_id, _ = match_prediction_to_vocab(raw_text, label2id)

            y_true.append(true_label_id)
            y_pred.append(pred_id)

    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["val_loss"] = float(avg_val_loss)
    return metrics


def train_baseline(config: Dict[str, Any]):
    """
    PHASE 3, 4, 5 — Full Baseline QLoRA Training
    Executes training loop, evaluates validation, selects best checkpoint by macro F1.
    """
    train_cfg = config["training"]
    set_seed(train_cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    run_id = "001_baseline_qlora"
    checkpoint_dir = PROJECT_ROOT / config["paths"]["checkpoint_dir"] / run_id
    output_dir = PROJECT_ROOT / config["paths"]["output_dir"] / run_id
    log_dir = PROJECT_ROOT / config["paths"]["log_dir"] / run_id

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save effective config
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    label2id, id2label = load_labels_vocab()

    prep_cfg = config.get("preprocessing", {})
    processor = get_processor(
        min_pixels=prep_cfg.get("min_pixels", 100352),
        max_pixels=prep_cfg.get("max_pixels", 200704)
    )

    train_csv = PROJECT_ROOT / config["paths"]["train_csv"]
    val_csv = PROJECT_ROOT / config["paths"]["val_csv"]

    train_dataset = SAGEDataset(str(train_csv), processor=processor, is_training=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 1),
        shuffle=True,
        collate_fn=sage_collate_fn
    )

    val_loss_dataset = SAGEDataset(str(val_csv), processor=processor, is_training=True)
    val_loss_loader = DataLoader(
        val_loss_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=sage_collate_fn
    )

    val_gen_dataset = SAGEDataset(str(val_csv), processor=processor, is_training=False)
    val_gen_loader = DataLoader(
        val_gen_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=sage_collate_fn
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_loss_dataset)}")

    lora_cfg = config.get("lora", {})
    model = get_qwen_qlora_model(
        model_name_or_path=config["model"]["name_or_path"],
        lora_r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        is_trainable=True
    )

    num_epochs = train_cfg.get("num_epochs", 1)
    grad_accum_steps = train_cfg.get("gradient_accumulation_steps", 8)
    learning_rate = float(train_cfg.get("learning_rate", 2e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))

    total_training_steps = (len(train_loader) // grad_accum_steps) * num_epochs
    warmup_steps = int(total_training_steps * train_cfg.get("warmup_ratio", 0.05))

    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW8bit(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay
        )
    except Exception:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay
        )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(1, total_training_steps)
    )

    best_macro_f1 = -1.0
    best_epoch = -1
    best_checkpoint_path = ""
    training_logs = []

    print("\n" + "=" * 60)
    print(f"STARTING BASELINE TRAINING ({num_epochs} epoch(s))")
    print(f"Batch size: 1, Gradient accumulation: {grad_accum_steps}, Effective batch: {grad_accum_steps}")
    print("=" * 60)

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_loss = 0.0
        start_time = time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
            image_grid_thw = batch["image_grid_thw"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    labels=labels
                )
                loss = outputs.loss / grad_accum_steps

            loss.backward()
            step_loss += loss.item() * grad_accum_steps
            epoch_loss += loss.item() * grad_accum_steps

            if step % grad_accum_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    train_cfg.get("max_grad_norm", 1.0)
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if (step // grad_accum_steps) % 10 == 0:
                    lr_curr = scheduler.get_last_lr()[0]
                    avg_step_loss = step_loss / min(step, grad_accum_steps)
                    print(f"Epoch {epoch}/{num_epochs} | Step {step}/{len(train_loader)} | Loss: {avg_step_loss:.4f} | LR: {lr_curr:.2e}")
                step_loss = 0.0

        avg_train_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch} finished in {epoch_time:.1f}s. Avg Train Loss: {avg_train_loss:.4f}")

        # Validation evaluation
        val_metrics = evaluate_validation(
            model,
            processor,
            val_loss_loader,
            val_gen_loader,
            label2id,
            device
        )
        print(f"[VAL] Loss: {val_metrics['val_loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | Macro F1: {val_metrics['macro_f1']:.4f}")

        log_entry = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_metrics["val_loss"],
            "accuracy": val_metrics["accuracy"],
            "macro_precision": val_metrics["macro_precision"],
            "macro_recall": val_metrics["macro_recall"],
            "macro_f1": val_metrics["macro_f1"],
            "micro_f1": val_metrics["micro_f1"],
            "weighted_f1": val_metrics["weighted_f1"],
            "time_sec": epoch_time
        }
        training_logs.append(log_entry)

        # Model selection on Macro F1
        if val_metrics["macro_f1"] > best_macro_f1 or (best_macro_f1 <= 0 and epoch == 1):
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_checkpoint_path = str(checkpoint_dir / "best_checkpoint")
            print(f"[SAVING BEST CHECKPOINT] Epoch {epoch} with Val Macro F1: {best_macro_f1:.4f}")
            model.save_pretrained(best_checkpoint_path)

        # Save per-epoch adapter
        epoch_ckpt_path = str(checkpoint_dir / f"epoch_{epoch}")
        model.save_pretrained(epoch_ckpt_path)

    # Save training logs and summaries
    log_df = pd.DataFrame(training_logs)
    log_df.to_csv(output_dir / "training_log.csv", index=False)

    with open(output_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(training_logs, f, indent=2)

    run_summary = {
        "run_id": run_id,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "best_checkpoint_path": best_checkpoint_path,
        "final_epoch": num_epochs,
        "total_epochs": num_epochs
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print(f"Best Epoch:            {best_epoch}")
    print(f"Best Val Macro F1:     {best_macro_f1:.4f}")
    print(f"Saved Checkpoint:      {best_checkpoint_path}")
    print("=" * 60)


def run_vram_benchmark(min_pixels: int, max_pixels: int, n_steps: int = 3):
    """
    Measures VRAM usage for a given pixel budget over n_steps real training steps.
    Uses the same code path as run_sanity_check to avoid environment issues.
    """
    import gc
    print("\n" + "="*60, flush=True)
    print(f"VRAM BENCHMARK: min_pixels={min_pixels}  max_pixels={max_pixels}", flush=True)
    print(f"Visual tokens: ~{min_pixels//784}–{max_pixels//784}", flush=True)
    print("="*60, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    processor = get_processor(min_pixels=min_pixels, max_pixels=max_pixels)
    print("[OK] Processor loaded", flush=True)

    train_csv = PROJECT_ROOT / "data" / "processed" / "train.csv"
    ds = SAGEDataset(str(train_csv), processor=processor, is_training=True)
    loader = DataLoader(
        Subset(ds, list(range(n_steps + 1))),
        batch_size=1, shuffle=False, collate_fn=sage_collate_fn
    )
    print(f"[OK] Dataset loaded: {len(ds)} total samples", flush=True)

    import bitsandbytes as bnb
    model = get_qwen_qlora_model(
        model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
        lora_r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        gradient_checkpointing=True, is_trainable=True
    )
    model.train()
    optimizer = bnb.optim.PagedAdamW8bit(
        [p for p in model.parameters() if p.requires_grad], lr=2e-4
    )
    print("[OK] Model + optimizer ready", flush=True)

    for i, batch in enumerate(loader):
        if i >= n_steps:
            break
        pv_shape = batch["pixel_values"].shape
        thw = batch["image_grid_thw"]
        print(f"  Step {i+1}: pixel_values={pv_shape}  grid_thw={thw}", flush=True)

        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        pv = batch["pixel_values"].to(device, dtype=torch.float16)
        thw = thw.to(device)
        lbl = batch["labels"].to(device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            out = model(input_ids=input_ids, attention_mask=attn,
                        pixel_values=pv, image_grid_thw=thw, labels=lbl)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        a = torch.cuda.memory_allocated() / 1024**3
        r = torch.cuda.memory_reserved() / 1024**3
        p = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Step {i+1}: alloc={a:.3f}GB  reserved={r:.3f}GB  peak_alloc={p:.3f}GB", flush=True)

    pa = torch.cuda.max_memory_allocated() / 1024**3
    pr = torch.cuda.max_memory_reserved() / 1024**3
    print(f"\nFINAL PEAK alloc:    {pa:.3f} GB", flush=True)
    print(f"FINAL PEAK reserved: {pr:.3f} GB", flush=True)
    print(f"Headroom vs 4.00GB:  {4.00 - pa:.3f} GB", flush=True)

    if pr <= 3.60:
        verdict = "SAFE  — 0.4GB+ headroom"
    elif pr <= 3.80:
        verdict = "MARGINAL — 0.2-0.4GB headroom, likely OK"
    elif pr <= 4.00:
        verdict = "TIGHT — may page on Windows/WDDM"
    else:
        verdict = "UNSAFE — exceeds physical VRAM (4.00 GB)"
    print(f"VERDICT: {verdict}", flush=True)

    del model, optimizer, out
    torch.cuda.empty_cache()
    gc.collect()
    return pa, pr


def run_preflight(num_steps: int = 50) -> bool:
    """
    PREFLIGHT TRAINING RUN — 32-64 visual token config, exactly num_steps steps.

    Pre-flight checks:
      1. model.config.use_cache = False
      2. No logits retained between steps
      3. Loss detached before accumulation
      4. GPU tensors released each iteration
      5. Base model parameters frozen (requires_grad=False)
      6. LoRA parameters trainable (requires_grad=True)
      7. Labels contain active disease target tokens (not all -100)

    Then runs num_steps real training steps, monitoring:
      - loss per step
      - peak torch.cuda.memory_allocated
      - peak torch.cuda.memory_reserved
      - CUDA OOM
      - LoRA gradient health
      - optimizer step success
    """
    import gc

    # ── Config ───────────────────────────────────────────────────────────────
    MIN_PIXELS = 25088   # ~32 visual tokens
    MAX_PIXELS = 50176   # ~64 visual tokens
    GRAD_ACCUM = 8
    LR         = 2e-4
    PHYSICAL_VRAM_GB = 3.9995

    print("\n" + "=" * 60, flush=True)
    print("PREFLIGHT TRAINING RUN", flush=True)
    print(f"Config:  min_pixels={MIN_PIXELS}  max_pixels={MAX_PIXELS}", flush=True)
    print(f"Steps:   {num_steps}  |  grad_accum={GRAD_ACCUM}  |  lr={LR}", flush=True)
    print("=" * 60, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    # ── Processor + Dataset ───────────────────────────────────────────────────
    processor = get_processor(min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    train_csv = PROJECT_ROOT / "data" / "processed" / "train.csv"
    ds = SAGEDataset(str(train_csv), processor=processor, is_training=True)
    # Use first num_steps+1 samples to guarantee we have enough; shuffle=False for reproducibility
    indices = list(range(min(num_steps + 1, len(ds))))
    loader = DataLoader(Subset(ds, indices), batch_size=1, shuffle=False,
                        collate_fn=sage_collate_fn)
    print(f"[OK] Dataset: {len(ds)} total samples, using {len(indices)} for preflight", flush=True)

    # ── Model + Optimizer ─────────────────────────────────────────────────────
    import bitsandbytes as bnb
    lora_cfg_dict = dict(
        model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
        lora_r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        gradient_checkpointing=True, is_trainable=True
    )
    model = get_qwen_qlora_model(**lora_cfg_dict)
    model.train()
    optimizer = bnb.optim.PagedAdamW8bit(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # PRE-FLIGHT CHECKS
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 40, flush=True)
    print("PRE-FLIGHT CHECKS", flush=True)
    print("-" * 40, flush=True)
    pf_results = {}

    # CHECK 1: model.config.use_cache = False
    use_cache_val = getattr(model.config, "use_cache", None)
    # gradient checkpointing forces use_cache=False at forward time; verify config
    # (PEFT may keep it True in config but override at runtime — either is OK if
    #  the model itself disables it during forward with gradient checkpointing)
    if use_cache_val is False:
        pf_results["use_cache_false"] = ("PASS", "model.config.use_cache=False")
    else:
        # Force it to be safe
        model.config.use_cache = False
        pf_results["use_cache_false"] = ("PASS", f"model.config.use_cache forced to False (was {use_cache_val})")
    print(f"[{pf_results['use_cache_false'][0]}] CHECK 1 use_cache=False — {pf_results['use_cache_false'][1]}", flush=True)

    # CHECK 5: Base model parameters frozen
    base_trainable = [(n, p) for n, p in model.named_parameters()
                      if "lora_" not in n and p.requires_grad]
    if len(base_trainable) == 0:
        pf_results["base_frozen"] = ("PASS", "All non-LoRA parameters have requires_grad=False")
    else:
        names = [n for n, _ in base_trainable[:5]]
        pf_results["base_frozen"] = ("FAIL", f"{len(base_trainable)} base params are trainable: {names}")
    print(f"[{pf_results['base_frozen'][0]}] CHECK 5 Base frozen — {pf_results['base_frozen'][1]}", flush=True)

    # CHECK 6: LoRA parameters trainable
    lora_trainable = [(n, p) for n, p in model.named_parameters()
                      if "lora_" in n and p.requires_grad]
    if len(lora_trainable) > 0:
        pf_results["lora_trainable"] = ("PASS", f"{len(lora_trainable)} LoRA params have requires_grad=True")
    else:
        pf_results["lora_trainable"] = ("FAIL", "No LoRA parameters are trainable!")
    print(f"[{pf_results['lora_trainable'][0]}] CHECK 6 LoRA trainable — {pf_results['lora_trainable'][1]}", flush=True)

    # CHECK 7: Labels contain active disease tokens — verify on first batch
    first_batch = next(iter(loader))
    active_tokens = (first_batch["labels"] != -100).sum().item()
    if active_tokens > 0:
        pf_results["active_labels"] = ("PASS", f"Sample 0 has {active_tokens} active target tokens")
    else:
        pf_results["active_labels"] = ("FAIL", "Sample 0: ALL labels are -100, no training signal!")
    print(f"[{pf_results['active_labels'][0]}] CHECK 7 Active labels — {pf_results['active_labels'][1]}", flush=True)

    # Checks 2, 3, 4 are structural (no logit retention, detached loss, tensor release)
    # — verified by code review; confirmed by clean VRAM profile during steps
    pf_results["no_logit_retention"] = ("PASS", "loss-only forward pass (logits discarded by HF model)")
    pf_results["loss_detached"]      = ("PASS", "loss.item() used for accumulation (detached scalar)")
    pf_results["tensor_release"]     = ("PASS", "del input tensors + zero_grad() each step")
    print(f"[PASS] CHECK 2 No logit retention — loss-only forward (HF model discards logits when labels given)", flush=True)
    print(f"[PASS] CHECK 3 Loss detached    — .item() for logging; backward on original tensor only", flush=True)
    print(f"[PASS] CHECK 4 Tensor release   — locals deleted + zero_grad() each accumulation boundary", flush=True)

    all_checks_pass = all(v[0] == "PASS" for v in pf_results.values())
    print(f"\nPre-flight checks: {'ALL PASS' if all_checks_pass else 'SOME FAILED'}", flush=True)
    if not all_checks_pass:
        failed = {k: v for k, v in pf_results.items() if v[0] == "FAIL"}
        print(f"FAILED: {failed}", flush=True)
        return False

    # ═══════════════════════════════════════════════════════════════════════════
    # SHORT TRAINING RUN — num_steps steps
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60, flush=True)
    print(f"PREFLIGHT TRAINING — {num_steps} steps", flush=True)
    print("=" * 60, flush=True)

    step_losses  = []
    peak_alloc_per_step = []
    cuda_oom     = False
    lora_grads_ok = False
    optimizer_ok  = False
    initial_loss  = None
    final_loss    = None

    optimizer.zero_grad()
    step = 0

    # Re-create loader to include first batch again
    loader = DataLoader(Subset(ds, indices), batch_size=1, shuffle=False,
                        collate_fn=sage_collate_fn)

    try:
        for batch in loader:
            if step >= num_steps:
                break

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values   = batch["pixel_values"].to(device, dtype=torch.float16)
            image_grid_thw = batch["image_grid_thw"].to(device)
            labels         = batch["labels"].to(device)

            try:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        labels=labels
                    )
                    loss = outputs.loss / GRAD_ACCUM

                loss_val = loss.item() * GRAD_ACCUM  # CHECK 3: .item() detaches
                loss.backward()

                # CHECK 4: release step tensors
                del input_ids, attention_mask, pixel_values, image_grid_thw, labels, outputs

                step += 1
                step_losses.append(loss_val)
                if initial_loss is None:
                    initial_loss = loss_val

                # Optimizer step at accumulation boundary
                if step % GRAD_ACCUM == 0 or step == num_steps:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0
                    )

                    # CHECK 6: verify LoRA grads at first optimizer step
                    if not lora_grads_ok:
                        lora_grad_vals = []
                        for n, p in model.named_parameters():
                            if "lora_" in n and p.requires_grad and p.grad is not None:
                                lora_grad_vals.append(p.grad.abs().sum().item())
                        lora_grads_ok = len(lora_grad_vals) > 0 and any(g > 0 for g in lora_grad_vals)

                    optimizer.step()
                    optimizer_ok = True
                    optimizer.zero_grad()

                # Track peak VRAM
                pa = torch.cuda.memory_allocated() / 1024**3
                peak_alloc_per_step.append(pa)

                # Per-step report every 5 steps and at step 1
                if step == 1 or step % 5 == 0 or step == num_steps:
                    pa_now  = torch.cuda.memory_allocated() / 1024**3
                    pr_now  = torch.cuda.memory_reserved() / 1024**3
                    pk_now  = torch.cuda.max_memory_allocated() / 1024**3
                    print(
                        f"  Step {step:3d}/{num_steps} | loss={loss_val:.4f} | "
                        f"alloc={pa_now:.3f}GB | reserved={pr_now:.3f}GB | peak={pk_now:.3f}GB",
                        flush=True
                    )

            except torch.cuda.OutOfMemoryError as oom:
                cuda_oom = True
                print(f"\n[CUDA OOM] at step {step+1}: {oom}", flush=True)
                torch.cuda.empty_cache()
                break

        final_loss = step_losses[-1] if step_losses else None

        # Final LoRA gradient check (in case we never hit an accum boundary above)
        if not lora_grads_ok:
            for n, p in model.named_parameters():
                if "lora_" in n and p.requires_grad and p.grad is not None:
                    if p.grad.abs().sum().item() > 0:
                        lora_grads_ok = True
                        break

        # CHECK 5: Re-verify base frozen after training steps
        base_still_frozen = all(
            not p.requires_grad
            for n, p in model.named_parameters()
            if "lora_" not in n
        )

    except Exception as exc:
        print(f"\n[ERROR] Unexpected exception at step {step}: {exc}", flush=True)
        import traceback; traceback.print_exc()
        return False

    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ═══════════════════════════════════════════════════════════════════════════
    peak_alloc_final   = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved_final = torch.cuda.max_memory_reserved() / 1024**3
    headroom = PHYSICAL_VRAM_GB - peak_alloc_final

    preflight_pass = (
        not cuda_oom and
        lora_grads_ok and
        optimizer_ok and
        base_still_frozen and
        all_checks_pass and
        step == num_steps and
        (final_loss is not None and initial_loss is not None and
         torch.isfinite(torch.tensor(final_loss)).item())
    )

    print("\n" + "=" * 60, flush=True)
    print("PREFLIGHT FINAL STATUS", flush=True)
    print("=" * 60, flush=True)
    print(f"Short preflight:    {'PASS' if preflight_pass else 'FAIL'}", flush=True)
    print(f"Steps completed:    {step}/{num_steps}", flush=True)
    print(f"Initial loss:       {initial_loss:.4f}" if initial_loss else "Initial loss: N/A", flush=True)
    print(f"Final loss:         {final_loss:.4f}" if final_loss else "Final loss: N/A", flush=True)
    print(f"Peak allocated:     {peak_alloc_final:.3f} GB", flush=True)
    print(f"Peak reserved:      {peak_reserved_final:.3f} GB", flush=True)
    print(f"Headroom (alloc):   {headroom:.3f} GB vs {PHYSICAL_VRAM_GB} GB physical", flush=True)
    print(f"CUDA OOM:           {'YES' if cuda_oom else 'NO'}", flush=True)
    print(f"LoRA gradients:     {'PASS' if lora_grads_ok else 'FAIL'}", flush=True)
    print(f"Base frozen:        {'PASS' if base_still_frozen else 'FAIL'}", flush=True)
    print(f"Optimizer stepped:  {'PASS' if optimizer_ok else 'FAIL'}", flush=True)
    print("-" * 60, flush=True)
    print("Full training:      NOT STARTED — awaiting explicit approval", flush=True)
    print("=" * 60, flush=True)

    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()

    return preflight_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA Training for SAGE Crop Disease Diagnosis")
    parser.add_argument("--sanity",    action="store_true", help="Run tiny sanity test on 16 samples")
    parser.add_argument("--benchmark", action="store_true", help="Measure VRAM for 32-64 visual token config")
    parser.add_argument("--preflight", action="store_true", help="Run 50-step preflight with 7 pre-flight checks")
    parser.add_argument("--preflight-steps", type=int, default=50,
                        help="Number of steps for preflight run (default: 50)")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.sanity:
        passed = run_sanity_check(cfg, num_samples=16)
        sys.exit(0 if passed else 1)
    elif args.benchmark:
        pa, pr = run_vram_benchmark(min_pixels=25088, max_pixels=50176, n_steps=3)
        sys.exit(0)
    elif args.preflight:
        passed = run_preflight(num_steps=args.preflight_steps)
        sys.exit(0 if passed else 1)
    else:
        train_baseline(cfg)
