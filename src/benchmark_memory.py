"""
Memory Benchmarking Script for Optimizer and Visual Resolution Presets.
Runs isolated single-step forward + backward + optimizer step benchmarks
to accurately measure peak allocated and reserved CUDA memory.
"""

import os
import sys
import gc
import json
import argparse
import pandas as pd
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import set_seed
from src.preprocessing import get_processor
from src.dataset import SAGEDataset, sage_collate_fn
from src.model import get_qwen_qlora_model
from src.train import load_config


def run_benchmark(
    optimizer_type: str = "AdamW",
    min_pixels: int = 100352,
    max_pixels: int = 200704
):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    config = load_config()
    train_csv = PROJECT_ROOT / config["paths"]["train_csv"]

    processor = get_processor(min_pixels=min_pixels, max_pixels=max_pixels)
    dataset = SAGEDataset(str(train_csv), processor=processor, is_training=True)
    sample = sage_collate_fn([dataset[0]])

    lora_cfg = config.get("lora", {})
    model = get_qwen_qlora_model(
        model_name_or_path=config["model"]["name_or_path"],
        lora_r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        gradient_checkpointing=True,
        is_trainable=True
    )
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_type == "PagedAdamW8bit":
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW8bit(trainable_params, lr=2e-4)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=2e-4)

    input_ids = sample["input_ids"].to(device)
    attention_mask = sample["attention_mask"].to(device)
    pixel_values = sample["pixel_values"].to(device, dtype=torch.float16)
    image_grid_thw = sample["image_grid_thw"].to(device)
    labels = sample["labels"].to(device)

    # Forward
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels
        )
        loss = out.loss

    loss_val = loss.item()

    # Backward
    loss.backward()

    # Optimizer step
    step_success = False
    try:
        optimizer.step()
        optimizer.zero_grad()
        step_success = True
    except Exception as e:
        print(f"Optimizer step failed: {e}")

    peak_allocated_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)

    result = {
        "optimizer": optimizer_type,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "loss": round(loss_val, 4),
        "step_success": step_success,
        "peak_allocated_gb": round(peak_allocated_gb, 3),
        "peak_reserved_gb": round(peak_reserved_gb, 3),
        "patches": int(pixel_values.shape[0])
    }
    print(f"BENCHMARK_RESULT: {json.dumps(result)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["AdamW", "PagedAdamW8bit"])
    parser.add_argument("--min-pixels", type=int, default=100352)
    parser.add_argument("--max-pixels", type=int, default=200704)
    args = parser.parse_args()

    run_benchmark(
        optimizer_type=args.optimizer,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels
    )
