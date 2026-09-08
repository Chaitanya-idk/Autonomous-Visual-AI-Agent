"""
Model loading: Qwen2.5-VL with full-precision LoRA (no quantization).
Designed for GCP / Kaggle GPU instances (T4 16GB, L4 24GB, A100).
"""

import sys
import torch
from pathlib import Path
from typing import Optional, List

from transformers import Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel


def get_qwen_lora_model(
    model_name_or_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    torch_dtype: str = "bfloat16",
    gradient_checkpointing: bool = True,
    local_files_only: bool = False,
    is_trainable: bool = True,
):
    """
    Loads Qwen2.5-VL in full precision (bfloat16 or float16) and applies LoRA.
    No quantization — requires ~7 GB VRAM for 3B model base.
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    dtype = torch.bfloat16 if torch_dtype == "bfloat16" else torch.float16

    print(f"Loading {model_name_or_path} in {torch_dtype}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map="auto",
        local_files_only=local_files_only,
    )

    if is_trainable:
        if gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.config.use_cache = False

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        model.eval()

    return model


def load_trained_lora_model(
    adapter_path: str,
    base_model_name_or_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype: str = "bfloat16",
    local_files_only: bool = False,
    merge_weights: bool = False,
):
    """
    Loads base model and applies saved LoRA adapter.
    Set merge_weights=True to produce a self-contained model (no PEFT needed at inference).
    """
    dtype = torch.bfloat16 if torch_dtype == "bfloat16" else torch.float16

    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_name_or_path,
        torch_dtype=dtype,
        device_map="auto",
        local_files_only=local_files_only,
    )
    model = PeftModel.from_pretrained(base, adapter_path)

    if merge_weights:
        print("Merging LoRA weights into base model...")
        model = model.merge_and_unload()

    model.eval()
    print(f"[OK] LoRA adapter loaded from: {adapter_path}")
    return model
