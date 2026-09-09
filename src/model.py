"""Qwen2.5-VL model loading and LoRA adapter management."""

import torch
from typing import Optional, List

from transformers import Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel


# Kaggle compatibility for environments where PEFT detects an incompatible
# torchao installation even though torchao is not needed for this LoRA setup.
try:
    import peft.import_utils

    _orig_torchao_check = getattr(
        peft.import_utils,
        "is_torchao_available",
        None,
    )

    if _orig_torchao_check is not None:
        def _safe_torchao_check():
            try:
                return _orig_torchao_check()
            except ImportError:
                return False

        peft.import_utils.is_torchao_available = _safe_torchao_check

    try:
        import peft.tuners.lora.torchao as _torchao_tuner
        if hasattr(_torchao_tuner, "is_torchao_available"):
            _torchao_tuner.is_torchao_available = _safe_torchao_check
    except Exception:
        pass
except Exception:
    pass


def _resolve_dtype(torch_dtype: str):
    name = str(torch_dtype).lower()
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float16", "fp16", "half"}:
        return torch.float16
    raise ValueError(
        f"Unsupported torch_dtype={torch_dtype!r}. Use float16 or bfloat16."
    )


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
    """Load Qwen2.5-VL and attach a LoRA adapter."""
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    dtype = _resolve_dtype(torch_dtype)

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
    is_trainable: bool = False,
):
    """Load a base Qwen model and a saved LoRA adapter."""
    dtype = _resolve_dtype(torch_dtype)

    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_name_or_path,
        torch_dtype=dtype,
        device_map="auto",
        local_files_only=local_files_only,
    )

    model = PeftModel.from_pretrained(
        base,
        adapter_path,
        is_trainable=is_trainable,
    )

    if merge_weights:
        print("Merging LoRA weights into base model...")
        model = model.merge_and_unload()

    model.eval()
    print(f"[OK] LoRA adapter loaded from: {adapter_path}")
    return model
