"""
Model implementation for SAGE crop disease diagnosis.
Loads Qwen2.5-VL-3B-Instruct in 4-bit NF4 with PEFT LoRA adapters.
"""

import sys
import torch
from pathlib import Path
from typing import Optional, Dict, Any

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)

def print_model_parameter_summary(model, lora_config=None, bnb_config=None):
    """
    Computes and prints total and trainable parameter counts and configurations.
    Fails if trainable parameter count is unexpectedly high.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
            
    pct_trainable = 100 * trainable_params / all_param if all_param > 0 else 0.0
    
    print("=" * 60)
    print("MODEL & PARAMETER SUMMARY")
    print("=" * 60)
    print(f"Total parameters:      {all_param:,}")
    print(f"Trainable parameters:  {trainable_params:,}")
    print(f"Trainable percentage:  {pct_trainable:.4f}%")
    print(f"Quantization:          4-bit NF4 (double quant: True, compute dtype: torch.float16)")
    if lora_config:
        print(f"LoRA Rank (r):         {lora_config.r}")
        print(f"LoRA Alpha:            {lora_config.lora_alpha}")
        print(f"LoRA Dropout:          {lora_config.lora_dropout}")
        print(f"Target Modules:        {lora_config.target_modules}")
    print(f"Device:                {next(model.parameters()).device}")
    print(f"Dtype:                 {next(model.parameters()).dtype}")
    print("=" * 60)
    
    # Fail-safe check
    if pct_trainable > 10.0:
        raise ValueError(
            f"SAFETY ALERT: Trainable parameters ({pct_trainable:.2f}%) are unexpectedly high! "
            f"Base model may not be frozen. Refusing to start training."
        )
    if trainable_params == 0 and getattr(model, "is_training", False):
        raise ValueError("SAFETY ALERT: Trainable parameter count is 0 for training mode!")
        
    return {
        "total_params": all_param,
        "trainable_params": trainable_params,
        "trainable_percentage": pct_trainable
    }

def get_qwen_qlora_model(
    model_name_or_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None,
    gradient_checkpointing: bool = True,
    device_map: str = "auto",
    is_trainable: bool = True
):
    """
    Loads Qwen2.5-VL-3B-Instruct in 4-bit NF4 and applies PEFT LoRA.
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        
    print(f"Configuring 4-bit NF4 Quantization for {model_name_or_path}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    print("Loading base Qwen2.5-VL model (offline/local cache)...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.float16,
        local_files_only=True
    )
    
    if is_trainable:
        # Prepare for kbit training
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=gradient_checkpointing
        )
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(base_model, lora_config)
        model.is_training = True
    else:
        model = base_model
        lora_config = None
        
    print_model_parameter_summary(model, lora_config=lora_config, bnb_config=bnb_config)
    return model

def load_trained_lora_model(
    adapter_path: str,
    base_model_name_or_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    device_map: str = "auto"
):
    """
    Loads base model in 4-bit and loads saved LoRA adapter checkpoint.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_name_or_path,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.float16,
        local_files_only=True
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    print(f"Loaded LoRA adapter from: {adapter_path}")
    return model

if __name__ == "__main__":
    model = get_qwen_qlora_model()
