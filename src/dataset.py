"""
PyTorch Dataset and DataLoader for SAGE Crop Disease Diagnosis.
Loads samples from CSV splits, formats Qwen2.5-VL conversations,
and builds masked training labels for causal language modeling.
"""

import os
import sys
import traceback

# IMPORTANT: pandas must be imported before torch on Windows to avoid
# MKL DLL conflicts that cause the C CSV parser to hard-crash.
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import Dict, Any, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prompts import build_conversation
from src.preprocessing import resolve_image_path


class SAGEDataset(Dataset):
    """
    Dataset class for SAGE crop disease images with Qwen2.5-VL conversational formatting.
    Reads from train.csv, val.csv, or test.csv.
    Builds masked labels for causal LM (teacher forcing on disease name only).
    """
    def __init__(
        self,
        csv_path: str,
        processor,
        is_training: bool = True,
        include_crop: bool = False
    ):
        self.csv_path = Path(csv_path)
        if not self.csv_path.is_absolute():
            self.csv_path = PROJECT_ROOT / self.csv_path

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {self.csv_path}")

        # engine='python' avoids the pandas C-parser MKL conflict on Windows
        self.df = pd.read_csv(self.csv_path, engine='python')
        self.processor = processor
        self.is_training = is_training
        self.include_crop = include_crop

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        image_id = int(row["image_id"])
        crop_name = str(row.get("crop", "")).strip()
        disease_label = str(row["disease"]).strip()
        label_id = int(row.get("label_id", -1))
        raw_path = str(row["image_path"]).strip()

        resolved_img_path = resolve_image_path(raw_path)

        try:
            with Image.open(resolved_img_path) as img:
                image = img.convert("RGB")
                image.load()
        except Exception as e:
            raise RuntimeError(f"Error loading image {resolved_img_path}: {e}")

        # Build prompt conversation (user only, no label)
        prompt_msgs = build_conversation(
            image=image,
            disease_label=None,
            include_crop=self.include_crop,
            crop_name=crop_name
        )
        prompt_text = self.processor.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True
        )

        if self.is_training:
            # Build full conversation (user + assistant disease label)
            full_msgs = build_conversation(
                image=image,
                disease_label=disease_label,
                include_crop=self.include_crop,
                crop_name=crop_name
            )
            full_text = self.processor.apply_chat_template(
                full_msgs,
                tokenize=False,
                add_generation_prompt=False
            )

            # Tokenize prompt-only to get the length of the prompt prefix
            prompt_tok = self.processor.tokenizer(
                prompt_text,
                add_special_tokens=False,
                return_tensors="pt"
            )
            prompt_len = prompt_tok["input_ids"].shape[1]

            # Process the full multimodal input
            inputs = self.processor(
                text=[full_text],
                images=[image],
                return_tensors="pt"
            )

            input_ids = inputs["input_ids"][0]
            attention_mask = inputs["attention_mask"][0]
            pixel_values = inputs["pixel_values"]
            image_grid_thw = inputs.get("image_grid_thw")

            # Count extra visual tokens inserted by the processor
            # The prompt_text uses <|image_pad|> placeholders replaced by actual visual tokens
            # So prompt_len from text tokenizer doesn't directly correspond to full multimodal input_ids length
            # Use full tokenizer to compute prompt portion correctly
            full_tok = self.processor.tokenizer(
                prompt_text,
                add_special_tokens=False,
                return_tensors="pt"
            )
            prompt_text_len = full_tok["input_ids"].shape[1]

            # Find actual prompt boundary in multimodal input_ids
            # We can detect boundary by finding where image tokens end + prompt text ends
            # Heuristic: total tokens = visual tokens + text tokens
            # visual tokens = input_ids.shape[0] - full_text_token_len
            full_text_tok = self.processor.tokenizer(
                full_text,
                add_special_tokens=False,
                return_tensors="pt"
            )
            full_text_len = full_text_tok["input_ids"].shape[1]
            total_len = input_ids.shape[0]
            visual_tokens = total_len - full_text_len
            # Prompt boundary in the multimodal sequence = visual_tokens + prompt_text_len
            prompt_boundary = max(0, visual_tokens + prompt_text_len)
            prompt_boundary = min(prompt_boundary, total_len)

            # Mask everything before the assistant response (-100 = ignored in loss)
            labels = input_ids.clone()
            labels[:prompt_boundary] = -100

            return {
                "image_id": image_id,
                "crop": crop_name,
                "disease": disease_label,
                "label_id": label_id,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "labels": labels
            }
        else:
            # Inference/evaluation mode
            inputs = self.processor(
                text=[prompt_text],
                images=[image],
                return_tensors="pt"
            )
            return {
                "image_id": image_id,
                "crop": crop_name,
                "disease": disease_label,
                "label_id": label_id,
                "prompt_text": prompt_text,
                "input_ids": inputs["input_ids"][0],
                "attention_mask": inputs["attention_mask"][0],
                "pixel_values": inputs["pixel_values"],
                "image_grid_thw": inputs.get("image_grid_thw")
            }


def sage_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for batch_size=1 (required for QLoRA VLM training)."""
    assert len(batch) == 1, "Batch size must be 1 for QLoRA VLM training."
    item = batch[0]
    collated = {
        "image_id": item["image_id"],
        "crop": item["crop"],
        "disease": item["disease"],
        "label_id": item["label_id"],
        "input_ids": item["input_ids"].unsqueeze(0),
        "attention_mask": item["attention_mask"].unsqueeze(0),
        "pixel_values": item["pixel_values"],
        "image_grid_thw": item["image_grid_thw"]
    }
    if "labels" in item:
        collated["labels"] = item["labels"].unsqueeze(0)
    if "prompt_text" in item:
        collated["prompt_text"] = item["prompt_text"]
    return collated


def get_dataloader(
    csv_path: str,
    processor,
    batch_size: int = 1,
    is_training: bool = True,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    dataset = SAGEDataset(csv_path=csv_path, processor=processor, is_training=is_training)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=sage_collate_fn
    )


if __name__ == "__main__":
    import traceback
    stage = "init"
    try:
        print("\n" + "="*60, flush=True)
        print("[DATASET TEST] Starting...", flush=True)
        print("="*60, flush=True)

        # Step 1: CSV path check (no read yet)
        stage = "CSV Path"
        csv_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
        print(f"\n[CSV] Path:   {csv_path}", flush=True)
        print(f"[CSV] Exists: {csv_path.exists()}", flush=True)

        # Step 2: Processor (loads transformers/torch)
        stage = "Processor"
        print("\n[PROCESSOR] Loading Qwen processor...", flush=True)
        from src.preprocessing import get_processor
        proc = get_processor()
        print(f"[OK] Processor loaded: {type(proc)}", flush=True)

        # Step 3: Dataset — SAGEDataset.__init__ reads the CSV internally
        stage = "Dataset"
        ds = SAGEDataset(str(csv_path), proc, is_training=True)
        # Reuse the already-loaded dataframe from ds.df
        row = ds.df.iloc[0]
        print(f"\n[OK] Dataset created. Rows: {len(ds)}", flush=True)
        print(f"[OK] Columns: {list(ds.df.columns)}", flush=True)

        # Step 4: Sample 0 info
        stage = "Sample 0"
        print(f"\n[SAMPLE 0]", flush=True)
        print(f"  image_id:   {row['image_id']}", flush=True)
        print(f"  disease:    {row['disease']}", flush=True)
        print(f"  label_id:   {row['label_id']}", flush=True)
        print(f"  image_path: {row['image_path']}", flush=True)
        resolved = resolve_image_path(str(row["image_path"]))
        print(f"  resolved:   {resolved}", flush=True)
        print(f"  exists:     {resolved.exists()}", flush=True)

        # Image decode
        stage = "Image decode"
        with Image.open(resolved) as img:
            img = img.convert("RGB")
            img.load()
        print(f"[OK] Image decoded. Size: {img.size}, Mode: {img.mode}", flush=True)

        # ds[0]
        stage = "ds[0]"
        s = ds[0]
        active_final = (s["labels"] != -100).sum().item()
        print(f"\n[OK] ds[0] returned successfully.", flush=True)
        print(f"  input_ids shape: {s['input_ids'].shape}", flush=True)
        print(f"  labels shape:    {s['labels'].shape}", flush=True)
        print(f"  pixel_values:    {s['pixel_values'].shape}", flush=True)
        print(f"  image_grid_thw:  {s['image_grid_thw']}", flush=True)
        print(f"  Active tgt tkns: {active_final}", flush=True)

        if active_final == 0:
            raise ValueError("Active target tokens is 0. Label masking is incorrect — stopping.")

        print(f"\n[OK] Loaded train dataset with {len(ds)} samples.")
        print(f"[OK] Sample 0: image_id={s['image_id']}, disease={s['disease']}")
        print(f"[OK] input_ids shape: {s['input_ids'].shape}, labels shape: {s['labels'].shape}")
        print(f"[OK] Active training target tokens: {active_final}")
        print("\n" + "="*60)
        print("[DATASET TEST] ALL CHECKS PASSED")
        print("="*60)

    except Exception as e:
        print(f"\n[FAIL] Stage: {stage}", flush=True)
        print(f"Exception: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)

