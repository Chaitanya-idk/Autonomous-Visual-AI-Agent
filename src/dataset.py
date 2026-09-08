"""
PyTorch Dataset for SAGE Crop Disease — reads directly from parquet files
or HuggingFace Hub. No image extraction step required.
"""

import io
import os
import sys
import glob
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prompts import build_conversation


def _decode_image(raw) -> Image.Image:
    """Decode image from PIL Image, bytes, or HuggingFace dict."""
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    if isinstance(raw, bytes):
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if isinstance(raw, dict) and "bytes" in raw:
        return Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
    raise ValueError(f"Cannot decode image of type: {type(raw)}")


class SAGEDataset(Dataset):
    """
    SAGE dataset loader.
    Supports:
      - Local parquet files (parquet_dir/*.parquet)
      - HuggingFace Hub streaming (hf_dataset passed directly)
    """
    def __init__(
        self,
        processor,
        parquet_dir: Optional[str] = None,
        hf_dataset=None,
        is_training: bool = True,
        image_col: str = "image",
        disease_col: str = "disease",
        crop_col: str = "crop",
        label_id_col: str = "label_id",
        label2id: Optional[Dict[str, int]] = None,
        split: str = "train",
    ):
        self.processor = processor
        self.is_training = is_training
        self.image_col = image_col
        self.disease_col = disease_col
        self.crop_col = crop_col if crop_col else None
        self.label_id_col = label_id_col if label_id_col else None

        if hf_dataset is not None:
            # HuggingFace datasets.Dataset object
            self._data = hf_dataset
            self._use_hf = True
        elif parquet_dir:
            # Load from local parquet files
            files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
            if not files:
                raise FileNotFoundError(f"No parquet files found in: {parquet_dir}")
            print(f"[Dataset] Loading {len(files)} parquet files from {parquet_dir}...")
            self._df = pd.concat(
                [pd.read_parquet(f, engine="pyarrow") for f in files],
                ignore_index=True
            )
            print(f"[Dataset] Loaded {len(self._df)} rows. Columns: {list(self._df.columns)}")
            self._use_hf = False
        else:
            raise ValueError("Provide either parquet_dir or hf_dataset.")

        # Build label2id mapping
        if label2id is not None:
            self.label2id = label2id
        else:
            self.label2id = self._build_label2id()

    def _build_label2id(self) -> Dict[str, int]:
        if self._use_hf:
            labels = sorted(set(self._data[self.disease_col]))
        else:
            labels = sorted(self._df[self.disease_col].dropna().unique().tolist())
        return {label: idx for idx, label in enumerate(labels)}

    def __len__(self) -> int:
        return len(self._data) if self._use_hf else len(self._df)

    def _get_row(self, idx: int) -> Dict[str, Any]:
        if self._use_hf:
            return dict(self._data[idx])
        return self._df.iloc[idx].to_dict()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self._get_row(idx)

        image = _decode_image(row[self.image_col])
        disease_label = str(row[self.disease_col]).strip()
        crop_name = str(row[self.crop_col]).strip() if self.crop_col and self.crop_col in row else ""
        label_id = int(row[self.label_id_col]) if self.label_id_col and self.label_id_col in row \
                   else self.label2id.get(disease_label, -1)

        # Build prompt (no label) for inference / prompt boundary detection
        prompt_msgs = build_conversation(
            image=image,
            disease_label=None,
            include_crop=bool(crop_name),
            crop_name=crop_name
        )
        prompt_text = self.processor.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )

        if self.is_training:
            full_msgs = build_conversation(
                image=image,
                disease_label=disease_label,
                include_crop=bool(crop_name),
                crop_name=crop_name
            )
            full_text = self.processor.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False
            )

            inputs = self.processor(
                text=[full_text], images=[image], return_tensors="pt"
            )
            input_ids      = inputs["input_ids"][0]
            attention_mask = inputs["attention_mask"][0]
            pixel_values   = inputs["pixel_values"]
            image_grid_thw = inputs.get("image_grid_thw")

            # Compute prompt boundary to mask non-target tokens
            full_text_tok = self.processor.tokenizer(
                full_text, add_special_tokens=False, return_tensors="pt"
            )
            prompt_tok = self.processor.tokenizer(
                prompt_text, add_special_tokens=False, return_tensors="pt"
            )
            visual_tokens  = input_ids.shape[0] - full_text_tok["input_ids"].shape[1]
            prompt_boundary = max(0, min(visual_tokens + prompt_tok["input_ids"].shape[1], input_ids.shape[0]))

            labels = input_ids.clone()
            labels[:prompt_boundary] = -100

            return {
                "disease": disease_label,
                "crop": crop_name,
                "label_id": label_id,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "labels": labels,
            }
        else:
            inputs = self.processor(
                text=[prompt_text], images=[image], return_tensors="pt"
            )
            return {
                "disease": disease_label,
                "crop": crop_name,
                "label_id": label_id,
                "input_ids": inputs["input_ids"][0],
                "attention_mask": inputs["attention_mask"][0],
                "pixel_values": inputs["pixel_values"],
                "image_grid_thw": inputs.get("image_grid_thw"),
                "prompt_text": prompt_text,
            }


def sage_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate with left-padding to support batch_size > 1.
    All sequences padded to the longest in the batch.
    """
    # Non-tensor metadata
    result = {
        "disease":  [b["disease"]  for b in batch],
        "crop":     [b["crop"]     for b in batch],
        "label_id": torch.tensor([b["label_id"] for b in batch], dtype=torch.long),
    }

    # Left-pad input_ids and attention_mask
    pad_id   = 0  # Qwen tokenizer pad token id
    max_len  = max(b["input_ids"].shape[0] for b in batch)

    input_ids_list  = []
    attn_mask_list  = []
    for b in batch:
        seq = b["input_ids"]
        pad_len = max_len - seq.shape[0]
        input_ids_list.append(
            torch.cat([torch.full((pad_len,), pad_id, dtype=seq.dtype), seq])
        )
        attn = b["attention_mask"]
        attn_mask_list.append(
            torch.cat([torch.zeros(pad_len, dtype=attn.dtype), attn])
        )

    result["input_ids"]       = torch.stack(input_ids_list)
    result["attention_mask"]  = torch.stack(attn_mask_list)

    # pixel_values: concatenate along patch dim (dim 0), grid_thw stacked
    result["pixel_values"]   = torch.cat([b["pixel_values"] for b in batch], dim=0)
    result["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in batch], dim=0)

    if "labels" in batch[0]:
        labels_list = []
        for b in batch:
            lbl = b["labels"]
            pad_len = max_len - lbl.shape[0]
            labels_list.append(
                torch.cat([torch.full((pad_len,), -100, dtype=lbl.dtype), lbl])
            )
        result["labels"] = torch.stack(labels_list)

    if "prompt_text" in batch[0]:
        result["prompt_text"] = [b["prompt_text"] for b in batch]

    return result
