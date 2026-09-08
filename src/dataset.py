"""
PyTorch Dataset for SAGE Crop Disease.
Supports:
  - HuggingFace Hub streaming (no download, reads live over network)
  - Local parquet files (glob data/parquet/*.parquet)
Images are decoded from bytes at runtime — no extraction step needed.
"""

import io
import os
import glob
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, IterableDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prompts import build_conversation


# ── Image decoding ────────────────────────────────────────────────────────────

def _decode_image(raw) -> Image.Image:
    """Decode image from PIL Image, raw bytes, or HF dict {bytes, path}."""
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    if isinstance(raw, bytes):
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if isinstance(raw, dict) and "bytes" in raw:
        data = raw["bytes"]
        if data:
            return Image.open(io.BytesIO(data)).convert("RGB")
    raise ValueError(f"Cannot decode image of type {type(raw)}: {str(raw)[:80]}")


# ── Sample processing (shared logic) ─────────────────────────────────────────

def _process_sample(
    row: Dict[str, Any],
    processor,
    is_training: bool,
    image_col: str,
    disease_col: str,
    crop_col: Optional[str],
    label_id_col: Optional[str],
    label2id: Dict[str, int],
) -> Dict[str, Any]:
    image        = _decode_image(row[image_col])
    disease      = str(row[disease_col]).strip()
    crop         = str(row.get(crop_col, "")).strip() if crop_col else ""
    label_id     = int(row[label_id_col]) if label_id_col and label_id_col in row \
                   else label2id.get(disease, -1)

    prompt_msgs  = build_conversation(image=image, disease_label=None,
                                      include_crop=bool(crop), crop_name=crop)
    prompt_text  = processor.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True)

    if is_training:
        full_msgs = build_conversation(image=image, disease_label=disease,
                                       include_crop=bool(crop), crop_name=crop)
        full_text = processor.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False)

        inputs        = processor(text=[full_text], images=[image], return_tensors="pt")
        input_ids     = inputs["input_ids"][0]
        attn_mask     = inputs["attention_mask"][0]
        pixel_values  = inputs["pixel_values"]
        grid_thw      = inputs.get("image_grid_thw")

        # Compute prompt boundary for label masking
        full_tok_len   = processor.tokenizer(full_text, add_special_tokens=False,
                                             return_tensors="pt")["input_ids"].shape[1]
        prompt_tok_len = processor.tokenizer(prompt_text, add_special_tokens=False,
                                             return_tensors="pt")["input_ids"].shape[1]
        visual_tokens  = input_ids.shape[0] - full_tok_len
        boundary       = max(0, min(visual_tokens + prompt_tok_len, input_ids.shape[0]))

        labels         = input_ids.clone()
        labels[:boundary] = -100

        return {"disease": disease, "crop": crop, "label_id": label_id,
                "input_ids": input_ids, "attention_mask": attn_mask,
                "pixel_values": pixel_values, "image_grid_thw": grid_thw,
                "labels": labels}
    else:
        inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
        return {"disease": disease, "crop": crop, "label_id": label_id,
                "input_ids": inputs["input_ids"][0],
                "attention_mask": inputs["attention_mask"][0],
                "pixel_values": inputs["pixel_values"],
                "image_grid_thw": inputs.get("image_grid_thw"),
                "prompt_text": prompt_text}


# ── Streaming Dataset (HuggingFace Hub, zero download) ───────────────────────

class StreamingSAGEDataset(IterableDataset):
    """
    Streams samples from HuggingFace Hub without downloading the full dataset.
    Requires internet access. Uses datasets.load_dataset(..., streaming=True).
    """
    def __init__(
        self,
        repo_id: str,
        processor,
        split: str = "train",
        max_samples: Optional[int] = None,
        is_training: bool = True,
        image_col: str = "image",
        disease_col: str = "disease",
        crop_col: str = "crop",
        label_id_col: str = "",
        label2id: Optional[Dict[str, int]] = None,
        seed: int = 42,
    ):
        from datasets import load_dataset
        print(f"[StreamingDataset] Connecting to {repo_id} (streaming=True, split={split})...")
        hf_ds = load_dataset(repo_id, split=split, streaming=True)
        hf_ds = hf_ds.shuffle(seed=seed, buffer_size=1000)
        if max_samples:
            hf_ds = hf_ds.take(max_samples)

        self._hf_ds      = hf_ds
        self.processor   = processor
        self.is_training = is_training
        self.image_col   = image_col
        self.disease_col = disease_col
        self.crop_col    = crop_col or None
        self.label_id_col = label_id_col or None
        self.max_samples = max_samples
        self.label2id    = label2id or {}

    def _build_label2id_from_streaming(self, scan_samples: int = 5000):
        """Scan first N samples to build label vocab (only needed once)."""
        print(f"[StreamingDataset] Scanning {scan_samples} samples to build label vocab...")
        labels = set()
        from datasets import load_dataset
        scan_ds = load_dataset(
            self._hf_ds.info.builder_name if hasattr(self._hf_ds, 'info') else "tirtho149/SAGE",
            split="train", streaming=True
        ).take(scan_samples)
        for row in scan_ds:
            labels.add(str(row[self.disease_col]).strip())
        self.label2id = {l: i for i, l in enumerate(sorted(labels))}
        print(f"[StreamingDataset] Found {len(self.label2id)} unique disease labels.")
        return self.label2id

    def __iter__(self):
        for row in self._hf_ds:
            try:
                yield _process_sample(
                    row, self.processor, self.is_training,
                    self.image_col, self.disease_col,
                    self.crop_col, self.label_id_col, self.label2id
                )
            except Exception as e:
                # Skip corrupted samples silently
                continue

    def __len__(self):
        # IterableDataset: return max_samples if known, else raise
        if self.max_samples:
            return self.max_samples
        raise TypeError("StreamingSAGEDataset length is unknown without max_samples set.")


# ── Map-style Dataset (local parquet files) ───────────────────────────────────

class SAGEDataset(Dataset):
    """
    Loads from local parquet files. Fast random access.
    Use when parquet files are already downloaded on disk.
    """
    def __init__(
        self,
        parquet_dir: str,
        processor,
        is_training: bool = True,
        image_col: str = "image",
        disease_col: str = "disease",
        crop_col: str = "crop",
        label_id_col: str = "",
        label2id: Optional[Dict[str, int]] = None,
    ):
        files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
        if not files:
            raise FileNotFoundError(f"No parquet files in: {parquet_dir}")
        print(f"[Dataset] Loading {len(files)} parquet files...")
        self._df         = pd.concat([pd.read_parquet(f, engine="pyarrow") for f in files],
                                     ignore_index=True)
        print(f"[Dataset] {len(self._df)} rows loaded. Columns: {list(self._df.columns)}")

        self.processor   = processor
        self.is_training = is_training
        self.image_col   = image_col
        self.disease_col = disease_col
        self.crop_col    = crop_col or None
        self.label_id_col = label_id_col or None

        if label2id is not None:
            self.label2id = label2id
        else:
            labels = sorted(self._df[disease_col].dropna().unique().tolist())
            self.label2id = {l: i for i, l in enumerate(labels)}

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        row = self._df.iloc[idx].to_dict()
        return _process_sample(
            row, self.processor, self.is_training,
            self.image_col, self.disease_col,
            self.crop_col, self.label_id_col, self.label2id
        )


# ── Collate function (supports batch_size ≥ 1) ────────────────────────────────

def sage_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Left-pads sequences to the longest in the batch."""
    pad_id  = 0
    max_len = max(b["input_ids"].shape[0] for b in batch)

    input_ids_list, attn_list = [], []
    for b in batch:
        seq = b["input_ids"]
        pad = max_len - seq.shape[0]
        input_ids_list.append(torch.cat([torch.full((pad,), pad_id, dtype=seq.dtype), seq]))
        attn = b["attention_mask"]
        attn_list.append(torch.cat([torch.zeros(pad, dtype=attn.dtype), attn]))

    result = {
        "disease":         [b["disease"] for b in batch],
        "crop":            [b["crop"]    for b in batch],
        "label_id":        torch.tensor([b["label_id"] for b in batch], dtype=torch.long),
        "input_ids":       torch.stack(input_ids_list),
        "attention_mask":  torch.stack(attn_list),
        "pixel_values":    torch.cat([b["pixel_values"]   for b in batch], dim=0),
        "image_grid_thw":  torch.cat([b["image_grid_thw"] for b in batch], dim=0),
    }
    if "labels" in batch[0]:
        labels_list = []
        for b in batch:
            lbl = b["labels"]
            pad = max_len - lbl.shape[0]
            labels_list.append(torch.cat([torch.full((pad,), -100, dtype=lbl.dtype), lbl]))
        result["labels"] = torch.stack(labels_list)
    if "prompt_text" in batch[0]:
        result["prompt_text"] = [b["prompt_text"] for b in batch]
    return result
