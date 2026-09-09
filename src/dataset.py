"""
PyTorch datasets for the SAGE crop-disease task.

The project supports three access patterns:
  1. Hugging Face streaming (reads samples from the Hub without materialising
     the full dataset locally).
  2. Temporary Parquet shards downloaded by src/train.py in rolling/chunked
     mode. Only the active shards are kept on Kaggle disk.
  3. Ordinary local Parquet files.

A sample can be prepared in either:
  - training mode: image + prompt + ground-truth answer, with prompt tokens
    masked out in labels;
  - generation/evaluation mode: image + prompt only. The answer is never
    included in model inputs, preventing label leakage during generation.
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


# -----------------------------------------------------------------------------
# Image decoding
# -----------------------------------------------------------------------------

def _decode_image(raw) -> Image.Image:
    """Decode a SAGE image from PIL, raw bytes, or an HF image dictionary."""
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")

    if isinstance(raw, bytes):
        return Image.open(io.BytesIO(raw)).convert("RGB")

    if isinstance(raw, dict):
        data = raw.get("bytes")
        if data:
            return Image.open(io.BytesIO(data)).convert("RGB")

        # Some HF image features may expose a path instead of bytes.
        path = raw.get("path")
        if path and os.path.exists(path):
            return Image.open(path).convert("RGB")

    raise ValueError(
        f"Cannot decode image of type {type(raw)}: {str(raw)[:120]}"
    )


# -----------------------------------------------------------------------------
# Sample processing
# -----------------------------------------------------------------------------

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
    """Convert one raw dataset row into Qwen2.5-VL inputs."""
    image = _decode_image(row[image_col])
    disease = str(row[disease_col]).strip()
    crop = str(row.get(crop_col, "")).strip() if crop_col else ""

    if label_id_col and label_id_col in row and row[label_id_col] is not None:
        label_id = int(row[label_id_col])
    else:
        label_id = label2id.get(disease, -1)

    # IMPORTANT: this conversation contains no answer. It is used both for
    # generation evaluation and as the prefix of the supervised conversation.
    prompt_msgs = build_conversation(
        image=image,
        disease_label=None,
        include_crop=bool(crop),
        crop_name=crop,
    )
    prompt_text = processor.apply_chat_template(
        prompt_msgs,
        tokenize=False,
        add_generation_prompt=True,
    )

    if not is_training:
        inputs = processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
        )

        return {
            "disease": disease,
            "crop": crop,
            "label_id": label_id,
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs.get("image_grid_thw"),
            "prompt_text": prompt_text,
        }

    # Training: append the ground-truth assistant answer.
    full_msgs = build_conversation(
        image=image,
        disease_label=disease,
        include_crop=bool(crop),
        crop_name=crop,
    )
    full_text = processor.apply_chat_template(
        full_msgs,
        tokenize=False,
        add_generation_prompt=False,
    )

    inputs = processor(
        text=[full_text],
        images=[image],
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"][0]
    attn_mask = inputs["attention_mask"][0]
    pixel_values = inputs["pixel_values"]
    grid_thw = inputs.get("image_grid_thw")

    # Qwen's processor expands image placeholders into visual tokens. We need
    # to account for that expansion when locating the beginning of the answer.
    full_tok_len = processor.tokenizer(
        full_text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].shape[1]
    prompt_tok_len = processor.tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].shape[1]

    visual_tokens = max(0, input_ids.shape[0] - full_tok_len)
    boundary = max(
        0,
        min(visual_tokens + prompt_tok_len, input_ids.shape[0]),
    )

    labels = input_ids.clone()
    labels[:boundary] = -100

    # Sanity check: at least one target token should remain unmasked. If the
    # processor/template changes in a future Transformers version, fail early
    # instead of silently training on an empty target.
    if not torch.any(labels != -100):
        raise ValueError(
            "No target tokens remained after prompt masking. "
            "The Qwen chat-template boundary calculation needs updating."
        )

    return {
        "disease": disease,
        "crop": crop,
        "label_id": label_id,
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": grid_thw,
        "labels": labels,
    }


# -----------------------------------------------------------------------------
# Hugging Face streaming dataset
# -----------------------------------------------------------------------------

class StreamingSAGEDataset(IterableDataset):
    """Stream SAGE samples from Hugging Face without downloading the dataset."""

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
        shuffle_buffer: int = 1000,
    ):
        from datasets import load_dataset

        print(
            f"[StreamingDataset] Connecting to {repo_id} "
            f"(streaming=True, split={split})..."
        )
        hf_ds = load_dataset(repo_id, split=split, streaming=True)
        hf_ds = hf_ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
        if max_samples:
            hf_ds = hf_ds.take(max_samples)

        self._hf_ds = hf_ds
        self.repo_id = repo_id
        self.processor = processor
        self.is_training = is_training
        self.image_col = image_col
        self.disease_col = disease_col
        self.crop_col = crop_col or None
        self.label_id_col = label_id_col or None
        self.max_samples = max_samples
        self.label2id = label2id or {}

    def _build_label2id_from_streaming(self, scan_samples: int = 5000):
        """Build a vocabulary from a bounded HF streaming scan."""
        from datasets import load_dataset

        print(
            f"[StreamingDataset] Scanning {scan_samples} samples "
            "to build label vocab..."
        )
        scan_stream = (
            load_dataset(self.repo_id, split="train", streaming=True)
            .shuffle(seed=42, buffer_size=1000)
            .take(scan_samples)
        )

        labels = set()
        for row in scan_stream:
            lbl = row.get(self.disease_col)
            if lbl is not None:
                labels.add(str(lbl).strip())

        self.label2id = {label: i for i, label in enumerate(sorted(labels))}
        print(
            f"[StreamingDataset] Found {len(self.label2id)} disease classes."
        )
        return self.label2id

    def __iter__(self):
        for row in self._hf_ds:
            try:
                yield _process_sample(
                    row,
                    self.processor,
                    self.is_training,
                    self.image_col,
                    self.disease_col,
                    self.crop_col,
                    self.label_id_col,
                    self.label2id,
                )
            except Exception as exc:
                # Do not silently hide data-quality problems. Keep the run
                # alive, but make the skipped sample visible in the log.
                print(
                    f"[StreamingDataset] Skipping invalid sample: {exc}",
                    flush=True,
                )

    def __len__(self):
        if self.max_samples:
            return self.max_samples
        raise TypeError(
            "StreamingSAGEDataset length is unknown without max_samples."
        )


# -----------------------------------------------------------------------------
# Map-style Parquet dataset
# -----------------------------------------------------------------------------

class SAGEDataset(Dataset):
    """Load one or more local Parquet shards into a map-style dataset."""

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
        self._df = pd.concat(
            [pd.read_parquet(path, engine="pyarrow") for path in files],
            ignore_index=True,
        )
        print(
            f"[Dataset] {len(self._df)} rows loaded. "
            f"Columns: {list(self._df.columns)}"
        )

        required = {image_col, disease_col}
        missing = required - set(self._df.columns)
        if missing:
            raise KeyError(
                f"Missing required dataset columns: {sorted(missing)}"
            )

        self.processor = processor
        self.is_training = is_training
        self.image_col = image_col
        self.disease_col = disease_col
        self.crop_col = crop_col or None
        self.label_id_col = label_id_col or None

        if label2id is not None:
            self.label2id = dict(label2id)
        else:
            labels = sorted(
                str(label).strip()
                for label in self._df[disease_col].dropna().unique().tolist()
            )
            self.label2id = {label: i for i, label in enumerate(labels)}

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        row = self._df.iloc[idx].to_dict()
        return _process_sample(
            row,
            self.processor,
            self.is_training,
            self.image_col,
            self.disease_col,
            self.crop_col,
            self.label_id_col,
            self.label2id,
        )


# -----------------------------------------------------------------------------
# Collation
# -----------------------------------------------------------------------------

def sage_collate_fn(
    batch: List[Dict[str, Any]],
    pad_token_id: int = 0,
) -> Dict[str, Any]:
    """Left-pad text tensors and concatenate Qwen visual tensors."""
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids_list = []
    attn_list = []
    for item in batch:
        seq = item["input_ids"]
        pad = max_len - seq.shape[0]

        input_ids_list.append(
            torch.cat(
                [
                    torch.full(
                        (pad,),
                        pad_token_id,
                        dtype=seq.dtype,
                    ),
                    seq,
                ]
            )
        )

        attn = item["attention_mask"]
        attn_list.append(
            torch.cat(
                [
                    torch.zeros(pad, dtype=attn.dtype),
                    attn,
                ]
            )
        )

    result = {
        "disease": [item["disease"] for item in batch],
        "crop": [item["crop"] for item in batch],
        "label_id": torch.tensor(
            [item["label_id"] for item in batch],
            dtype=torch.long,
        ),
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attn_list),
        "pixel_values": torch.cat(
            [item["pixel_values"] for item in batch],
            dim=0,
        ),
        "image_grid_thw": torch.cat(
            [item["image_grid_thw"] for item in batch],
            dim=0,
        ),
    }

    # Training batches have labels; generation batches intentionally do not.
    if all("labels" in item for item in batch):
        labels_list = []
        for item in batch:
            labels = item["labels"]
            pad = max_len - labels.shape[0]
            labels_list.append(
                torch.cat(
                    [
                        torch.full(
                            (pad,),
                            -100,
                            dtype=labels.dtype,
                        ),
                        labels,
                    ]
                )
            )
        result["labels"] = torch.stack(labels_list)

    if all("prompt_text" in item for item in batch):
        result["prompt_text"] = [item["prompt_text"] for item in batch]

    return result
