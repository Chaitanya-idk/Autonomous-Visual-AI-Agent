"""
PyTorch datasets for the SAGE crop-disease task.

Supports:
  1. Hugging Face streaming datasets.
  2. Temporary Parquet shards downloaded by the rolling-window trainer.
  3. Ordinary local Parquet files.

Important:
    Every image is normalized to a standard RGB PIL image before being
    passed to Qwen2.5-VL. This is necessary because SAGE/Hugging Face
    image representations can occasionally contain unusual channel layouts.
"""

import io
import os
import glob
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, IterableDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prompts import build_conversation


# =============================================================================
# IMAGE DECODING / NORMALIZATION
# =============================================================================

def _array_to_rgb_image(array: Any) -> Image.Image:
    """
    Convert a NumPy/array-like image into a valid RGB PIL image.

    Handles common layouts:

        H x W
        H x W x 1
        H x W x 3
        H x W x 4
        1 x H x W
        3 x H x W
        4 x H x W

    The function is deliberately conservative when deciding whether an
    array is CHW or HWC.

    Raises:
        ValueError: if the array cannot be interpreted as an image.
    """

    array = np.asarray(array)

    if array.size == 0:
        raise ValueError("Image array is empty.")

    # Remove unnecessary dimensions such as:
    # (1, H, W) / (H, W, 1) only when appropriate below.
    original_shape = tuple(array.shape)

    # -------------------------------------------------------------------------
    # Scalar / 1-D data is not a valid image.
    # -------------------------------------------------------------------------
    if array.ndim < 2:
        raise ValueError(
            f"Image array has invalid shape {original_shape}."
        )

    # -------------------------------------------------------------------------
    # More than 3 dimensions.
    #
    # Occasionally image data may arrive with a singleton leading/trailing
    # dimension. Remove ONLY singleton dimensions. Do not blindly squeeze
    # arbitrary dimensions because that can destroy legitimate image shapes.
    # -------------------------------------------------------------------------
    if array.ndim > 3:
        squeezed = np.squeeze(array)

        if squeezed.ndim != 2 and squeezed.ndim != 3:
            raise ValueError(
                f"Unsupported image array shape {original_shape} "
                f"after squeeze: {squeezed.shape}"
            )

        array = squeezed

    # -------------------------------------------------------------------------
    # Grayscale image: H x W
    # -------------------------------------------------------------------------
    if array.ndim == 2:
        return Image.fromarray(
            _normalize_array_dtype(array)
        ).convert("RGB")

    # -------------------------------------------------------------------------
    # 3-D image.
    # -------------------------------------------------------------------------
    if array.ndim == 3:
        h, w, c = array.shape

        # -------------------------------------------------------------
        # HWC detection.
        #
        # Standard image formats have channels at the end:
        # H x W x 1
        # H x W x 3
        # H x W x 4
        # -------------------------------------------------------------
        if c in (1, 3, 4):
            array = _normalize_array_dtype(array)

            if c == 1:
                array = np.repeat(array, 3, axis=2)

            elif c == 4:
                # RGBA -> RGB
                return Image.fromarray(array).convert("RGB")

            return Image.fromarray(array).convert("RGB")

        # -------------------------------------------------------------
        # CHW detection.
        #
        # Common layouts:
        # 1 x H x W
        # 3 x H x W
        # 4 x H x W
        # -------------------------------------------------------------
        if h in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))
            array = _normalize_array_dtype(array)

            channels = array.shape[2]

            if channels == 1:
                array = np.repeat(array, 3, axis=2)

            elif channels == 4:
                return Image.fromarray(array).convert("RGB")

            return Image.fromarray(array).convert("RGB")

        # -------------------------------------------------------------
        # Special handling for unusual shapes such as:
        #
        #     (1, 30, 3)
        #
        # This is exactly the type of shape encountered in the failed
        # training run.
        #
        # It has 3 channels at the END, therefore it should be interpreted
        # as HWC, even though the height is only 1.
        # -------------------------------------------------------------
        if c == 3:
            array = _normalize_array_dtype(array)
            return Image.fromarray(array).convert("RGB")

        raise ValueError(
            f"Unsupported 3-D image shape {original_shape}. "
            f"Expected HxWxC or CxHxW with 1, 3, or 4 channels."
        )

    raise ValueError(
        f"Unsupported image array dimensionality: "
        f"{array.ndim}, shape={original_shape}"
    )


def _normalize_array_dtype(array: np.ndarray) -> np.ndarray:
    """
    Convert an arbitrary numeric image array into uint8 RGB-compatible data.
    """

    array = np.asarray(array)

    if array.dtype == np.uint8:
        return array

    if np.issubdtype(array.dtype, np.floating):
        finite = np.isfinite(array)

        if not finite.all():
            array = np.nan_to_num(
                array,
                nan=0.0,
                posinf=255.0,
                neginf=0.0,
            )

        min_val = float(array.min())
        max_val = float(array.max())

        # Common normalized image representation: [0, 1]
        if min_val >= 0.0 and max_val <= 1.0:
            array = array * 255.0

        # Otherwise assume an approximately [0,255] representation.
        array = np.clip(array, 0.0, 255.0)

        return array.astype(np.uint8)

    if np.issubdtype(array.dtype, np.integer):
        return np.clip(array, 0, 255).astype(np.uint8)

    raise ValueError(
        f"Unsupported image dtype: {array.dtype}"
    )


def _decode_image(raw) -> Image.Image:
    """
    Decode a SAGE image into a guaranteed RGB PIL Image.

    Supported representations include:

      - PIL.Image.Image
      - raw image bytes
      - Hugging Face image dictionaries
      - NumPy arrays
      - array-like objects
    """

    # -------------------------------------------------------------------------
    # PIL image
    # -------------------------------------------------------------------------
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")

    # -------------------------------------------------------------------------
    # Raw bytes / bytearray / memoryview
    # -------------------------------------------------------------------------
    if isinstance(raw, (bytes, bytearray, memoryview)):
        try:
            image = Image.open(io.BytesIO(bytes(raw)))
            image.load()
            return image.convert("RGB")
        except Exception as exc:
            raise ValueError(
                f"Unable to decode raw image bytes: {exc}"
            ) from exc

    # -------------------------------------------------------------------------
    # Hugging Face image dictionary.
    #
    # Typical forms:
    #
    # {"bytes": b"...", "path": "..."}
    #
    # or:
    #
    # {"path": "..."}
    # -------------------------------------------------------------------------
    if isinstance(raw, dict):

        data = raw.get("bytes")

        if data:
            try:
                image = Image.open(io.BytesIO(data))
                image.load()
                return image.convert("RGB")
            except Exception as exc:
                raise ValueError(
                    f"Unable to decode image dictionary bytes: {exc}"
                ) from exc

        path = raw.get("path")

        if path:
            if os.path.exists(path):
                try:
                    image = Image.open(path)
                    image.load()
                    return image.convert("RGB")
                except Exception as exc:
                    raise ValueError(
                        f"Unable to open image path '{path}': {exc}"
                    ) from exc

            raise ValueError(
                f"Image path does not exist: {path}"
            )

        # Some dataset implementations may expose an array in a dictionary.
        for key in ("array", "data", "image"):
            if key in raw and raw[key] is not None:
                return _decode_image(raw[key])

    # -------------------------------------------------------------------------
    # NumPy / array-like image
    # -------------------------------------------------------------------------
    if isinstance(raw, np.ndarray):
        return _array_to_rgb_image(raw)

    # -------------------------------------------------------------------------
    # Some Arrow / HF objects can expose an array through to_numpy().
    # -------------------------------------------------------------------------
    if hasattr(raw, "to_numpy"):
        try:
            array = raw.to_numpy()
            return _array_to_rgb_image(array)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Last-resort array conversion.
    #
    # Only attempt this for objects that are reasonably likely to be
    # array-like. This prevents confusing error messages for arbitrary data.
    # -------------------------------------------------------------------------
    try:
        array = np.asarray(raw)

        if array.ndim >= 2:
            return _array_to_rgb_image(array)

    except Exception:
        pass

    raise ValueError(
        f"Cannot decode image. "
        f"type={type(raw)}, "
        f"representation={str(raw)[:200]}"
    )


# =============================================================================
# SAMPLE PROCESSING
# =============================================================================

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

    # -------------------------------------------------------------------------
    # Decode image FIRST.
    # -------------------------------------------------------------------------
    try:
        image = _decode_image(row[image_col])
    except Exception as exc:
        raise ValueError(
            f"Image decoding failed for sample. "
            f"image_col='{image_col}', "
            f"error={exc}"
        ) from exc

    # Final safety check.
    if not isinstance(image, Image.Image):
        raise TypeError(
            f"Decoded image is not PIL.Image.Image: {type(image)}"
        )

    # Qwen2.5-VL expects RGB images.
    if image.mode != "RGB":
        image = image.convert("RGB")

    disease = str(row[disease_col]).strip()

    crop = (
        str(row.get(crop_col, "")).strip()
        if crop_col
        else ""
    )

    # -------------------------------------------------------------------------
    # Label ID
    # -------------------------------------------------------------------------
    if (
        label_id_col
        and label_id_col in row
        and row[label_id_col] is not None
    ):
        label_id = int(row[label_id_col])
    else:
        label_id = label2id.get(disease, -1)

    # -------------------------------------------------------------------------
    # Prompt only.
    #
    # IMPORTANT:
    # There is NO answer in this conversation.
    #
    # This is used for generation/evaluation and as the prefix of the
    # supervised training conversation.
    # -------------------------------------------------------------------------
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

    # =========================================================================
    # GENERATION / EVALUATION MODE
    # =========================================================================

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

    # =========================================================================
    # TRAINING MODE
    # =========================================================================

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

    # -------------------------------------------------------------------------
    # Calculate token lengths before visual-token expansion.
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Qwen's processor expands image placeholders into visual tokens.
    # Account for that expansion when locating the assistant answer.
    # -------------------------------------------------------------------------
    visual_tokens = max(
        0,
        input_ids.shape[0] - full_tok_len,
    )

    boundary = max(
        0,
        min(
            visual_tokens + prompt_tok_len,
            input_ids.shape[0],
        ),
    )

    labels = input_ids.clone()

    # Mask prompt tokens.
    labels[:boundary] = -100

    # -------------------------------------------------------------------------
    # Safety check.
    # -------------------------------------------------------------------------
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


# =============================================================================
# HUGGING FACE STREAMING DATASET
# =============================================================================

class StreamingSAGEDataset(IterableDataset):
    """
    Stream SAGE samples from Hugging Face without downloading the entire
    dataset.
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
        shuffle_buffer: int = 1000,
    ):

        from datasets import load_dataset

        print(
            f"[StreamingDataset] Connecting to {repo_id} "
            f"(streaming=True, split={split})..."
        )

        hf_ds = load_dataset(
            repo_id,
            split=split,
            streaming=True,
        )

        hf_ds = hf_ds.shuffle(
            seed=seed,
            buffer_size=shuffle_buffer,
        )

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

    def _build_label2id_from_streaming(
        self,
        scan_samples: int = 5000,
    ):
        """Build a vocabulary from a bounded HF streaming scan."""

        from datasets import load_dataset

        print(
            f"[StreamingDataset] Scanning {scan_samples} samples "
            "to build label vocab..."
        )

        scan_stream = (
            load_dataset(
                self.repo_id,
                split="train",
                streaming=True,
            )
            .shuffle(
                seed=42,
                buffer_size=1000,
            )
            .take(scan_samples)
        )

        labels = set()

        for row in scan_stream:

            lbl = row.get(self.disease_col)

            if lbl is not None:
                labels.add(
                    str(lbl).strip()
                )

        self.label2id = {
            label: i
            for i, label in enumerate(
                sorted(labels)
            )
        }

        print(
            f"[StreamingDataset] Found "
            f"{len(self.label2id)} disease classes."
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

                # Keep the streaming run alive while making the problematic
                # sample visible.
                print(
                    "[StreamingDataset] "
                    f"Skipping invalid sample: {exc}",
                    flush=True,
                )

    def __len__(self):

        if self.max_samples:
            return self.max_samples

        raise TypeError(
            "StreamingSAGEDataset length is unknown "
            "without max_samples."
        )


# =============================================================================
# MAP-STYLE PARQUET DATASET
# =============================================================================

class SAGEDataset(Dataset):
    """
    Load one or more local Parquet shards into a map-style dataset.
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

        files = sorted(
            glob.glob(
                os.path.join(
                    parquet_dir,
                    "*.parquet",
                )
            )
        )

        if not files:
            raise FileNotFoundError(
                f"No parquet files in: {parquet_dir}"
            )

        print(
            f"[Dataset] Loading {len(files)} parquet files..."
        )

        self._df = pd.concat(
            [
                pd.read_parquet(
                    path,
                    engine="pyarrow",
                )
                for path in files
            ],
            ignore_index=True,
        )

        print(
            f"[Dataset] {len(self._df)} rows loaded. "
            f"Columns: {list(self._df.columns)}"
        )

        required = {
            image_col,
            disease_col,
        }

        missing = required - set(
            self._df.columns
        )

        if missing:
            raise KeyError(
                f"Missing required dataset columns: "
                f"{sorted(missing)}"
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
                for label in
                self._df[disease_col]
                .dropna()
                .unique()
                .tolist()
            )

            self.label2id = {
                label: i
                for i, label in enumerate(labels)
            }

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):

        row = self._df.iloc[idx].to_dict()

        try:

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

        except Exception as exc:

            # Include the exact local dataset index. This is extremely useful
            # if another malformed image is encountered during training.
            raise RuntimeError(
                f"Failed to process dataset sample idx={idx}. "
                f"Error: {exc}"
            ) from exc


# =============================================================================
# COLLATION
# =============================================================================

def sage_collate_fn(
    batch: List[Dict[str, Any]],
    pad_token_id: int = 0,
) -> Dict[str, Any]:
    """
    Left-pad text tensors and concatenate Qwen visual tensors.
    """

    if not batch:
        raise ValueError(
            "Cannot collate an empty batch."
        )

    max_len = max(
        item["input_ids"].shape[0]
        for item in batch
    )

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
                    torch.zeros(
                        pad,
                        dtype=attn.dtype,
                    ),
                    attn,
                ]
            )
        )

    result = {
        "disease": [
            item["disease"]
            for item in batch
        ],

        "crop": [
            item["crop"]
            for item in batch
        ],

        "label_id": torch.tensor(
            [
                item["label_id"]
                for item in batch
            ],
            dtype=torch.long,
        ),

        "input_ids": torch.stack(
            input_ids_list
        ),

        "attention_mask": torch.stack(
            attn_list
        ),

        "pixel_values": torch.cat(
            [
                item["pixel_values"]
                for item in batch
            ],
            dim=0,
        ),

        "image_grid_thw": torch.cat(
            [
                item["image_grid_thw"]
                for item in batch
            ],
            dim=0,
        ),
    }

    # -------------------------------------------------------------------------
    # Training batches contain labels.
    # Generation/evaluation batches intentionally do not.
    # -------------------------------------------------------------------------
    if all(
        "labels" in item
        for item in batch
    ):

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

        result["labels"] = torch.stack(
            labels_list
        )

    # -------------------------------------------------------------------------
    # Generation/evaluation batches contain prompt_text.
    # -------------------------------------------------------------------------
    if all(
        "prompt_text" in item
        for item in batch
    ):

        result["prompt_text"] = [
            item["prompt_text"]
            for item in batch
        ]

    return result