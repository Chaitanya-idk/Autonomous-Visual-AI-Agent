"""Qwen2.5-VL processor loading with Kaggle/Transformers compatibility.

The SAGE training environment may ship a Transformers release where
``Qwen2_5_VLProcessor`` exists but AutoProcessor cannot resolve the legacy
``Qwen2_5_VLImageProcessor`` name found in some Qwen2.5-VL checkpoints.
This module keeps the model weights untouched and falls back to the
Qwen2VL image processor, which is the image processor used by Qwen2.5-VL.
"""

import json
from pathlib import Path
from typing import Optional

from transformers import AutoProcessor, AutoTokenizer


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_qwen25_processor_compat(
    model_name_or_path: str,
    local_files_only: bool,
):
    """Manually construct Qwen2.5-VL processor when AutoProcessor fails.

    Some released Transformers versions contain Qwen2.5-VL model/processor
    support but do not expose the ``Qwen2_5_VLImageProcessor`` class name
    referenced by older checkpoint ``preprocessor_config.json`` files.

    Qwen2.5-VL uses the Qwen2VL image processor internally, so we construct
    that image processor directly and pair it with Qwen2_5_VLProcessor.
    No model weights are copied or modified.
    """
    model_dir = Path(model_name_or_path)
    preprocessor_path = model_dir / "preprocessor_config.json"

    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Missing preprocessor_config.json in {model_dir}"
        )

    try:
        from transformers import Qwen2_5_VLProcessor, Qwen2VLImageProcessor, Qwen2VLVideoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "This Transformers installation cannot provide the Qwen2.5-VL "
            "processor plus Qwen2VL image processor required for the "
            "compatibility fallback. Install a Transformers build with "
            "Qwen2.5-VL support before starting training."
        ) from exc

    image_config = _read_json(preprocessor_path)
    image_config.pop("image_processor_type", None)
    image_config.pop("processor_class", None)

    image_processor = Qwen2VLImageProcessor(**image_config)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        local_files_only=local_files_only,
    )

    # AutoTokenizer normally obtains the chat template from tokenizer_config.
    # Some local model exports also provide chat_template.json; use it only if
    # the tokenizer did not already contain a template.
    chat_template = getattr(tokenizer, "chat_template", None)
    chat_template_path = model_dir / "chat_template.json"

    if not chat_template and chat_template_path.exists():
        try:
            chat_data = _read_json(chat_template_path)
            if isinstance(chat_data, dict):
                chat_template = chat_data.get("chat_template")
            elif isinstance(chat_data, str):
                chat_template = chat_data
        except Exception:
            chat_template = None

    # Transformers 5.x validates that Qwen2_5_VLProcessor receives a
    # BaseVideoProcessor instance, even when the application only supplies
    # still images. In the Kaggle Transformers build used by this project,
    # constructing Qwen2VLVideoProcessor from the image-only checkpoint
    # config can return None. That causes ProcessorMixin validation to fail.
    #
    # SAGE contains still images only, so use an uninitialized
    # Qwen2VLVideoProcessor instance strictly as the required type placeholder.
    # No video-processing method is called by the image-only training path.
    #
    # We deliberately set the two attributes that Qwen2.5-VL would need if
    # video processing were ever requested, while keeping the actual image
    # processor fully functional.
    try:
        video_processor = object.__new__(Qwen2VLVideoProcessor)
        video_processor.merge_size = int(image_config.get("merge_size", 2))
        video_processor.temporal_patch_size = int(
            image_config.get("temporal_patch_size", 2)
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not create the Qwen2VLVideoProcessor compatibility "
            "placeholder required by Qwen2_5_VLProcessor."
        ) from exc

    if video_processor is None:
        raise RuntimeError(
            "Qwen2VLVideoProcessor compatibility placeholder unexpectedly "
            "resolved to None."
        )

    processor_kwargs = {
        "image_processor": image_processor,
        "tokenizer": tokenizer,
        "video_processor": video_processor,
    }
    if chat_template:
        processor_kwargs["chat_template"] = chat_template

    processor = Qwen2_5_VLProcessor(**processor_kwargs)

    print(
        "[Processor] AutoProcessor could not resolve the checkpoint image "
        "processor; using Qwen2VLImageProcessor compatibility fallback."
    )
    print(
        f"[Processor] image_processor={type(processor.image_processor).__name__}, "
        f"video_processor={type(processor.video_processor).__name__}"
    )
    return processor


def get_processor(
    model_name_or_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    local_files_only: bool = False,
):
    """Load a Qwen2.5-VL processor robustly across Transformers releases."""
    try:
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )
    except (ValueError, OSError, ImportError) as exc:
        message = str(exc).lower()
        processor_error = (
            "unrecognized image processor" in message
            or "qwen2_5_vlimageprocessor" in message
            or "image processor" in message and "qwen2_5_vl" in message
        )

        if not processor_error:
            raise

        processor = _load_qwen25_processor_compat(
            model_name_or_path,
            local_files_only=local_files_only,
        )

    if min_pixels is not None:
        processor.image_processor.min_pixels = int(min_pixels)
        # Older/newer Qwen processors may store the bounds in ``size`` too.
        if hasattr(processor.image_processor, "size") and isinstance(
            processor.image_processor.size, dict
        ):
            processor.image_processor.size["shortest_edge"] = int(min_pixels)

    if max_pixels is not None:
        processor.image_processor.max_pixels = int(max_pixels)
        if hasattr(processor.image_processor, "size") and isinstance(
            processor.image_processor.size, dict
        ):
            processor.image_processor.size["longest_edge"] = int(max_pixels)

    return processor
