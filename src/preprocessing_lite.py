"""
Processor loader for Qwen2.5-VL with configurable resolution bounds.
"""
from transformers import AutoProcessor


def get_processor(
    model_name_or_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    min_pixels: int = None,
    max_pixels: int = None,
    local_files_only: bool = False,
):
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        local_files_only=local_files_only,
    )
    if min_pixels is not None:
        processor.image_processor.min_pixels = min_pixels
    if max_pixels is not None:
        processor.image_processor.max_pixels = max_pixels
    return processor
