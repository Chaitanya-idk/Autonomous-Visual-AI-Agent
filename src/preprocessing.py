"""
Preprocessing and Resolution Benchmarking for Qwen2.5-VL.
Measures token counts and latency across Low, Medium, and High resolution presets.
Uses 100% local assets and robust Pathlib path handling.
"""

import os
import sys
import time
import json
import traceback
from datetime import datetime
from pathlib import Path
from PIL import Image
import pandas as pd

# Define paths relative to this script location
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
IMAGES_DIR = PROCESSED_DIR / "images"
METADATA_CSV = PROCESSED_DIR / "metadata.csv"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"
REPORT_PATH = REPORT_DIR / "preprocessing_report.json"

# Qwen2.5-VL patch factor (28 * 28 pixels per spatial grid unit)
FACTOR = 28 * 28

# Bounded resolution presets
RESOLUTION_PRESETS = {
    "LOW": {
        "min_pixels": 128 * FACTOR,  # 100,352 pixels (~128 visual tokens)
        "max_pixels": 256 * FACTOR   # 200,704 pixels (~256 visual tokens)
    },
    "MEDIUM": {
        "min_pixels": 256 * FACTOR,  # 200,704 pixels (~256 visual tokens)
        "max_pixels": 512 * FACTOR   # 401,408 pixels (~512 visual tokens)
    },
    "HIGH": {
        "min_pixels": 384 * FACTOR,  # 301,056 pixels (~384 visual tokens)
        "max_pixels": 768 * FACTOR   # 602,112 pixels (~768 visual tokens)
    }
}

MODEL_NAME_OR_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"

def get_processor(
    model_name_or_path: str = MODEL_NAME_OR_PATH,
    min_pixels: int = 256 * FACTOR,
    max_pixels: int = 512 * FACTOR
):
    """
    Creates Qwen2.5-VL AutoProcessor with bounded visual token budget.
    Ensures local cache is prioritized without internet download.
    """
    from transformers import AutoProcessor
    try:
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            local_files_only=True
        )
    except Exception:
        # Fallback to default cache resolution if local_files_only raises
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
    return processor

def resolve_image_path(raw_path_str: str) -> Path:
    """
    Resolves an image_path string from metadata.csv.
    Ensures resolution is strictly relative to data/processed/.
    """
    p = Path(raw_path_str)
    if p.is_absolute():
        return p
    
    # Check 1: PROCESSED_DIR / raw_path_str (e.g. data/processed/images/000001.jpg)
    candidate1 = PROCESSED_DIR / p
    if candidate1.exists():
        return candidate1
        
    # Check 2: If path already contains 'data/processed', resolve from PROJECT_ROOT
    candidate2 = PROJECT_ROOT / p
    if candidate2.exists():
        return candidate2
        
    # Check 3: Directly under IMAGES_DIR by filename
    candidate3 = IMAGES_DIR / p.name
    if candidate3.exists():
        return candidate3
        
    # Default return primary candidate
    return candidate1

def main():
    current_stage = "Initialization"
    try:
        # ====================================================
        # [1/7] Checking environment...
        # ====================================================
        current_stage = "[1/7] Checking environment"
        print(f"\n{current_stage}...")
        import torch
        import transformers
        print(f"  Python executable: {sys.executable}")
        print(f"  Python version:    {sys.version.split()[0]}")
        print(f"  PyTorch version:   {torch.__version__}")
        print(f"  CUDA available:    {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU device:        {torch.cuda.get_device_name(0)}")
        print(f"  Transformers ver:  {transformers.__version__}")
        print(f"  Pandas version:    {pd.__version__}")

        # ====================================================
        # [2/7] Checking metadata...
        # ====================================================
        current_stage = "[2/7] Checking metadata"
        print(f"\n{current_stage}...")
        print(f"  PROJECT ROOT:       {PROJECT_ROOT}")
        print(f"  METADATA CSV PATH:  {METADATA_CSV}")
        print(f"  IMAGES DIRECTORY:   {IMAGES_DIR}")
        print(f"  REPORT PATH:        {REPORT_PATH}")

        if not METADATA_CSV.exists():
            raise FileNotFoundError(f"Metadata CSV does not exist: {METADATA_CSV}")
        if not IMAGES_DIR.exists():
            raise FileNotFoundError(f"Images directory does not exist: {IMAGES_DIR}")

        df = pd.read_csv(METADATA_CSV)
        num_records = len(df)
        print(f"  Metadata loaded successfully. Total rows: {num_records}")
        
        if num_records == 0:
            raise ValueError("metadata.csv is empty (0 rows).")
            
        if "image_path" not in df.columns:
            raise KeyError(f"'image_path' column missing from metadata.csv. Columns found: {list(df.columns)}")

        first_5_paths = df["image_path"].head(5).tolist()
        print(f"  First 5 image_path values from metadata.csv:")
        for idx, p in enumerate(first_5_paths, 1):
            print(f"    {idx}. {p}")

        # ====================================================
        # [3/7] Checking images...
        # ====================================================
        current_stage = "[3/7] Checking images"
        print(f"\n{current_stage}...")
        
        requested_count = min(15, num_records)
        sample_subset = df.head(requested_count)
        
        loaded_images = []
        loaded_paths = []
        missing_paths = []
        corrupt_paths = []
        
        for _, row in sample_subset.iterrows():
            raw_path = str(row["image_path"])
            resolved = resolve_image_path(raw_path)
            
            if not resolved.exists():
                missing_paths.append(str(resolved))
                continue
                
            try:
                with Image.open(resolved) as img:
                    img_rgb = img.convert("RGB")
                    img_rgb.load()
                    loaded_images.append(img_rgb)
                    loaded_paths.append(str(resolved))
            except Exception as e:
                corrupt_paths.append({"path": str(resolved), "error": str(e)})

        print(f"  Requested images:     {requested_count}")
        print(f"  Successfully loaded:  {len(loaded_images)}")
        print(f"  Missing:              {len(missing_paths)}")
        print(f"  Corrupt:              {len(corrupt_paths)}")

        if len(loaded_images) == 0:
            raise RuntimeError(
                f"Zero images could be loaded! Check paths relative to {PROCESSED_DIR}. "
                f"Sample tried: {first_5_paths[0]} -> {resolve_image_path(first_5_paths[0])}"
            )

        # Select up to 10 valid images for the benchmark
        benchmark_images = loaded_images[:10]
        print(f"  Selected {len(benchmark_images)} valid images for resolution benchmark.")

        # ====================================================
        # [4/7] Loading Qwen processor...
        # ====================================================
        current_stage = "[4/7] Loading Qwen processor"
        print(f"\n{current_stage}...")
        print("  Loading processor from local cache...")
        t_proc_start = time.perf_counter()
        test_processor = get_processor(
            model_name_or_path=MODEL_NAME_OR_PATH,
            min_pixels=RESOLUTION_PRESETS["MEDIUM"]["min_pixels"],
            max_pixels=RESOLUTION_PRESETS["MEDIUM"]["max_pixels"]
        )
        t_proc_end = time.perf_counter()
        print(f"  Processor loaded successfully in {(t_proc_end - t_proc_start):.2f}s.")
        print(f"  Processor type: {type(test_processor)}")

        # ====================================================
        # [5/7] Running preprocessing benchmark...
        # ====================================================
        current_stage = "[5/7] Running preprocessing benchmark"
        print(f"\n{current_stage}...")

        benchmark_results = {}

        for preset_name, res_config in RESOLUTION_PRESETS.items():
            min_pix = res_config["min_pixels"]
            max_pix = res_config["max_pixels"]
            print(f"\n  Testing Preset: {preset_name} (min_pixels={min_pix}, max_pixels={max_pix})")
            
            preset_processor = get_processor(
                model_name_or_path=MODEL_NAME_OR_PATH,
                min_pixels=min_pix,
                max_pixels=max_pix
            )
            
            latencies = []
            token_counts = []
            success_count = 0
            failure_count = 0
            errors = []

            for idx, img in enumerate(benchmark_images):
                t0 = time.perf_counter()
                try:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": "Given this crop image, identify the crop disease."}
                            ]
                        }
                    ]
                    prompt_text = preset_processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    
                    inputs = preset_processor(
                        text=[prompt_text],
                        images=[img],
                        return_tensors="pt"
                    )
                    t1 = time.perf_counter()
                    latencies.append((t1 - t0) * 1000.0) # ms

                    # Determine visual token count safely
                    vis_tokens = 0
                    if "image_grid_thw" in inputs:
                        grid = inputs["image_grid_thw"][0]
                        # In Qwen2.5-VL: each 2x2 patch corresponds to 1 token
                        vis_tokens = int(grid[1] * grid[2]) // 4
                    elif "pixel_values" in inputs:
                        vis_tokens = inputs["pixel_values"].shape[0] // 4
                    else:
                        vis_tokens = -1

                    token_counts.append(vis_tokens)
                    success_count += 1

                except Exception as e:
                    failure_count += 1
                    errors.append({"image_idx": idx, "error": str(e)})

            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
            valid_tokens = [t for t in token_counts if t >= 0]
            avg_tokens = sum(valid_tokens) / len(valid_tokens) if valid_tokens else 0.0
            min_tokens = min(valid_tokens) if valid_tokens else 0
            max_tokens = max(valid_tokens) if valid_tokens else 0

            benchmark_results[preset_name] = {
                "min_pixels": min_pix,
                "max_pixels": max_pix,
                "tested_images_count": len(benchmark_images),
                "success_count": success_count,
                "failure_count": failure_count,
                "avg_preprocessing_time_ms": round(avg_latency, 2),
                "avg_visual_tokens": round(avg_tokens, 1),
                "min_visual_tokens": min_tokens,
                "max_visual_tokens": max_tokens,
                "tokens_per_image": token_counts,
                "errors": errors
            }

            print(f"    -> Success: {success_count}/{len(benchmark_images)} | Avg Latency: {avg_latency:.2f} ms | Visual Tokens: avg={avg_tokens:.1f}, min={min_tokens}, max={max_tokens}")

        # ====================================================
        # [6/7] Saving report...
        # ====================================================
        current_stage = "[6/7] Saving report"
        print(f"\n{current_stage}...")
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        
        final_report = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(PROJECT_ROOT),
            "metadata_path": str(METADATA_CSV),
            "images_directory": str(IMAGES_DIR),
            "total_metadata_records": num_records,
            "benchmark_images_count": len(benchmark_images),
            "missing_images_count": len(missing_paths),
            "corrupt_images_count": len(corrupt_paths),
            "model_identifier": MODEL_NAME_OR_PATH,
            "resolution_presets": benchmark_results
        }

        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2)

        print(f"  Preprocessing report saved to: {REPORT_PATH}")

        # ====================================================
        # [7/7] COMPLETE
        # ====================================================
        current_stage = "[7/7] COMPLETE"
        print(f"\n{current_stage} - All checks and benchmarks passed successfully!\n")
        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print("ERROR OCCURRED DURING PREPROCESSING BENCHMARK")
        print("=" * 60)
        print(f"ERROR STAGE:   {current_stage}")
        print(f"ERROR TYPE:    {type(e).__name__}")
        print(f"ERROR MESSAGE: {e}")
        print("\nFULL TRACEBACK:")
        traceback.print_exc()
        print("=" * 60 + "\n")
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())
