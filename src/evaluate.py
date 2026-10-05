"""
Full generation-based evaluation for the SAGE crop-disease LoRA model.

The model receives the image and the canonical diagnosis prompt only.
The ground-truth disease is never placed in the generation prompt.

The evaluator is intentionally strict:
- invalid ground-truth labels are fatal
- unknown predictions are counted explicitly
- raw model generations are saved
- empty evaluation sets are fatal
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import torch
import yaml
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import SAGEDataset, sage_collate_fn
from src.model import load_trained_lora_model
from src.utils import (
    compute_classification_metrics,
    load_labels_vocab,
    match_prediction_to_vocab,
)
from src.train import download_shard
from src.preprocessing_lite import get_processor


def load_config(path: str = None):
    cfg_path = (
        Path(path)
        if path
        else PROJECT_ROOT / "configs" / "config.yaml"
    )

    with open(
        cfg_path,
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def evaluate(
    cfg: Dict[str, Any],
    checkpoint: str,
    max_samples: int = None,
):
    ds_cfg = cfg["dataset"]

    common = dict(
        image_col=ds_cfg.get(
            "image_col",
            "image",
        ),
        disease_col=ds_cfg.get(
            "disease_col",
            "disease",
        ),
        crop_col=ds_cfg.get(
            "crop_col",
            "crop",
        ),
        label_id_col=ds_cfg.get(
            "label_id_col",
            "",
        ),
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Full evaluation requires CUDA."
        )

    dtype_name = str(
        cfg["model"].get(
            "torch_dtype",
            "float16",
        )
    ).lower()

    dtype = (
        torch.bfloat16
        if dtype_name in {"bfloat16", "bf16"}
        else torch.float16
    )

    device = torch.device("cuda")

    # -------------------------------------------------------------------------
    # Processor
    # -------------------------------------------------------------------------
    prep_cfg = cfg.get(
        "preprocessing",
        {},
    )

    processor = get_processor(
        model_name_or_path=cfg["model"]["name_or_path"],
        local_files_only=cfg["model"].get(
            "local_files_only",
            False,
        ),
        min_pixels=prep_cfg.get("min_pixels"),
        max_pixels=prep_cfg.get("max_pixels"),
    )

    # -------------------------------------------------------------------------
    # Labels
    # -------------------------------------------------------------------------
    labels_path = Path(
        cfg["paths"].get(
            "labels_json",
            "/kaggle/working/labels.json",
        )
    )

    label2id, id2label = load_labels_vocab(
        labels_path
    )

    print(
        f"[Evaluation] Loaded {len(label2id)} disease classes."
    )

    # -------------------------------------------------------------------------
    # LoRA checkpoint
    # -------------------------------------------------------------------------
    adapter_path = Path(checkpoint)

    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {adapter_path}"
        )

    model = load_trained_lora_model(
        adapter_path=str(adapter_path),
        base_model_name_or_path=cfg["model"][
            "name_or_path"
        ],
        torch_dtype=cfg["model"].get(
            "torch_dtype",
            "float16",
        ),
        local_files_only=cfg["model"].get(
            "local_files_only",
            False,
        ),
        merge_weights=False,
    )

    model.eval()

    # -------------------------------------------------------------------------
    # Validation shard
    # -------------------------------------------------------------------------
    chunk_dir = Path(
        ds_cfg.get(
            "chunk_dir",
            "/kaggle/working/data_cache",
        )
    )

    val_dir = chunk_dir / "val"
    val_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    val_shard = (
        int(
            ds_cfg.get(
                "total_shards",
                48,
            )
        )
        - 1
    )

    token = os.environ.get(
        "HF_TOKEN"
    )

    download_shard(
        ds_cfg["hf_repo_id"],
        val_shard,
        val_dir,
        token=token,
    )

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    val_ds = SAGEDataset(
        parquet_dir=str(val_dir),
        processor=processor,
        is_training=False,
        label2id=label2id,
        **common,
    )

    if len(val_ds) == 0:
        raise RuntimeError(
            "Validation dataset contains zero samples."
        )

    if max_samples is not None:
        max_samples = min(
            int(max_samples),
            len(val_ds),
        )

        from torch.utils.data import Subset

        val_ds = Subset(
            val_ds,
            list(range(max_samples)),
        )

    loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=sage_collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    y_true: List[int] = []
    y_pred: List[int] = []
    rows: List[Dict[str, Any]] = []

    unknown_count = 0

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------
    with torch.no_grad():
        for sample_index, batch in enumerate(
            tqdm(
                loader,
                desc="Full evaluation",
                unit="image",
                dynamic_ncols=True,
            )
        ):
            ids = batch["input_ids"].to(
                device,
                non_blocking=True,
            )

            attn = batch["attention_mask"].to(
                device,
                non_blocking=True,
            )

            pv = batch["pixel_values"].to(
                device=device,
                dtype=dtype,
                non_blocking=True,
            )

            thw = batch["image_grid_thw"].to(
                device,
                non_blocking=True,
            )

            true_id = int(
                batch["label_id"][0].item()
            )

            true_label = str(
                batch["disease"][0]
            ).strip()

            # Ground truth is never allowed to be -1.
            if true_id < 0:
                raise RuntimeError(
                    "INVALID GROUND-TRUTH LABEL\n"
                    f"sample={sample_index}\n"
                    f"disease={true_label!r}\n"
                    f"label_id={true_id}\n"
                    "Check labels.json against the validation shard."
                )

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=attn,
                pixel_values=pv,
                image_grid_thw=thw,
                max_new_tokens=32,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )

            prompt_len = ids.shape[1]

            raw = processor.tokenizer.decode(
                generated_ids[
                    0,
                    prompt_len:,
                ],
                skip_special_tokens=True,
            ).strip()

            (
                pred_label,
                pred_id,
                status,
            ) = match_prediction_to_vocab(
                raw,
                label2id,
            )

            if status == "unknown":
                unknown_count += 1

            correct = (
                true_id == pred_id
            )

            y_true.append(true_id)
            y_pred.append(pred_id)

            row = {
                "index": sample_index,
                "true_label": true_label,
                "true_id": true_id,
                "predicted_label": pred_label,
                "predicted_id": pred_id,
                "status": status,
                "correct": bool(correct),
                "raw_model_output": raw,
            }

            rows.append(row)

            # Always print the first 20 samples. This is the most important
            # diagnostic when metrics unexpectedly become zero.
            if sample_index < 20:
                print(
                    "\n"
                    + "=" * 72
                )
                print(
                    f"SAMPLE {sample_index}"
                )
                print(
                    "=" * 72
                )
                print(
                    f"TRUE LABEL : {true_label}"
                )
                print(
                    f"TRUE ID    : {true_id}"
                )
                print(
                    f"RAW OUTPUT : {raw!r}"
                )
                print(
                    f"PRED LABEL : {pred_label}"
                )
                print(
                    f"PRED ID    : {pred_id}"
                )
                print(
                    f"STATUS     : {status}"
                )
                print(
                    f"CORRECT    : {correct}"
                )

    # -------------------------------------------------------------------------
    # Hard sanity checks
    # -------------------------------------------------------------------------
    if not y_true:
        raise RuntimeError(
            "No evaluation samples were processed. "
            "Refusing to report zero metrics."
        )

    if len(y_true) != len(y_pred):
        raise RuntimeError(
            f"Metric arrays have different lengths: "
            f"y_true={len(y_true)}, y_pred={len(y_pred)}"
        )

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------
    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        labels_list=list(
            label2id.values()
        ),
    )

    metrics.update(
        {
            "num_samples": len(y_true),
            "unknown_predictions": unknown_count,
            "unknown_rate": (
                unknown_count / len(y_true)
            ),
            "checkpoint": str(
                adapter_path
            ),
            "validation_shard": val_shard,
        }
    )

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    output_dir = Path(
        cfg["paths"]["output_dir"]
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        output_dir
        / "final_evaluation_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
            ensure_ascii=False,
        )

    predictions_path = (
        output_dir
        / "predictions.csv"
    )

    with open(
        predictions_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        fieldnames = [
            "index",
            "true_label",
            "true_id",
            "predicted_label",
            "predicted_id",
            "status",
            "correct",
            "raw_model_output",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        "\n"
        + "=" * 72
    )
    print(
        "FINAL EVALUATION"
    )
    print(
        "=" * 72
    )

    for key, value in metrics.items():
        print(
            f"{key}: {value}"
        )

    print(
        f"\nMetrics     -> {metrics_path}"
    )
    print(
        f"Predictions -> {predictions_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    evaluate(
        load_config(args.config),
        args.checkpoint,
        args.max_samples,
    )
