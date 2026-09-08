"""
Splitting script for SAGE-sample dataset.
Creates train.csv, val.csv, test.csv, authoritative labels.json, and data/reports/split_report.json.
Handles rare classes deterministically and mathematically soundly.
"""

import os
import csv
import json
import random
from collections import Counter, defaultdict

def prepare_splits(
    metadata_csv_path: str,
    processed_dir: str,
    reports_dir: str,
    seed: int = 42
):
    print("=" * 60)
    print("PREPARING DATASET SPLITS & AUTHORITATIVE LABELS")
    print("=" * 60)
    
    random.seed(seed)
    
    records = []
    with open(metadata_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            records.append(r)
            
    total_records = len(records)
    print(f"Loaded {total_records} records.")
    
    # 1. Authoritative label vocabulary
    all_diseases = sorted(list(set(r["disease"].strip() for r in records if r["disease"].strip())))
    label2id = {name: idx for idx, name in enumerate(all_diseases)}
    id2label = {str(idx): name for idx, name in enumerate(all_diseases)}
    
    labels_path = os.path.join(processed_dir, "labels.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)
    print(f"Authoritative labels.json saved to: {labels_path} (Total classes: {len(all_diseases)})")
    
    # Group records by disease
    class_to_records = defaultdict(list)
    for r in records:
        r["label"] = r["disease"].strip()
        r["label_id"] = label2id[r["label"]]
        class_to_records[r["label"]].append(r)
        
    train_records = []
    val_records = []
    test_records = []
    
    rare_class_handling = {
        "1_sample_classes": [],
        "2_samples_classes": [],
        "3_samples_classes": [],
        "4_samples_classes": [],
        "5_or_more_samples_classes": []
    }
    
    # Deterministic split per class
    for disease_name, items in sorted(class_to_records.items()):
        # Deterministic shuffle with seed derived from global seed and class name
        cls_rng = random.Random(f"{seed}_{disease_name}")
        shuffled = items.copy()
        cls_rng.shuffle(shuffled)
        
        n = len(shuffled)
        if n == 1:
            # 1 sample: must go to train
            train_records.extend(shuffled)
            rare_class_handling["1_sample_classes"].append(disease_name)
        elif n == 2:
            # 2 samples: 1 train, 1 val
            train_records.append(shuffled[0])
            val_records.append(shuffled[1])
            rare_class_handling["2_samples_classes"].append(disease_name)
        elif n == 3:
            # 3 samples: 2 train, 1 val
            train_records.extend(shuffled[:2])
            val_records.append(shuffled[2])
            rare_class_handling["3_samples_classes"].append(disease_name)
        elif n == 4:
            # 4 samples: 2 train, 1 val, 1 test
            train_records.extend(shuffled[:2])
            val_records.append(shuffled[2])
            test_records.append(shuffled[3])
            rare_class_handling["4_samples_classes"].append(disease_name)
        else:
            # n >= 5: Target ~80% train, ~10% val, ~10% test
            n_test = max(1, round(n * 0.10))
            n_val = max(1, round(n * 0.10))
            n_train = n - n_val - n_test
            
            # Ensure at least 1 in train
            if n_train < 1:
                n_train = 1
                if n_test > 1:
                    n_test -= 1
                elif n_val > 1:
                    n_val -= 1
                    
            train_records.extend(shuffled[:n_train])
            val_records.extend(shuffled[n_train:n_train + n_val])
            test_records.extend(shuffled[n_train + n_val:])
            rare_class_handling["5_or_more_samples_classes"].append(disease_name)

    # Verification: check disjointness and total
    all_assigned_ids = set()
    for name, split_recs in [("train", train_records), ("val", val_records), ("test", test_records)]:
        split_ids = set(r["image_id"] for r in split_recs)
        if not split_ids.isdisjoint(all_assigned_ids):
            overlap = split_ids.intersection(all_assigned_ids)
            raise ValueError(f"FATAL: Overlap detected in {name} split! Overlapping IDs: {overlap}")
        all_assigned_ids.update(split_ids)
        
    assert len(all_assigned_ids) == total_records, f"Mismatch: {len(all_assigned_ids)} assigned vs {total_records} total"
    
    # Mark split in records
    for r in train_records:
        r["split"] = "train"
    for r in val_records:
        r["split"] = "val"
    for r in test_records:
        r["split"] = "test"
        
    # Write updated metadata.csv
    updated_records = sorted(train_records + val_records + test_records, key=lambda x: int(x["image_id"]))
    fieldnames = [
        "image_id", "image_path", "crop", "disease", "label", "label_id", "split",
        "original_filename", "original_width", "original_height",
        "original_mode", "original_format"
    ]
    with open(metadata_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_records)
        
    # Write train.csv, val.csv, test.csv
    split_files = {
        "train": os.path.join(processed_dir, "train.csv"),
        "val": os.path.join(processed_dir, "val.csv"),
        "test": os.path.join(processed_dir, "test.csv")
    }
    for s_name, s_records in [("train", train_records), ("val", val_records), ("test", test_records)]:
        path = split_files[s_name]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(s_records)
        print(f"Saved {s_name}.csv: {len(s_records)} samples")
        
    # Split report
    train_dist = dict(Counter(r["label"] for r in train_records))
    val_dist = dict(Counter(r["label"] for r in val_records))
    test_dist = dict(Counter(r["label"] for r in test_records))
    
    split_report = {
        "random_seed": seed,
        "total_samples": total_records,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "test_samples": len(test_records),
        "split_percentages": {
            "train": round(len(train_records) / total_records * 100, 2),
            "val": round(len(val_records) / total_records * 100, 2),
            "test": round(len(test_records) / total_records * 100, 2)
        },
        "total_classes": len(all_diseases),
        "classes_in_train": len(train_dist),
        "classes_in_val": len(val_dist),
        "classes_in_test": len(test_dist),
        "rare_classes_handling": {
            "1_sample_classes_count": len(rare_class_handling["1_sample_classes"]),
            "1_sample_policy": "Placed in train set only (model must observe rare patterns)",
            "2_sample_classes_count": len(rare_class_handling["2_samples_classes"]),
            "2_sample_policy": "Split 1 in train, 1 in val",
            "3_sample_classes_count": len(rare_class_handling["3_samples_classes"]),
            "3_sample_policy": "Split 2 in train, 1 in val",
            "4_sample_classes_count": len(rare_class_handling["4_samples_classes"]),
            "4_sample_policy": "Split 2 in train, 1 in val, 1 in test",
            "5_plus_classes_count": len(rare_class_handling["5_or_more_samples_classes"]),
            "5_plus_policy": "Stratified 80% train, 10% val, 10% test"
        },
        "distributions": {
            "train": train_dist,
            "val": val_dist,
            "test": test_dist
        }
    }
    
    report_json_path = os.path.join(reports_dir, "split_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(split_report, f, indent=2)
        
    print(f"Split report saved to: {report_json_path}")
    print("=" * 60)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    meta_csv = os.path.join(base_dir, "data", "processed", "metadata.csv")
    proc_dir = os.path.join(base_dir, "data", "processed")
    rep_dir = os.path.join(base_dir, "data", "reports")
    
    prepare_splits(meta_csv, proc_dir, rep_dir, seed=42)
