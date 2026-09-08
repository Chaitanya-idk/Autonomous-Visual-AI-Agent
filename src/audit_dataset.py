"""
Dataset Quality Audit Script for SAGE-sample.
Performs thorough audit on the extracted images and metadata.csv.
Outputs data/reports/dataset_audit.json and prints human-readable summary.
"""

import os
import csv
import json
import hashlib
from collections import Counter
from PIL import Image
import numpy as np

def audit_dataset(
    metadata_csv_path: str,
    base_dir: str,
    output_report_json: str
):
    print("=" * 60)
    print("STARTING DATASET QUALITY AUDIT")
    print("=" * 60)
    
    if not os.path.exists(metadata_csv_path):
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv_path}")
        
    records = []
    with open(metadata_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            records.append(r)
            
    total_records = len(records)
    print(f"Loaded {total_records} records from {metadata_csv_path}")
    
    missing_images = []
    unreadable_images = []
    dimension_mismatches = []
    empty_labels = []
    empty_crops = []
    
    file_hashes = {} # hash -> list of image_ids
    actual_dimensions = []
    extreme_aspect_ratios = []
    
    for r in records:
        img_id = r["image_id"]
        rel_path = r["image_path"]
        abs_path = os.path.join(base_dir, rel_path)
        crop = r.get("crop", "").strip()
        disease = r.get("disease", "").strip()
        
        if not crop:
            empty_crops.append(img_id)
        if not disease:
            empty_labels.append(img_id)
            
        if not os.path.exists(abs_path):
            missing_images.append({
                "image_id": img_id,
                "path": rel_path
            })
            continue
            
        # Try opening and verifying image
        try:
            with Image.open(abs_path) as img:
                img.verify()
            # Open again to read properties (verify can invalidate the object)
            with Image.open(abs_path) as img:
                w, h = img.size
                actual_dimensions.append((w, h))
                aspect_ratio = max(w / h, h / w)
                if aspect_ratio > 3.0:
                    extreme_aspect_ratios.append({
                        "image_id": img_id,
                        "width": w,
                        "height": h,
                        "aspect_ratio": round(aspect_ratio, 2)
                    })
                
                # Check byte hash
                with open(abs_path, "rb") as bf:
                    hval = hashlib.md5(bf.read()).hexdigest()
                    if hval not in file_hashes:
                        file_hashes[hval] = []
                    file_hashes[hval].append(img_id)
                    
        except Exception as e:
            unreadable_images.append({
                "image_id": img_id,
                "path": rel_path,
                "error": str(e)
            })
            
    # Duplicate image files analysis
    duplicates_groups = {k: v for k, v in file_hashes.items() if len(v) > 1}
    total_duplicate_files = sum(len(v) - 1 for v in duplicates_groups.values())
    
    # Class distribution analysis
    diseases = [r["disease"].strip() for r in records if r["disease"].strip()]
    disease_counts = Counter(diseases)
    
    crops = [r["crop"].strip() for r in records if r["crop"].strip()]
    crop_counts = Counter(crops)
    
    classes_with_1 = [k for k, v in disease_counts.items() if v == 1]
    classes_with_2 = [k for k, v in disease_counts.items() if v == 2]
    classes_with_lt5 = [k for k, v in disease_counts.items() if v < 5]
    classes_with_gte5 = [k for k, v in disease_counts.items() if v >= 5]
    
    widths = [dim[0] for dim in actual_dimensions]
    heights = [dim[1] for dim in actual_dimensions]
    
    audit_report = {
        "metadata_csv": os.path.abspath(metadata_csv_path),
        "total_records": total_records,
        "valid_images": len(actual_dimensions),
        "missing_images_count": len(missing_images),
        "missing_images": missing_images,
        "unreadable_images_count": len(unreadable_images),
        "unreadable_images": unreadable_images,
        "empty_labels_count": len(empty_labels),
        "empty_labels": empty_labels,
        "empty_crops_count": len(empty_crops),
        "empty_crops": empty_crops,
        "duplicate_image_content": {
            "duplicate_groups_count": len(duplicates_groups),
            "redundant_images_count": total_duplicate_files,
            "groups": duplicates_groups
        },
        "dimension_statistics": {
            "width_min": int(np.min(widths)) if widths else 0,
            "width_max": int(np.max(widths)) if widths else 0,
            "width_mean": float(np.mean(widths)) if widths else 0,
            "height_min": int(np.min(heights)) if heights else 0,
            "height_max": int(np.max(heights)) if heights else 0,
            "height_mean": float(np.mean(heights)) if heights else 0,
            "extreme_aspect_ratios_count": len(extreme_aspect_ratios),
            "extreme_aspect_ratios_sample": extreme_aspect_ratios[:10]
        },
        "classes_statistics": {
            "num_crops": len(crop_counts),
            "num_unique_diseases": len(disease_counts),
            "classes_with_1_sample_count": len(classes_with_1),
            "classes_with_2_samples_count": len(classes_with_2),
            "classes_with_less_than_5_count": len(classes_with_lt5),
            "classes_with_5_or_more_count": len(classes_with_gte5),
            "classes_with_1_sample": sorted(classes_with_1),
            "classes_with_2_samples": sorted(classes_with_2),
            "crop_counts": dict(crop_counts),
            "top_10_diseases": disease_counts.most_common(10)
        }
    }
    
    os.makedirs(os.path.dirname(output_report_json), exist_ok=True)
    with open(output_report_json, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)
        
    print("\n--- AUDIT SUMMARY ---")
    print(f"Total records in metadata: {total_records}")
    print(f"Missing images on disk:    {len(missing_images)}")
    print(f"Unreadable images:         {len(unreadable_images)}")
    print(f"Empty labels (disease):    {len(empty_labels)}")
    print(f"Empty crops:               {len(empty_crops)}")
    print(f"Duplicate image contents:  {len(duplicates_groups)} groups ({total_duplicate_files} duplicate images)")
    print(f"Extreme aspect ratios (>3): {len(extreme_aspect_ratios)}")
    print(f"Total disease classes:     {len(disease_counts)}")
    print(f"Classes with 1 sample:     {len(classes_with_1)}")
    print(f"Classes with 2 samples:    {len(classes_with_2)}")
    print(f"Classes with >= 5 samples: {len(classes_with_gte5)}")
    print(f"Audit report saved to:     {output_report_json}")
    print("=" * 60)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    meta_csv = os.path.join(base_dir, "data", "processed", "metadata.csv")
    report_json = os.path.join(base_dir, "data", "reports", "dataset_audit.json")
    
    audit_dataset(meta_csv, base_dir, report_json)
