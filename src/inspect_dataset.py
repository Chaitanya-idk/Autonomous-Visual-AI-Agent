"""
Inspection script for SAGE-sample dataset.
Reads the Hugging Face dataset in read-only mode and computes schema and statistics.
"""

import json
import os
import hashlib
from collections import Counter
from datasets import load_from_disk
import numpy as np

def inspect_dataset(dataset_path: str, output_dirs: list):
    print(f"Loading dataset from: {dataset_path}...")
    ds_dict = load_from_disk(dataset_path)
    
    splits = list(ds_dict.keys())
    print(f"Available splits: {splits}")
    
    # We will analyze each split (here 'train')
    all_split_stats = {}
    for split_name in splits:
        split = ds_dict[split_name]
        total_records = len(split)
        features = list(split.features.keys())
        print(f"Split '{split_name}': {total_records} records, features: {features}")
        
        # Analyze columns
        crops = []
        diseases = []
        filenames = []
        image_sizes = []
        image_modes = []
        image_formats = []
        byte_hashes = []
        corrupted_count = 0
        missing_count = 0
        
        for idx in range(total_records):
            item = split[idx]
            crops.append(item.get("crop"))
            diseases.append(item.get("disease"))
            filenames.append(item.get("filename"))
            
            img = item.get("image")
            if img is None:
                missing_count += 1
                continue
            try:
                image_sizes.append(img.size)
                image_modes.append(img.mode)
                image_formats.append(img.format if img.format else "UNKNOWN")
                h = hashlib.md5(img.tobytes()).hexdigest()
                byte_hashes.append(h)
            except Exception as e:
                corrupted_count += 1
        
        crop_counts = dict(Counter(crops))
        disease_counts = dict(Counter(diseases))
        unique_filenames = len(set(filenames))
        unique_byte_hashes = len(set(byte_hashes))
        
        widths = [s[0] for s in image_sizes]
        heights = [s[1] for s in image_sizes]
        
        # Rare class analysis
        classes_with_1 = [k for k, v in disease_counts.items() if v == 1]
        classes_with_2 = [k for k, v in disease_counts.items() if v == 2]
        classes_with_lt5 = [k for k, v in disease_counts.items() if v < 5]
        classes_with_gte5 = [k for k, v in disease_counts.items() if v >= 5]
        
        split_stats = {
            "split_name": split_name,
            "total_records": total_records,
            "features": features,
            "image_field": "image",
            "disease_field": "disease",
            "crop_field": "crop",
            "filename_field": "filename",
            "missing_records": missing_count,
            "corrupted_records": corrupted_count,
            "num_crops": len(crop_counts),
            "num_diseases": len(disease_counts),
            "unique_filenames": unique_filenames,
            "unique_image_hashes": unique_byte_hashes,
            "duplicate_filenames_count": total_records - unique_filenames,
            "duplicate_image_hashes_count": total_records - unique_byte_hashes,
            "image_stats": {
                "width_min": int(np.min(widths)) if widths else 0,
                "width_max": int(np.max(widths)) if widths else 0,
                "width_mean": float(np.mean(widths)) if widths else 0,
                "height_min": int(np.min(heights)) if heights else 0,
                "height_max": int(np.max(heights)) if heights else 0,
                "height_mean": float(np.mean(heights)) if heights else 0,
                "modes": dict(Counter(image_modes)),
                "formats": dict(Counter(image_formats)),
                "top_10_resolutions": Counter(image_sizes).most_common(10)
            },
            "crops_distribution": crop_counts,
            "disease_class_distribution": disease_counts,
            "rare_classes": {
                "count_with_1_sample": len(classes_with_1),
                "count_with_2_samples": len(classes_with_2),
                "count_with_less_than_5_samples": len(classes_with_lt5),
                "count_with_5_or_more_samples": len(classes_with_gte5),
                "classes_with_1_sample": sorted(classes_with_1),
                "classes_with_2_samples": sorted(classes_with_2)
            }
        }
        all_split_stats[split_name] = split_stats

    inspection_data = {
        "dataset_name": "SAGE-sample",
        "dataset_path": os.path.abspath(dataset_path),
        "splits": splits,
        "split_stats": all_split_stats
    }
    
    # Write JSON and TXT reports
    for out_dir in output_dirs:
        os.makedirs(out_dir, exist_ok=True)
        json_path = os.path.join(out_dir, "dataset_inspection.json")
        txt_path = os.path.join(out_dir, "dataset_inspection.txt")
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(inspection_data, f, indent=2)
            
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("="*60 + "\n")
            f.write("SAGE-SAMPLE DATASET INSPECTION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Dataset Path: {dataset_path}\n")
            f.write(f"Splits: {splits}\n\n")
            for split_name, stats in all_split_stats.items():
                f.write(f"--- SPLIT: {split_name} ---\n")
                f.write(f"Total records: {stats['total_records']}\n")
                f.write(f"Features: {stats['features']}\n")
                f.write(f"Missing images: {stats['missing_records']}\n")
                f.write(f"Corrupted images: {stats['corrupted_records']}\n")
                f.write(f"Unique Crops: {stats['num_crops']}\n")
                f.write(f"Unique Diseases: {stats['num_diseases']}\n")
                f.write(f"Unique Filenames: {stats['unique_filenames']} (Duplicates: {stats['duplicate_filenames_count']})\n")
                f.write(f"Unique Image Hashes: {stats['unique_image_hashes']} (Duplicates: {stats['duplicate_image_hashes_count']})\n")
                f.write("\nImage Dimensions:\n")
                f.write(f"  Width:  min={stats['image_stats']['width_min']}, max={stats['image_stats']['width_max']}, mean={stats['image_stats']['width_mean']:.1f}\n")
                f.write(f"  Height: min={stats['image_stats']['height_min']}, max={stats['image_stats']['height_max']}, mean={stats['image_stats']['height_mean']:.1f}\n")
                f.write(f"  Modes: {stats['image_stats']['modes']}\n")
                f.write(f"  Formats: {stats['image_stats']['formats']}\n")
                f.write(f"  Top 10 resolutions: {stats['image_stats']['top_10_resolutions']}\n\n")
                f.write("Rare Classes Summary:\n")
                f.write(f"  Classes with 1 sample: {stats['rare_classes']['count_with_1_sample']}\n")
                f.write(f"  Classes with 2 samples: {stats['rare_classes']['count_with_2_samples']}\n")
                f.write(f"  Classes with <5 samples: {stats['rare_classes']['count_with_less_than_5_samples']}\n")
                f.write(f"  Classes with >=5 samples: {stats['rare_classes']['count_with_5_or_more_samples']}\n\n")
                f.write("Crops breakdown:\n")
                for c, count in sorted(stats['crops_distribution'].items(), key=lambda x: -x[1]):
                    f.write(f"  - {c}: {count}\n")
                f.write("\nTop 15 Most Common Diseases:\n")
                top15 = sorted(stats['disease_class_distribution'].items(), key=lambda x: -x[1])[:15]
                for d, count in top15:
                    f.write(f"  - {d}: {count}\n")
                f.write("\n")
                
        print(f"Saved inspection reports to: {json_path} and {txt_path}")

if __name__ == "__main__":
    raw_ds_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SAGE-sample")
    reports_dirs = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "reports")
    ]
    inspect_dataset(raw_ds_path, reports_dirs)
