"""
Extraction script for SAGE-sample dataset.
Extracts images into data/processed/images/{id:06d}.jpg and builds metadata.csv.
The source dataset is strictly read-only.
"""

import os
import csv
import json
from datasets import load_from_disk
from PIL import Image

def extract_dataset(
    source_path: str,
    output_images_dir: str,
    output_metadata_csv: str,
    report_path: str
):
    print(f"Reading raw dataset from: {source_path}")
    ds_dict = load_from_disk(source_path)
    split = ds_dict["train"]
    total = len(split)
    print(f"Total records to extract: {total}")
    
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_metadata_csv), exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    extracted_rows = []
    failed_records = []
    
    for idx in range(total):
        item = split[idx]
        image_id = idx + 1
        img_filename = f"{image_id:06d}.jpg"
        img_dest_path = os.path.join(output_images_dir, img_filename)
        rel_img_path = os.path.join("data", "processed", "images", img_filename).replace("\\", "/")
        
        crop = item.get("crop", "")
        disease = item.get("disease", "")
        orig_filename = item.get("filename", "")
        
        raw_img = item.get("image")
        if raw_img is None:
            failed_records.append({
                "image_id": image_id,
                "reason": "Image object is None",
                "original_filename": orig_filename
            })
            continue
            
        try:
            orig_w, orig_h = raw_img.size
            orig_mode = raw_img.mode
            orig_fmt = raw_img.format if raw_img.format else "UNKNOWN"
            
            # Ensure RGB
            if raw_img.mode != "RGB":
                save_img = raw_img.convert("RGB")
            else:
                save_img = raw_img
                
            save_img.save(img_dest_path, format="JPEG", quality=95)
            
            extracted_rows.append({
                "image_id": image_id,
                "image_path": rel_img_path,
                "crop": crop,
                "disease": disease,
                "original_filename": orig_filename,
                "original_width": orig_w,
                "original_height": orig_h,
                "original_mode": orig_mode,
                "original_format": orig_fmt
            })
            
        except Exception as e:
            failed_records.append({
                "image_id": image_id,
                "reason": str(e),
                "original_filename": orig_filename
            })
            
        if (idx + 1) % 400 == 0 or (idx + 1) == total:
            print(f"Extracted {idx + 1}/{total} images...")
            
    # Write metadata.csv
    fieldnames = [
        "image_id", "image_path", "crop", "disease",
        "original_filename", "original_width", "original_height",
        "original_mode", "original_format"
    ]
    with open(output_metadata_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(extracted_rows)
        
    # Write extraction report
    extraction_report = {
        "source_path": os.path.abspath(source_path),
        "total_records": total,
        "successfully_extracted": len(extracted_rows),
        "failed_records_count": len(failed_records),
        "failed_records": failed_records,
        "metadata_csv": os.path.abspath(output_metadata_csv),
        "images_directory": os.path.abspath(output_images_dir)
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(extraction_report, f, indent=2)
        
    print(f"\nExtraction complete!")
    print(f"Successfully saved {len(extracted_rows)} images to {output_images_dir}")
    print(f"Metadata saved to {output_metadata_csv}")
    print(f"Report saved to {report_path}")
    if failed_records:
        print(f"WARNING: {len(failed_records)} records failed extraction!")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(base_dir, "SAGE-sample")
    out_img_dir = os.path.join(base_dir, "data", "processed", "images")
    out_meta_csv = os.path.join(base_dir, "data", "processed", "metadata.csv")
    out_report = os.path.join(base_dir, "data", "reports", "extraction_report.json")
    
    extract_dataset(src_dir, out_img_dir, out_meta_csv, out_report)
