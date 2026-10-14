#!/usr/bin/env python3
"""03_coco_ke_yolo.py
Mengonversi format anotasi COCO (.json) kembali ke format YOLO (.txt per citra).
Digunakan untuk menguji konversi dua arah (round-trip test) antara YOLO dan COCO.
Sesuai materi Slide 5 & Slide 35 RET503.
Penggunaan:
    python 03_coco_ke_yolo.py dataset_yolo/annotations_coco.json hasil_balik
"""

import sys
import json
import argparse
from pathlib import Path


def coco_to_yolo(coco_json_path: Path, output_dir: Path):
    if not coco_json_path.exists():
        print(f"Error: File '{coco_json_path}' tidak ditemukan!")
        sys.exit(1)

    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Petakan image_id ke informasi citra
    images_info = {img["id"]: img for img in coco.get("images", [])}

    # Kelompokkan anotasi per image_id
    img_annotations = {}
    for ann in coco.get("annotations", []):
        img_id = ann["image_id"]
        img_annotations.setdefault(img_id, []).append(ann)

    converted_files = 0
    total_boxes = 0

    for img_id, img_info in images_info.items():
        width = img_info["width"]
        height = img_info["height"]
        stem = Path(img_info["file_name"]).stem
        label_file = output_dir / f"{stem}.txt"

        lines = []
        for ann in img_annotations.get(img_id, []):
            cls_id = ann["category_id"]
            x_min, y_min, box_w, box_h = ann["bbox"]

            # Konversi dari piksel (COCO) ke normalisasi YOLO [0, 1]
            cx = (x_min + box_w / 2.0) / width
            cy = (y_min + box_h / 2.0) / height
            nw = box_w / width
            nh = box_h / height

            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
            total_boxes += 1

        with open(label_file, "w", encoding="utf-8") as f:
            f.writelines(lines)
        converted_files += 1

    print("=" * 60)
    print(" KONVERSI FORMAT COCO JSON KE YOLO (ROUND-TRIP) BERHASIL")
    print("=" * 60)
    print(f" File Sumber COCO: {coco_json_path}")
    print(f" Folder Hasil YOLO: {output_dir}")
    print(f" Total File Label: {converted_files}")
    print(f" Total Kotak     : {total_boxes}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Konversi anotasi format COCO JSON kembali ke format YOLO")
    parser.add_argument("coco_json", type=Path, nargs="?", default=Path("dataset_yolo/annotations_coco.json"), help="File input .json format COCO")
    parser.add_argument("output_dir", type=Path, nargs="?", default=Path("hasil_balik"), help="Folder output file label YOLO")
    args = parser.parse_args()

    coco_to_yolo(args.coco_json, args.output_dir)


if __name__ == "__main__":
    main()
