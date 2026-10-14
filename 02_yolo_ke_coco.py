#!/usr/bin/env python3
"""02_yolo_ke_coco.py
Mengonversi format anotasi YOLO (.txt per citra) ke format standar COCO (.json tunggal).
Sesuai materi Slide 5 & Slide 35 RET503.
Penggunaan:
    python 02_yolo_ke_coco.py dataset_yolo dataset_yolo/annotations_coco.json
"""

import sys
import json
import argparse
from pathlib import Path
from PIL import Image

CLASS_NAMES = {0: "box_cokelat", 1: "box_merah"}


def yolo_to_coco(yolo_dir: Path, output_json: Path):
    images_dir = yolo_dir / "images"
    labels_dir = yolo_dir / "labels"

    if not images_dir.is_dir() or not labels_dir.is_dir():
        print(f"Error: Folder '{images_dir}' atau '{labels_dir}' tidak ditemukan!")
        sys.exit(1)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in image_exts])

    coco = {
        "info": {
            "description": "Dataset Deteksi Objek Box Industri RET503",
            "version": "1.0",
            "year": 2026,
            "contributor": "Teknologi Rekayasa Robotika - Politeknik Negeri Batam"
        },
        "licenses": [{"id": 1, "name": "Academic Use Only"}],
        "categories": [
            {"id": cid, "name": cname, "supercategory": "industrial_box"}
            for cid, cname in sorted(CLASS_NAMES.items())
        ],
        "images": [],
        "annotations": []
    }

    ann_id = 1
    for img_id, img_path in enumerate(img_files, start=1):
        with Image.open(img_path) as im:
            width, height = im.size

        coco["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": width,
            "height": height
        })

        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                cls_id = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                nw = float(parts[3])
                nh = float(parts[4])

                # Konversi dari YOLO (cx, cy, nw, nh normalisasi) ke COCO (x_min, y_min, w, h piksel)
                box_w = nw * width
                box_h = nh * height
                x_min = (cx * width) - (box_w / 2.0)
                y_min = (cy * height) - (box_h / 2.0)

                area = box_w * box_h

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": [round(x_min, 2), round(y_min, 2), round(box_w, 2), round(box_h, 2)],
                    "area": round(area, 2),
                    "iscrowd": 0
                })
                ann_id += 1

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    print("=" * 60)
    print(" KONVERSI FORMAT YOLO KE COCO JSON BERHASIL")
    print("=" * 60)
    print(f" Output COCO JSON: {output_json}")
    print(f" Total Citra     : {len(coco['images'])}")
    print(f" Total Anotasi   : {len(coco['annotations'])}")
    print(f" Kategori/Kelas  : {[c['name'] for c in coco['categories']]}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Konversi anotasi format YOLO ke format COCO JSON")
    parser.add_argument("yolo_dir", type=Path, nargs="?", default=Path("dataset_yolo"), help="Folder dataset format YOLO")
    parser.add_argument("output_json", type=Path, nargs="?", default=Path("dataset_yolo/annotations_coco.json"), help="File target .json format COCO")
    args = parser.parse_args()

    yolo_to_coco(args.yolo_dir, args.output_json)


if __name__ == "__main__":
    main()
