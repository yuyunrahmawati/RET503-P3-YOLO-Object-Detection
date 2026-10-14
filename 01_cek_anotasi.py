#!/usr/bin/env python3
"""01_cek_anotasi.py
Memeriksa format anotasi YOLO:
- Memastikan pasangan file citra dan label saling cocok.
- Memastikan struktur kolom: <class_id> <cx> <cy> <w> <h>.
- Memvalidasi rentang normalisasi [0, 1].
- Menampilkan statistik kelas dan ukuran kotak deteksi.
Penggunaan:
    python 01_cek_anotasi.py dataset_yolo
"""

import sys
import argparse
from pathlib import Path

CLASS_NAMES = {0: "box_cokelat", 1: "box_merah"}


def check_annotations(dataset_path: Path):
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"

    if not images_dir.is_dir() or not labels_dir.is_dir():
        # Fallback jika images/labels langsung ada di folder
        print(f"Error: Folder '{images_dir}' atau '{labels_dir}' tidak ditemukan!")
        sys.exit(1)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = {p.stem: p for p in images_dir.iterdir() if p.suffix.lower() in image_exts}
    labels = {p.stem: p for p in labels_dir.iterdir() if p.suffix.lower() == ".txt"}

    print("=" * 60)
    print(" PEMERIKSAAN INTEGRITAS ANOTASI DATASET YOLO")
    print("=" * 60)
    print(f" Direktori Dataset: {dataset_path}")
    print(f" Jumlah Citra     : {len(images)}")
    print(f" Jumlah File Label: {len(labels)}")

    # 1. Cek kecocokan file
    missing_labels = set(images.keys()) - set(labels.keys())
    missing_images = set(labels.keys()) - set(images.keys())

    if missing_labels:
        print(f" [PERINGATAN] {len(missing_labels)} citra tidak memiliki file label:")
        for name in list(missing_labels)[:5]:
            print(f"   - {name}")
    if missing_images:
        print(f" [PERINGATAN] {len(missing_images)} label tidak memiliki file citra:")
        for name in list(missing_images)[:5]:
            print(f"   - {name}")

    # 2. Cek format dan validitas angka koordinat
    total_boxes = 0
    class_counts = {}
    errors = []
    box_widths = []
    box_heights = []

    for stem, lbl_path in sorted(labels.items()):
        with open(lbl_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            errors.append(f"{lbl_path.name}: File label kosong tanpa kotak.")
            continue

        for line_no, line in enumerate(lines, 1):
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"{lbl_path.name}:{line_no}: Format salah, harus ada 5 nilai (ditemukan {len(parts)}).")
                continue

            try:
                cls_id = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                errors.append(f"{lbl_path.name}:{line_no}: Tipe data tidak valid.")
                continue

            # Validasi class ID
            if cls_id not in CLASS_NAMES:
                errors.append(f"{lbl_path.name}:{line_no}: Class ID {cls_id} tidak dikenal.")

            # Validasi batas 0 - 1
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                errors.append(f"{lbl_path.name}:{line_no}: Koordinat di luar rentang normalisasi [0, 1].")

            # Validasi batas kotak
            x1 = cx - w / 2.0
            x2 = cx + w / 2.0
            y1 = cy - h / 2.0
            y2 = cy + h / 2.0
            tol = 0.01  # toleransi numerik pembulatan
            if x1 < -tol or y1 < -tol or x2 > 1.0 + tol or y2 > 1.0 + tol:
                errors.append(f"{lbl_path.name}:{line_no}: Kotak keluar dari bidang citra (x1={x1:.3f}, x2={x2:.3f}, y1={y1:.3f}, y2={y2:.3f}).")

            total_boxes += 1
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
            box_widths.append(w)
            box_heights.append(h)

    print("-" * 60)
    print(" HASIL VALIDASI:")
    if errors:
        print(f" [GAGAL] Ditemukan {len(errors)} masalah pada anotasi:")
        for err in errors[:10]:
            print(f"   x {err}")
    else:
        print(" [SUKSES] Seluruh file anotasi valid dan memenuhi kaidah YOLO!")

    print("-" * 60)
    print(" STATISTIK DATASET:")
    print(f" Total Bounding Box : {total_boxes}")
    for cid, name in sorted(CLASS_NAMES.items()):
        cnt = class_counts.get(cid, 0)
        pct = (cnt / total_boxes * 100) if total_boxes > 0 else 0
        print(f"  - Kelas {cid} ({name:12s}): {cnt:4d} kotak ({pct:.1f}%)")

    if box_widths:
        print(f" Rata-rata Lebar Kotak (W) : {sum(box_widths)/len(box_widths):.4f}")
        print(f" Rata-rata Tinggi Kotak (H): {sum(box_heights)/len(box_heights):.4f}")
    print("=" * 60)

    return len(errors) == 0


def main():
    parser = argparse.ArgumentParser(description="Pemeriksaan format anotasi YOLO")
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("dataset_yolo"), help="Path ke folder dataset_yolo")
    args = parser.parse_args()

    success = check_annotations(args.dataset)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
