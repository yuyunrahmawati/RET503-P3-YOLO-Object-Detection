#!/usr/bin/env python3
"""05_split_dataset_yolo.py
Membagi dataset_yolo menjadi subset train, val, dan test secara proporsional
(70% train, 20% val, 10% test) dengan temporal split untuk mencegah data leakage.
Menghasilkan struktur folder standar YOLO serta file konfigurasi data.yaml.
Sesuai materi Slide 6, 15, & 35 RET503.
Penggunaan:
    python 05_split_dataset_yolo.py dataset_yolo dataset_v1
"""

import sys
import shutil
import argparse
from pathlib import Path

CLASS_NAMES = {0: "box_cokelat", 1: "box_merah"}


def split_dataset(src_dir: Path, out_dir: Path, train_ratio=0.70, val_ratio=0.20, test_ratio=0.10):
    images_dir = src_dir / "images"
    labels_dir = src_dir / "labels"

    if not images_dir.is_dir() or not labels_dir.is_dir():
        print(f"Error: Folder '{images_dir}' atau '{labels_dir}' tidak ditemukan!")
        sys.exit(1)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in image_exts])

    # Kelompokkan per kelas berdasarkan prefix atau nama file
    by_class = {}
    for img_p in img_files:
        # Tentukan kelas dari label atau nama file
        lbl_p = labels_dir / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue
        with open(lbl_p, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if not first_line:
                continue
            cls_id = int(first_line.split()[0])
        by_class.setdefault(cls_id, []).append(img_p)

    # Buat direktori tujuan
    splits = ["train", "val", "test"]
    for s in splits:
        (out_dir / "images" / s).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / s).mkdir(parents=True, exist_ok=True)

    summary = {s: {} for s in splits}

    for cls_id, files in sorted(by_class.items()):
        # Urutkan secara temporal (berdasarkan nama frame)
        files = sorted(files, key=lambda p: p.name)
        n = len(files)

        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_test = n - n_train - n_val

        split_assignment = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:]
        }

        cname = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        for s, sfiles in split_assignment.items():
            summary[s][cname] = len(sfiles)
            for f in sfiles:
                # Salin citra
                dst_img = out_dir / "images" / s / f.name
                shutil.copy2(f, dst_img)

                # Salin label
                src_lbl = labels_dir / f"{f.stem}.txt"
                dst_lbl = out_dir / "labels" / s / f"{f.stem}.txt"
                if src_lbl.exists():
                    shutil.copy2(src_lbl, dst_lbl)

    # Buat file data.yaml
    data_yaml_path = out_dir / "data.yaml"
    # Gunakan forward slashes untuk path agar kompatibel lintas platform
    abs_out_path = out_dir.resolve().as_posix()
    yaml_content = f"""# Konfigurasi Dataset Deteksi Objek Box Industri RET503
path: {abs_out_path}
train: images/train
val: images/val
test: images/test

nc: {len(CLASS_NAMES)}
names:
  0: box_cokelat
  1: box_merah
"""
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print("=" * 60)
    print(" PEMBAGIAN DATASET YOLO BERHASIL (dataset_v1)")
    print("=" * 60)
    print(f" Output Folder : {out_dir}")
    print(f" File data.yaml: {data_yaml_path}")
    print("-" * 60)
    print(f" {'Split':<10} {'box_cokelat':<15} {'box_merah':<15} {'Total':<10}")
    print("-" * 60)
    for s in splits:
        cnt_cokelat = summary[s].get("box_cokelat", 0)
        cnt_merah = summary[s].get("box_merah", 0)
        tot = cnt_cokelat + cnt_merah
        print(f" {s:<10} {cnt_cokelat:<15} {cnt_merah:<15} {tot:<10}")
    print("-" * 60)
    total_all = sum(sum(v.values()) for v in summary.values())
    print(f" TOTAL SEMUA CITRA: {total_all}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Pemisahan dataset YOLO ke train, val, dan test")
    parser.add_argument("src_dir", type=Path, nargs="?", default=Path("dataset_yolo"), help="Folder sumber dataset_yolo")
    parser.add_argument("out_dir", type=Path, nargs="?", default=Path("dataset_v1"), help="Folder target dataset_v1")
    args = parser.parse_args()

    split_dataset(args.src_dir, args.out_dir)


if __name__ == "__main__":
    main()
