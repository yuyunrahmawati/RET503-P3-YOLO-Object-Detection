#!/usr/bin/env python3
"""00_generate_anotasi_yolo.py
Menghasilkan anotasi Bounding Box format YOLO dari data proyek industri P2:
- box_merah (label id: 1)
- box_cokelat (label id: 0)
Format label YOLO: <class_id> <cx> <cy> <w> <h> (normalisasi 0 - 1).
"""

import os
import glob
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "dataset_raw"
OUT_DIR = ROOT / "dataset_yolo"
IMG_OUT = OUT_DIR / "images"
LBL_OUT = OUT_DIR / "labels"

IMG_OUT.mkdir(parents=True, exist_ok=True)
LBL_OUT.mkdir(parents=True, exist_ok=True)


def get_box_merah_bbox(img):
    """Mendeteksi kubus box merah industri berdasarkan kontur warna merah."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0, 70, 50), (12, 255, 255))
    mask2 = cv2.inRange(hsv, (168, 70, 50), (180, 255, 255))
    mask = mask1 | mask2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) > 2000:
            return cv2.boundingRect(c)
    # Fallback default central region if obscured
    h, w = img.shape[:2]
    return int(w * 0.35), int(h * 0.35), int(w * 0.35), int(h * 0.45)


def get_box_cokelat_bbox(img):
    """Mendeteksi kubus kardus cokelat industri yang diletakkan di atas meja kerja."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Exclude red pad below box (H < 12 or H > 168 with S > 80)
    red = ((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 168)) & (hsv[:, :, 1] > 80)
    # Exclude dark curtain/background (V < 55)
    dark = hsv[:, :, 2] < 55
    cardboard = (~red) & (~dark)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[int(h * 0.02) : int(h * 0.78), int(w * 0.20) : int(w * 0.85)] = 255
    cardboard_roi = cv2.bitwise_and(cardboard.astype(np.uint8) * 255, mask)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    cardboard_roi = cv2.morphologyEx(cardboard_roi, cv2.MORPH_CLOSE, k)
    cardboard_roi = cv2.morphologyEx(cardboard_roi, cv2.MORPH_OPEN, k)

    cnts, _ = cv2.findContours(cardboard_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) > 5000:
            return cv2.boundingRect(c)
    # Fallback to standard box center if needed
    return int(w * 0.30), int(h * 0.05), int(w * 0.42), int(h * 0.65)


def main():
    print("=" * 60)
    print(" 00: Menghasilkan Anotasi Format YOLO dari Data Proyek")
    print("=" * 60)

    classes = [("box_cokelat", 0, get_box_cokelat_bbox), ("box_merah", 1, get_box_merah_bbox)]

    total_annotated = 0
    for class_name, class_id, detector in classes:
        src_folder = RAW_DIR / class_name
        files = sorted(glob.glob(str(src_folder / "*.jpg")))
        print(f"\nMemproses kelas '{class_name}' (Class ID: {class_id}) - {len(files)} citra...")

        for fpath in files:
            p = Path(fpath)
            img = cv2.imread(str(p))
            if img is None:
                continue
            h, w = img.shape[:2]

            x, y, bw, bh = detector(img)

            # Pastikan dalam batas gambar
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            bw = max(1, min(bw, w - x))
            bh = max(1, min(bh, h - y))

            # Konversi koordinat piksel ke YOLO normalized format (cx, cy, nw, nh)
            cx = (x + bw / 2.0) / w
            cy = (y + bh / 2.0) / h
            nw = bw / float(w)
            nh = bh / float(h)

            # Simpan label .txt
            label_file = LBL_OUT / f"{p.stem}.txt"
            with open(label_file, "w", encoding="utf-8") as f:
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

            # Salin citra ke dataset_yolo/images/
            dst_img = IMG_OUT / p.name
            if not dst_img.exists():
                cv2.imwrite(str(dst_img), img)

            total_annotated += 1

    print("\n" + "=" * 60)
    print(f" Selesai! Total citra dan label berhasil dibuat: {total_annotated}")
    print(f" Lokasi Citra : {IMG_OUT}")
    print(f" Lokasi Label : {LBL_OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
