#!/usr/bin/env python3
"""predict.py
Uji inferensi deteksi objek YOLO pada citra tunggal atau folder citra.
Memuat model terbaik hasil pelatihan dan menampilkan koordinat kotak deteksi,
label kelas, dan tingkat keyakinan (confidence).
Penggunaan:
    python predict.py <path_gambar.jpg>
    python predict.py dataset_v1/images/test/box_merah__frame_000005491.jpg
"""

import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def find_best_model():
    candidates = [
        ROOT / "runs" / "detect" / "E3_tuning" / "weights" / "best.pt",
        ROOT / "runs" / "detect" / "E1_pretrained" / "weights" / "best.pt",
        ROOT / "runs" / "detect" / "E2_scratch" / "weights" / "best.pt",
        ROOT / "yolo11n.pt"
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description="Inferensi Deteksi Objek Box Industri menggunakan Model YOLO")
    parser.add_argument("source", type=str, nargs="?", help="Path ke citra yang ingin diuji")
    parser.add_argument("--model", type=Path, default=None, help="Path ke checkpoint model .pt (default: best model)")
    parser.add_argument("--conf", type=float, default=0.25, help="Ambang batas keyakinan / confidence threshold (default: 0.25)")
    parser.add_argument("--save", action="store_true", default=True, help="Simpan hasil visualisasi prediksi")
    args = parser.parse_args()

    if not args.source:
        print("Penggunaan: python predict.py <path_gambar.jpg>")
        print("Contoh    : python predict.py dataset_v1/images/test/box_merah__frame_000005491.jpg")
        sys.exit(1)

    src_path = Path(args.source)
    if not src_path.exists():
        print(f"Error: File '{src_path}' tidak ditemukan!")
        sys.exit(1)

    model_path = args.model if args.model else find_best_model()
    if not model_path.exists():
        print(f"Error: Model '{model_path}' tidak ditemukan. Jalankan 06_latih_bandingkan.py terlebih dahulu!")
        sys.exit(1)

    print("=" * 60)
    print(" INFERENSI DETEKSI OBJEK YOLO (ROBOT VISI RET503)")
    print("=" * 60)
    print(f" Model Digunakan: {model_path}")
    print(f" Citra Masukan  : {src_path}")
    print(f" Ambang Conf    : {args.conf}")

    model = YOLO(str(model_path))
    results = model.predict(source=str(src_path), conf=args.conf, save=args.save, project=str(ROOT / "runs"), name="predict", exist_ok=True)

    print("-" * 60)
    print(" HASIL DETEKSI OBJEK:")
    found = 0
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            cls_name = model.names.get(cls_id, str(cls_id))
            conf = float(box.conf[0].item()) * 100
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            print(f"  [+] Kelas: {cls_name.upper():<12} | Conf: {conf:6.2f}% | BBox: [{x1}, {y1}, {x2}, {y2}]")
            found += 1

    if found == 0:
        print("  [-] Tidak ada objek yang terdeteksi di atas ambang batas confidence.")

    print("=" * 60)
    if args.save:
        save_dir = ROOT / "runs" / "predict"
        print(f" Visualisasi hasil deteksi disimpan ke folder: {save_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()
