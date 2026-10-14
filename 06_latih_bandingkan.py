#!/usr/bin/env python3
"""06_latih_bandingkan.py
Program Utama Praktik Terpadu Deteksi Objek YOLO (RET503 Minggu 4-7):
- Melatih 3 mode eksperimen secara sistematis:
    * E1_pretrained : Baseline model transfer learning dari COCO (yolo11n.pt)
    * E2_scratch    : Melatih dari nol dengan arsitektur sama tanpa transfer learning (yolo11n.yaml)
    * E3_tuning     : Eksperimen satu langkah perubahan (default: imgsz=640)
- Menghitung sidik jari dataset (label fingerprint) untuk autentikasi dataset.
- Memilih model terbaik berdasarkan mAP50-95 pada data val.
- Menguji model terpilih satu kali pada data test (Slide 31 & 36).
- Mencetak dan menyimpan format tabel evaluasi resmi sesuai Slide 37 ke file:
    * hasil_praktik_<NIM>.txt
    * hasil_praktik_<NIM>.csv
Sesuai materi Slide 19-31, 36-37 RET503.
Penggunaan:
    python 06_latih_bandingkan.py --nim 4222401078 --nama "Yuyun Rahmawati"
"""

import sys
import time
import hashlib
import platform
import argparse
from pathlib import Path
from datetime import datetime

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def compute_label_fingerprint(labels_dir: Path) -> str:
    """Menghitung checksum hash SHA256 dari seluruh file label dalam dataset."""
    hasher = hashlib.sha256()
    txt_files = sorted(labels_dir.rglob("*.txt"))
    if not txt_files:
        return "label_kosong"
    for fpath in txt_files:
        hasher.update(fpath.name.encode("utf-8"))
        with open(fpath, "rb") as f:
            hasher.update(f.read())
    return hasher.hexdigest()[:12]


def run_experiment(exp_name, model_source, data_yaml, epochs, imgsz, batch, seed, device, extra_args=None):
    """Menjalankan satu sesi pelatihan dan validasi YOLO."""
    print("\n" + "=" * 70)
    print(f" >>> MEMULAI EKSPERIMEN: {exp_name}")
    print(f"     Model Source : {model_source}")
    print(f"     Epochs={epochs}, Imgsz={imgsz}, Batch={batch}, Device={device}")
    print("=" * 70)

    train_args = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "seed": seed,
        "device": device,
        "project": str(ROOT / "runs" / "detect"),
        "name": exp_name,
        "exist_ok": True,
        "verbose": False,
        "plots": True,
    }

    if extra_args:
        train_args.update(extra_args)

    start_time = time.time()
    model = YOLO(model_source)
    model.train(**train_args)
    elapsed_minutes = (time.time() - start_time) / 60.0

    # Evaluasi pada data val dengan bobot terbaik
    best_pt = ROOT / "runs" / "detect" / exp_name / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = ROOT / "runs" / "detect" / exp_name / "weights" / "last.pt"

    val_model = YOLO(str(best_pt))
    metrics = val_model.val(data=str(data_yaml), split="val", imgsz=imgsz, device=device, verbose=False)

    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)

    # Ukur kecepatan inferensi murni per citra
    speed_dict = getattr(metrics, "speed", {})
    inference_ms = float(speed_dict.get("inference", 15.0))

    return {
        "name": exp_name,
        "model_path": str(best_pt),
        "map50": map50,
        "map50_95": map50_95,
        "precision": precision,
        "recall": recall,
        "latency_ms": inference_ms,
        "duration_min": elapsed_minutes
    }


def main():
    parser = argparse.ArgumentParser(description="Program Praktik Terpadu Latih dan Bandingkan YOLO (Slide 36-37)")
    parser.add_argument("--nim", type=str, default="4222401078", help="Nomor Induk Mahasiswa")
    parser.add_argument("--nama", type=str, default="Yuyun Rahmawati", help="Nama Lengkap Mahasiswa")
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset_v1" / "data.yaml", help="Path ke data.yaml")
    parser.add_argument("--epochs", type=int, default=15, help="Jumlah epoch pelatihan per mode (default: 15)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=416, help="Ukuran resolusi masukan standar (default: 416)")
    parser.add_argument("--ubah", type=str, default="imgsz=640", help="Satu perubahan untuk E3_tuning (format: kunci=nilai)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed reproduktibilitas (default: 42)")
    parser.add_argument("--device", type=str, default="", help="Perangkat komputasi: 'cpu' atau '0'/'cuda'")
    args = parser.parse_args()

    # Tentukan device
    if not args.device:
        device = "0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    data_yaml = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    if not data_yaml.exists():
        print(f"Error: File '{data_yaml}' tidak ditemukan! Jalankan 05_split_dataset_yolo.py terlebih dahulu.")
        sys.exit(1)

    labels_dir = data_yaml.parent / "labels"
    fingerprint = compute_label_fingerprint(labels_dir)
    comp_info = f"{platform.node()} ({platform.processor() or platform.machine()})"

    # Parse parameter perubahan untuk E3_tuning
    e3_extra = {}
    e3_imgsz = args.imgsz
    perubahan_str = args.ubah
    if "=" in args.ubah:
        k, v = args.ubah.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k == "imgsz":
            e3_imgsz = int(v)
        else:
            try:
                e3_extra[k] = float(v) if "." in v else int(v)
            except ValueError:
                e3_extra[k] = v

    print("=" * 80)
    print(" PRAKTIK TERPADU DETEKSI OBJEK YOLO - MINGGU 4-7 (RET503)")
    print("=" * 80)
    print(f" NIM / Nama    : {args.nim} / {args.nama}")
    print(f" Waktu Mulai   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Komputer      : {comp_info} [Device: {device}]")
    print(f" Dataset       : {data_yaml.parent.name} (sidik jari label: {fingerprint})")
    print("=" * 80)

    # 1. Eksperimen E1: Pretrained Baseline
    e1_res = run_experiment(
        exp_name="E1_pretrained",
        model_source="yolo11n.pt",
        data_yaml=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        device=device
    )

    # 2. Eksperimen E2: Training from Scratch (tanpa bobot pretrained)
    e2_res = run_experiment(
        exp_name="E2_scratch",
        model_source="yolo11n.yaml",
        data_yaml=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        device=device
    )

    # 3. Eksperimen E3: Tuning (satu perubahan terukur)
    e3_res = run_experiment(
        exp_name="E3_tuning",
        model_source="yolo11n.pt",
        data_yaml=data_yaml,
        epochs=args.epochs,
        imgsz=e3_imgsz,
        batch=args.batch,
        seed=args.seed,
        device=device,
        extra_args=e3_extra
    )

    # Pilih model terbaik berdasarkan mAP50-95 di data val (Slide 31 & 36)
    experiments = [
        (e1_res, "-"),
        (e2_res, "-"),
        (e3_res, perubahan_str)
    ]
    best_exp, _ = max(experiments, key=lambda item: item[0]["map50_95"])

    # Uji model terpilih SEKALI SAJA pada data TEST (Slide 31)
    print("\n" + "=" * 70)
    print(f" >>> EVALUASI AKHIR PADA DATA TEST (SEKALI SAJA): {best_exp['name']}")
    print("=" * 70)
    test_model = YOLO(best_exp["model_path"])
    test_metrics = test_model.val(data=str(data_yaml), split="test", device=device, verbose=False)
    test_map50 = float(test_metrics.box.map50)
    test_map50_95 = float(test_metrics.box.map)
    test_p = float(test_metrics.box.mp)
    test_r = float(test_metrics.box.mr)

    # Buat Format Output Persis Sesuai Slide 37
    out_lines = []
    out_lines.append(f"  NIM / Nama    : {args.nim} / {args.nama}")
    out_lines.append(f"  Waktu         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out_lines.append(f"  Komputer      : {comp_info} (Device: {device})")
    out_lines.append(f"  Dataset       : {data_yaml.parent.name}  (sidik jari label: {fingerprint})")
    out_lines.append(f"  {'Eksperimen':<14} {'Perubahan':<11} {'mAP50':<7} {'mAP50-95':<9} {'P':<6} {'R':<6} {'ms':<5} {'menit':<5}")
    for exp, perub in experiments:
        out_lines.append(
            f"  {exp['name']:<14} {perub:<11} {exp['map50']:<7.3f} {exp['map50_95']:<9.3f} "
            f"{exp['precision']:<6.3f} {exp['recall']:<6.3f} {exp['latency_ms']:<5.1f} {exp['duration_min']:<5.1f}"
        )
    out_lines.append(f"  Model terpilih (mAP50-95 val tertinggi): {best_exp['name']}")
    out_lines.append(f"  Hasil di data TEST (sekali saja)       : mAP50={test_map50:.3f}, mAP50-95={test_map50_95:.3f}, P={test_p:.3f}, R={test_r:.3f}")

    table_output = "\n".join(out_lines)

    print("\n" + "=" * 80)
    print(" RINGKASAN RESMI KELUARAN PROGRAM (FORMAT SLIDE 37)")
    print("=" * 80)
    print(table_output)
    print("=" * 80)

    # Simpan ke file teks resmi: hasil_praktik_<NIM>.txt
    txt_out_path = ROOT / f"hasil_praktik_{args.nim}.txt"
    with open(txt_out_path, "w", encoding="utf-8") as f:
        f.write(table_output + "\n")

    # Simpan ke file CSV resmi: hasil_praktik_<NIM>.csv
    csv_out_path = ROOT / f"hasil_praktik_{args.nim}.csv"
    with open(csv_out_path, "w", encoding="utf-8") as f:
        f.write("eksperimen,perubahan,map50,map50_95,precision,recall,latency_ms,duration_min\n")
        for exp, perub in experiments:
            f.write(f"{exp['name']},{perub},{exp['map50']:.4f},{exp['map50_95']:.4f},{exp['precision']:.4f},{exp['recall']:.4f},{exp['latency_ms']:.2f},{exp['duration_min']:.2f}\n")
        f.write(f"TEST_{best_exp['name']},test_split,{test_map50:.4f},{test_map50_95:.4f},{test_p:.4f},{test_r:.4f},-,-\n")

    print(f"\n[SUKSES] Laporan teks tersimpan di : {txt_out_path}")
    print(f"[SUKSES] Laporan CSV tersimpan di  : {csv_out_path}")


if __name__ == "__main__":
    main()
