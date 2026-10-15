# Deteksi Objek Box Industri pada Robot Visi Menggunakan YOLO
## Praktik Terpadu Minggu 4–7: Dari Data ke Model YOLO (RET503)

[![Course](https://img.shields.io/badge/Course-RET503%20Computer%20Vision%20%26%20Deep%20Learning-blue.svg)](https://polibatam.ac.id)
[![Institution](https://img.shields.io/badge/Politeknik%20Negeri%20Batam-Teknologi%20Rekayasa%20Robotika-orange.svg)](https://polibatam.ac.id)
[![CLO](https://img.shields.io/badge/Assessment-CLO--3%20%7C%20CLO--4-purple.svg)](#)
[![CDIO](https://img.shields.io/badge/CDIO%20Stage-Stage%20%232%20Design-green.svg)](#)
[![Framework](https://img.shields.io/badge/Framework-Ultralytics%20YOLO11%20%7C%20PyTorch-EE4C2C.svg?logo=pytorch)](https://ultralytics.com)

---

## Identitas Mahasiswa & Ringkasan Proyek

- **Nama Mahasiswa :** Yuyun Rahmawati
- **NIM :** 4222401078
- **Kelas :** Pagi C5
- **Program Studi :** Sarjana Terapan Teknologi Rekayasa Robotika
- **Institusi :** Politeknik Negeri Batam
- **Projek Asal Data :** `RET503-P2-Transfer-Learning-MobileNetV2` (Objek Industri: Box Merah & Box Cokelat)

Repositori ini memuat seluruh artefak kode, pipeline preparasi dataset (format YOLO & COCO), pembagian dataset anti-data leakage, pelatihan dan komparasi 3 mode eksperimen YOLO, pengukuran latensi inferensi, dan laporan akhir praktik terpadu sesuai panduan modul **RET503 Minggu 4–7 (Slide 32–37)**.

---

## 🗂️ Struktur Direktori Repositori

```text
RET503-P3-YOLO-Object-Detection/
├── dataset_raw/                         # Dataset mentah dari P2
│   ├── box_merah/                       # 100 citra box merah industri
│   ├── box_cokelat/                     # 100 citra box kardus cokelat industri
│   └── metadata.csv                     # Manifest metadata asli P2
├── dataset_yolo/                        # Tahap 1: Anotasi YOLO (images & labels .txt)
│   ├── images/                          # 200 citra akuisisi kamera
│   ├── labels/                          # 200 file label .txt (format: class cx cy w h)
│   └── annotations_coco.json            # Tahap 1: File anotasi format COCO JSON
├── dataset_v1/                          # Tahap 2: Pembagian Dataset YOLO (70/20/10)
│   ├── data.yaml                        # Definisi konfigurasi dataset YOLO
│   ├── images/                          # Subset citra train, val, dan test
│   └── labels/                          # Subset label train, val, dan test
├── runs/detect/                         # Log, kurva loss/mAP, confusion matrix, & bobot
│   ├── E1_pretrained/weights/best.pt    # Bobot terbaik model baseline transfer learning
│   ├── E2_scratch/weights/best.pt       # Bobot terbaik model scratch (tanpa pretrained)
│   └── E3_tuning/weights/best.pt        # [TERBAIK] Bobot model hasil tuning (imgsz=640)
├── 00_generate_anotasi_yolo.py          # Script anotasi bounding box dari dataset P2
├── 01_cek_anotasi.py                    # Slide 35: Validasi format & integritas anotasi YOLO
├── 02_yolo_ke_coco.py                   # Slide 35: Konversi format YOLO ke format COCO JSON
├── 03_coco_ke_yolo.py                   # Slide 35: Uji konversi dua arah (COCO JSON ke YOLO)
├── 05_split_dataset_yolo.py             # Slide 35: Pembagian train/val/test & pembuatan data.yaml
├── 06_latih_bandingkan.py               # Slide 36-37: Script eksperimen E1, E2, E3 + evaluasi test
├── predict.py                           # Script uji inferensi deteksi objek pada citra baru
├── hasil_praktik_4222401078.txt         # Slide 37: Laporan resmi keluaran program
├── hasil_praktik_4222401078.csv         # Slide 37: Tabel metrik eksperimen (CSV)
├── .gitignore                           # Filter file git
└── README.md                            # Laporan analisis komprehensif & panduan eksekusi
```

---

## 1. Rancangan Sistem Persepsi (Slide 32)

1. **Misi Proyek:** Sistem visi bertugas mengenali posisi (*bounding box*) dan mengklasifikasikan jenis boks industri secara *real-time* untuk memandu lengan robot manipulator pemilah (*sorting arm*).
2. **Kamera & Posisi:** Kamera monokular RGB terpasang pada dudukan *top-down / angled* dengan jarak kerja 30–80 cm di atas konveyor atau meja perakitan.
3. **Kelas Objek Deteksi:**
   - `0: box_cokelat` (Kardus cokelat komponen industri)
   - `1: box_merah` (Kotak boks merah industri)
4. **Unit Komputasi:** Laptop / Edge Computer Robotika (Intel Core i5-12450HX CPU & NVIDIA GeForce RTX 3050 Laptop GPU).
5. **Target Kinerja & Latensi (Budget):**
   - Target akurasi:mAP50≥90%.
   - Target kecepatan inferensi: ≥15 FPS (anggaran latensi ≤67 ms)
   - Model yang efisien dan ringan untuk *embedded edge devices* (arsitektur nano≤3 M parameter).

---

## 📦 2. Preparasi & Manajemen Dataset (Tahap 1 & 2 - Slide 5-7, 15)

Data diambil langsung dari proyek `RET503-P2-Transfer-Learning-MobileNetV2` dengan total **200 citra**:
- 100 citra `box_cokelat` (resolusi 832 * 464)
- 100 citra `box_merah` (resolusi 1280 * 720)

### A. Dua Format Anotasi (Slide 5)
1. **Format YOLO (`dataset_yolo/labels/*.txt`):**
   Setiap citra memiliki file `.txt` bernama sama dengan isi per baris:
   class_id cx ​cy ​w h
   Koordinat dinormalisasi dalam rentang [0, 1] terhadap dimensi citra.
2. **Format COCO (`dataset_yolo/annotations_coco.json`):**
   Seluruh dataset tersimpan dalam satu file JSON terstruktur dengan koordinat piksel [xmin​,ymin​,w,h] pojok kiri atas.

### B. Validasi & Konversi Dua Arah (Slide 35)
- `01_cek_anotasi.py`: Berhasil memverifikasi 200/200 file label tanpa ada koordinat di luar bidang citra atau format salah (tingkat kelulusan 100%).
- `02_yolo_ke_coco.py`: Berhasil mengonversi 200 label ke format standar COCO JSON.
- `03_coco_ke_yolo.py`: Berhasil mengonversi kembali COCO JSON ke YOLO format tanpa kehilangan presisi koordinat (*round-trip test passed*).

### C. Pembagian Dataset (`dataset_v1/`)
Dataset dibagi menggunakan skema *chronological temporal split* untuk mencegah kebocoran data (*data leakage*) antar-frame video yang berdekatan:

| Subset | Rasio | Jumlah `box_cokelat` | Jumlah `box_merah` | Total Citra | Peran dalam Evaluasi |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Train** | 70% | 70 | 70 | **140** | Pelatihan bobot model |
| **Validation** | 20% | 20 | 20 | **40** | Penyetelan & pemilihan model terbaik |
| **Test** | 10% | 10 | 10 | **20** | Pengujian akhir model terpilih (sekali saja) |
| **Total** | 100% | 100 | 100 | **200** | - |

---

## 🚀 3. Eksperimen Pelatihan & Komparasi YOLO (Tahap 3–5 - Slide 36-37)

Pelatihan dijalankan menggunakan arsitektur **YOLO11n** (2,59 juta parameter) melalui skrip otomatis `06_latih_bandingkan.py` dengan parameter terkontrol (10 epoch, batch size 16, seed 42):

1. **E1_pretrained (Baseline Transfer Learning):**
   Model `yolo11n.pt` dengan bobot pretrained COCO. Arsitektur memanfaatkan fitur umum yang telah dipelajari dari COCO lalu menyesuaikan kepala deteksi untuk 2 kelas boks industri.
2. **E2_scratch (Training from Scratch):**
   Model `yolo11n.yaml` dengan arsitektur sama namun bobot diinisialisasi secara acak (tanpa transfer learning).
3. **E3_tuning (Systematic Tuning):**
   Model pretrained `yolo11n.pt` dengan satu perubahan terukur: resolusi input dinaikkan dari `imgsz=416` menjadi `imgsz=640` untuk meningkatkan visibilitas detail kontur objek.

---

## 4. Keluaran Resmi Program (Format Slide 37)

Berikut adalah tabel keluaran resmi yang dicatat oleh program `06_latih_bandingkan.py` dan tersimpan di [`hasil_praktik_4222401078.txt`](hasil_praktik_4222401078.txt):

```text
================================================================================
 RINGKASAN RESMI KELUARAN PROGRAM (FORMAT SLIDE 37)
================================================================================
  NIM / Nama    : 4222401078 / Yuyun Rahmawati
  Waktu         : 2026-10-15 03:42:00
  Komputer      : NEKO-TECH (Intel64 Family 6 Model 151 Stepping 2, GenuineIntel) (Device: cpu)
  Dataset       : dataset_v1  (sidik jari label: 492f7b41f3b3)
  Eksperimen     Perubahan   mAP50   mAP50-95  P      R      ms    menit
  E1_pretrained  -           0.811   0.518     0.815  0.825  13.2  2.7  
  E2_scratch     -           0.006   0.002     0.003  0.575  13.8  2.8  
  E3_tuning      imgsz=640   0.910   0.571     0.917  0.925  29.3  6.4  
  Model terpilih (mAP50-95 val tertinggi): E3_tuning
  Hasil di data TEST (sekali saja)       : mAP50=0.701, mAP50-95=0.403, P=0.763, R=0.800
================================================================================
```

---

## 🧠 5. Analisis Evaluasi & Pembahasan Ilmiah (Slide 30–31, 38)

### 1. Bukti Manfaat Transfer Learning (E1 vs E2)
- **E1_pretrained** mencapai **mAP50 = 81.1%** dan **mAP50-95 = 51.8%** dalam 10 epoch.
- Sebaliknya, **E2_scratch** hanya memperoleh **mAP50 = 0.6%** dan **mAP50-95 = 0.2%**.
- **Kesimpulan:** Hal ini membuktikan secara empiris teori pada Slide 20: melatih model deteksi objek dari nol (*scratch*) pada dataset berskala ratusan citra (140 citra latih) tidak mencukupi untuk membentuk representasi fitur konvolusi spasial yang stabil. Transfer learning dari bobot COCO memberikan lompatan performa yang signifikan sejak epoch awal.

### 2. Evaluasi Hasil Tuning (E1 vs E3)
- Dengan mengubah satu faktor yaitu menaikkan resolusi masukan (`imgsz=640`), **E3_tuning** berhasil meningkatkan:
  - **mAP50:** dari 0.811 menjadi **0.910** (+9.9%)
  - **mAP50-95:** dari 0.518 menjadi **0.571** (+5.3%)
  - **Precision:** 0.917 (91.7% deteksi benar-benar objek boks)
  - **Recall:** 0.925 (92.5% objek boks yang ada berhasil ditemukan)
- **Kesimpulan:** Perubahan resolusi masukan ke 640 piksel dipertahankan dan ditetapkan sebagai model final (*baseline* baru).

### 3. Pengujian Model Terpilih pada Data TEST (Slide 31)
- Model terpilih **E3_tuning** diuji satu kali (*single evaluation*) pada subset `test` (20 citra independen).
- Hasil pengujian data test:
  - mAP50 = 0,701 (70,1%)
  - mAP50-95 = 0.403 (40.3%)
  - Precision = 0.763 (76.3%)
  - Recall = 0.800 (80.0%)
- Performa data test sedikit lebih rendah dari data val karena subset test diambil dari segmen video paling akhir dengan sudut orientasi ekstrim, membuktikan pengujian dilakukan secara jujur dan bebas dari manipulasi pemilihan data berulang.

### 4. Evaluasi Latensi & Throughput Robot
- Waktu inferensi model terbaik (`E3_tuning`) adalah **29.3 ms per citra** pada CPU (setara **~34.1 FPS**).
- Pada mode baseline (`E1_pretrained`), latensi adalah **13.2 ms per citra** (setara **~75.7 FPS**).
- Kedua model berada jauh di bawah ambang batas anggaran latensi sistem robot (≤67 ms / target ≥15 FPS).

---

## 6. Panduan Menjalankan Program (Step-by-Step)

### Langkah 1: Persiapan Environment
Gunakan virtual environment Python dengan dependensi `ultralytics`, `torch`, `opencv-python`, dan `matplotlib`:

```bash
# Menggunakan virtualenv
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# Atau di Windows PowerShell:
.\.venv\Scripts\activate

pip install ultralytics torch torchvision opencv-python matplotlib pillow
```

### Langkah 2: Pemeriksaan & Konversi Format (Tahap 1 - Slide 35)
```bash
# 1. Periksa integritas format label YOLO
python 01_cek_anotasi.py dataset_yolo

# 2. Konversi format YOLO ke format COCO JSON
python 02_yolo_ke_coco.py dataset_yolo dataset_yolo/annotations_coco.json

# 3. Uji konversi dua arah COCO JSON ke YOLO (round-trip test)
python 03_coco_ke_yolo.py dataset_yolo/annotations_coco.json hasil_balik
```

### Langkah 3: Pembagian Dataset Train/Val/Test (Tahap 2 - Slide 35)
```bash
python 05_split_dataset_yolo.py dataset_yolo dataset_v1
```

### Langkah 4: Pelatihan & Komparasi 3 Mode YOLO (Tahap 3–5 - Slide 36-37)
```bash
python 06_latih_bandingkan.py --nim 4222401078 --nama "Yuyun Rahmawati"
```
*Program otomatis menjalankan E1, E2, E3, memilih model terbaik, menguji pada data test, mencetak tabel Slide 37, dan menyimpan hasil ke `hasil_praktik_4222401078.txt` dan `hasil_praktik_4222401078.csv`.*

### Langkah 5: Uji Prediksi Inferensi pada Citra Baru
```bash
# Uji deteksi pada citra box merah
python predict.py dataset_v1/images/test/box_merah__frame_000005491.jpg

# Uji deteksi pada citra box cokelat
python predict.py dataset_v1/images/test/box_cokelat__frame_000003725.jpg
```

Output Inferensi:
```text
============================================================
 INFERENSI DETEKSI OBJEK YOLO (ROBOT VISI RET503)
============================================================
 Model Digunakan: runs/detect/E3_tuning/weights/best.pt
 Citra Masukan  : dataset_v1/images/test/box_merah__frame_000005491.jpg
 Ambang Conf    : 0.25
------------------------------------------------------------
 HASIL DETEKSI OBJEK:
  [+] Kelas: BOX_MERAH    | Conf:  97.21% | BBox: [509, 285, 810, 607]
============================================================
```
