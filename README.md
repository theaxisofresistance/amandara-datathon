# Nexar YOLO Near-Miss

Baseline analisis video dashcam menggunakan:

- Nexar Collision Prediction Dataset;
- YOLO11;
- ByteTrack atau BoT-SORT;
- trajectory objek;
- estimasi TTC relatif;
- risk score;
- output video, CSV, dan dashboard Streamlit.

## Catatan penting

Dataset Nexar memiliki sekitar 1.500 video training, terdiri dari kasus
collision/near-collision dan normal driving. Repository Hugging Face secara
keseluruhan juga menyediakan data evaluasi dan berukuran sekitar 31 GB.

Proyek ini menggunakan YOLO pretrained sebagai detector. Tidak ada training
YOLO khusus karena Nexar tidak menyediakan bounding box YOLO. Risk score
dihitung dari trajectory, posisi objek dalam jalur ego, ukuran bounding box,
pertumbuhan bounding box, kecepatan relatif dalam piksel, akselerasi, dan
prediksi apakah trajectory masuk ke zona ego kendaraan.

TTC yang dihasilkan adalah estimasi relatif, bukan pengukuran meter yang
sudah dikalibrasi.

## 1. Instalasi

Disarankan Python 3.11 atau 3.12.

macOS/Linux:

```bash
chmod +x install.sh
PYTHON_BIN=python3.12 ./install.sh
source .venv/bin/activate
```

Windows:

```bat
install_windows.bat
.venv\Scripts\activate
```

## 2. Unduh subset Nexar

Unduh 20 video pertama beserta metadata:

```bash
python download_nexar.py --max-videos 20
```

Unduh 100 video:

```bash
python download_nexar.py --max-videos 100
```

Lanjutkan batch berikutnya:

```bash
python download_nexar.py --start 100 --max-videos 100
```

Unduh snapshot penuh:

```bash
python download_nexar.py --full
```

Snapshot penuh memerlukan ruang kosong lebih dari 31 GB.

## 3. Temukan video

```bash
find dataset/nexar -type f -name "*.mp4" | head
```

Kemudian jalankan satu video:

```bash
python src/infer.py \
  --input "dataset/nexar/train/NAMA_VIDEO.mp4" \
  --output outputs/result.mp4 \
  --events outputs/events.csv \
  --summary outputs/frame_summary.csv \
  --model yolo11n.pt
```

Apple Silicon:

```bash
python src/infer.py \
  --input "dataset/nexar/train/NAMA_VIDEO.mp4" \
  --device mps
```

NVIDIA:

```bash
python src/infer.py \
  --input "dataset/nexar/train/NAMA_VIDEO.mp4" \
  --device 0
```

## 4. Batch processing

```bash
python src/batch_process.py \
  --input-dir dataset/nexar \
  --output-dir outputs/batch \
  --limit 10
```

## 5. Dashboard

```bash
streamlit run dashboard.py
```

Dashboard membaca `outputs/events.csv` atau file CSV yang diunggah.
Jika tersedia, dashboard juga membaca `outputs/frame_summary.csv` untuk risk
timeline per frame.

## 6. Model 1: Pretrained YOLO Baseline

Model 1 tidak melakukan training. Alurnya:

1. `src/infer.py` membaca video dashcam.
2. YOLO11 pretrained mendeteksi road user: person, bicycle, car, motorcycle,
   bus, dan truck.
3. ByteTrack atau BoT-SORT memberi ID tracking.
4. `src/risk_engine.py` menghitung fitur:
   - TTC relatif dari pertumbuhan bounding box;
   - jarak objek ke zona ego;
   - relative distance dari ukuran bounding box;
   - speed dan acceleration dalam piksel per detik;
   - bbox growth rate;
   - trajectory intersection ke zona ego.
5. Output hanya memakai dua kategori: `SAFE` dan `RISK`.
6. Kendaraan dikategorikan `SAFE` jika:
   - berada di area A/samping dan hanya lewat;
   - menuju area A lalu hilang dari frame;
   - menuju center tetapi bounding box tetap atau mengecil;
   - membesar tetapi tetap di samping atau bergerak ke samping.
7. Area ego adalah C: bagian bawah-tengah frame, dekat kamera/kendaraan sendiri.
   Area A dan B bukan ego area. Kendaraan dikategorikan `RISK` hanya jika
   bergerak menuju area C dan bounding box membesar. Pertumbuhan dihitung dari
   tinggi, lebar, dan area bounding box karena kendaraan dari samping sering
   membesar terutama pada lebar/area.
   Kendaraan yang membesar di pojok A lalu menghilang tetap `SAFE` karena tidak
   menuju C. `edge_intrusion` hanya dipakai untuk debug CSV, bukan syarat cukup
   untuk `RISK`.

Output model 1:

```text
outputs/result.mp4          hanya kandidat tabrakan dengan bbox, ID, trajectory, TTC, risk
outputs/events.csv          event kandidat tabrakan kategori RISK
outputs/frame_summary.csv   ringkasan kandidat tabrakan setiap frame
```

Contoh menjalankan model 1:

```bash
python src/infer.py \
  --input dataset/nexar/train/NAMA_VIDEO.mp4 \
  --output outputs/model1_result.mp4 \
  --events outputs/events.csv \
  --summary outputs/frame_summary.csv \
  --model yolo11n.pt \
  --tracker bytetrack.yaml \
  --conf 0.30
```

Secara default, video output tidak menampilkan semua kendaraan yang terdeteksi
YOLO. Kendaraan normal dipakai hanya untuk tracking internal, lalu diabaikan
jika tidak menuju koridor ego. Untuk debugging deteksi mentah, tambahkan:

```bash
python src/infer.py \
  --input dataset/nexar/train/NAMA_VIDEO.mp4 \
  --show-all-objects
```

Label peringatan pada video hanya `RISK`. Tidak ada countdown, warning,
high-risk, near-miss, atau alert hold. Bounding box mengikuti hasil tracking
kendaraan pada frame berjalan.

## 7. Integrasi Dataset untuk Training YOLO

Dataset Nexar tidak bisa langsung dipakai untuk `yolo detect train` karena
labelnya masih tingkat video, bukan bounding box per objek. Untuk training
YOLO detector, siapkan frame lalu anotasi objek jalan seperti mobil, motor,
bus, truk, sepeda, dan pejalan kaki.

Ekstrak frame dari video Nexar training:

```bash
python src/prepare_yolo_dataset.py \
  --input-dir dataset/nexar \
  --output-dir dataset/yolo_nexar_frames \
  --source-splits train \
  --frames-per-video 5 \
  --clean-output
```

Kalau belum ada video di `dataset/nexar/train`, unduh dulu video training:

```bash
python download_nexar.py --source-splits train --max-videos 100
```

Output yang dibuat:

```text
dataset/yolo_nexar_frames/
├── data.yaml
├── annotation_manifest.csv
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Anotasi gambar di `images/train` dan `images/val` memakai CVAT, Label Studio,
Roboflow, atau tool anotasi lain. Simpan label dalam format YOLO pada folder
`labels/train` dan `labels/val` dengan nama file yang sama seperti gambar.
Untuk baseline cepat, isi pseudo-label otomatis dari YOLO pretrained:

```bash
python src/autolabel_yolo_dataset.py \
  --dataset-dir dataset/yolo_nexar_frames \
  --model yolo11n.pt \
  --conf 0.25 \
  --imgsz 640 \
  --batch 16 \
  --preview-dir outputs/autolabel_preview \
  --overwrite
```

Preview bbox akan tersimpan di `outputs/autolabel_preview`. Periksa beberapa
gambar preview sebelum training; pseudo-label otomatis sebaiknya dianggap
baseline, bukan pengganti anotasi manual berkualitas.

Format label YOLO:

```text
class_id x_center y_center width height
```

Semua nilai koordinat harus ternormalisasi 0 sampai 1. Urutan class default
di `data.yaml` adalah:

```text
0 person
1 bicycle
2 car
3 motorcycle
4 bus
5 truck
```

Setelah label terisi, jalankan training:

```bash
yolo detect train \
  model=yolo11n.pt \
  data=dataset/yolo_nexar_frames/data.yaml \
  imgsz=640 \
  epochs=50 \
  batch=16
```

Gunakan hanya split `train` Nexar untuk training. Split `test-public` atau
`test-private` sebaiknya dipakai untuk evaluasi, bukan untuk melatih model.
Jika Ultralytics menampilkan error loading data dari `images/train`, cek dulu
bahwa folder tersebut berisi file `.jpg`; error itu biasanya berarti ekstraksi
frame belum berhasil atau split yang dipilih belum punya video lokal.

## Output video

- class dan tracking ID;
- trajectory;
- risk score biner: 0 untuk `SAFE`, 100 untuk `RISK`;
- status `SAFE` atau `RISK`.

## Output CSV

```text
frame,time_seconds,track_id,object,detection_confidence,risk_score,status,ttc_seconds,bbox_growth_rate,bbox_width_growth_rate,bbox_area_growth_rate,reason
```

## Struktur

```text
nexar_yolo_nearmiss/
├── dataset/
├── outputs/
├── src/
│   ├── batch_process.py
│   ├── infer.py
│   └── risk_engine.py
├── dashboard.py
├── download_nexar.py
├── install.sh
├── install_windows.bat
├── requirements.txt
└── README.md
```

## Batasan

- YOLO hanya digunakan untuk mendeteksi objek dan tracking.
- Dataset tidak dimasukkan ke ZIP karena ukurannya sekitar 31 GB dan memiliki
  ketentuan lisensi tersendiri.
- Near-miss ground truth Nexar adalah label tingkat video. Baseline ini
  menghasilkan event tingkat objek dari heuristik geometris.
- Untuk hasil penelitian lebih kuat, tambahkan model temporal yang dilatih
  memakai label dan `time_of_event` Nexar.
