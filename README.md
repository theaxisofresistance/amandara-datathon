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

## 5. Evaluasi Model 1

Evaluasi Model 1 dilakukan pada level video. Ground truth diambil dari struktur
folder Nexar: `positive` dianggap collision/near-collision dan `negative`
dianggap normal driving. Prediksi diambil dari CSV hasil batch processing.

Jalankan inference batch terlebih dahulu:

```bash
python src/batch_process.py \
  --input-dir dataset/nexar \
  --output-dir outputs/batch
```

Lalu evaluasi:

```bash
python src/evaluate_model1.py \
  --dataset-dir dataset/nexar \
  --predictions-dir outputs/batch \
  --threshold 50 \
  --output outputs/model1_evaluation.csv \
  --summary outputs/model1_evaluation_summary.json \
  --sweep-output outputs/model1_threshold_sweep.csv
```

Output evaluasi:

```text
outputs/model1_evaluation.csv          prediksi per video
outputs/model1_evaluation_summary.json accuracy, precision, recall, F1, confusion matrix
outputs/model1_threshold_sweep.csv     metrik untuk beberapa threshold
```

Catatan: jika `missing_predictions` besar, berarti belum semua video punya CSV
hasil inference di `outputs/batch`.

## 6. Dashboard

```bash
streamlit run dashboard.py
```

Dashboard membaca `outputs/events.csv` atau file CSV yang diunggah.
Jika tersedia, dashboard juga membaca `outputs/frame_summary.csv` untuk risk
timeline per frame.

## 7. Model 1: Pretrained YOLO Baseline

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
5. Output memakai empat kategori skor:
   - `DANGER` untuk score >= 75;
   - `HIGH RISK` untuk score 50-74;
   - `WARNING` untuk score 25-49;
   - `SAFE` untuk score < 25.
6. Kendaraan dikategorikan `SAFE` jika:
   - berada di area A/samping dan hanya lewat;
   - menuju area A lalu hilang dari frame;
   - menuju center tetapi bounding box tetap atau mengecil;
   - membesar tetapi tetap di samping atau bergerak ke samping.
7. Area ego adalah C: bagian bawah-tengah frame, dekat kamera/kendaraan sendiri.
   Area A dan B bukan ego area. Kendaraan masuk `HIGH RISK` atau `DANGER`
   hanya jika bergerak menuju area C dan bounding box membesar. Pertumbuhan
   dihitung dari tinggi, lebar, dan area bounding box karena kendaraan dari
   samping sering membesar terutama pada lebar/area.
   Kendaraan yang membesar di pojok A lalu menghilang tetap `SAFE` karena tidak
   menuju C. `edge_intrusion` hanya dipakai untuk debug CSV, bukan syarat cukup
   untuk alert.

Output model 1:

```text
outputs/result.mp4          semua road user dengan bbox, ID, trajectory, status, score
outputs/events.csv          event kategori HIGH RISK dan DANGER
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

Secara default, video output menampilkan semua kendaraan yang terdeteksi YOLO.
`SAFE` diberi bounding box hijau, `WARNING` kuning, `HIGH RISK` oranye, dan
`DANGER` merah.

Alert banner hanya muncul untuk `HIGH RISK` dan `DANGER`. `SAFE` dan `WARNING`
tetap digambar sebagai bounding box tanpa alert. Tidak ada countdown,
near-miss, atau alert hold. Bounding box mengikuti hasil tracking kendaraan
pada frame berjalan.

## 8. Model 2: Fine-tuned YOLO + ML Risk

Model 2 memakai file terpisah supaya baseline Model 1 tetap bisa dibandingkan.
Alurnya:

1. YOLO11 fine-tuned pada frame Nexar mendeteksi road user.
2. ByteTrack atau BoT-SORT memberi ID tracking.
3. Fitur trajectory, bbox growth, TTC relatif, speed, acceleration, dan zona ego
   dihitung ulang dengan engine fitur yang sama.
4. `src/model2_risk_engine.py` memuat model risk ML dari `.npz`.
5. Jika model risk ML belum ada, inference bisa fallback ke rule Model 1 agar
   detector fine-tuned tetap dapat dites.

Jalankan inference Model 2:

```bash
python src/infer_model2.py \
  --input dataset/nexar/train/NAMA_VIDEO.mp4 \
  --output outputs/model2_result.mp4 \
  --events outputs/model2_events.csv \
  --summary outputs/model2_frame_summary.csv \
  --detector runs/detect/model2_nexar/weights/best.pt \
  --risk-model models/model2_risk_model.npz \
  --tracker bytetrack.yaml \
  --conf 0.25
```

Kalau ingin memastikan risk ML wajib ada dan tidak fallback:

```bash
python src/infer_model2.py \
  --input dataset/nexar/train/NAMA_VIDEO.mp4 \
  --detector runs/detect/model2_nexar/weights/best.pt \
  --risk-model models/model2_risk_model.npz \
  --no-rule-fallback
```

Untuk testing sementara sebelum `best.pt` hasil fine-tuning tersedia, pakai
pretrained detector:

```bash
python src/infer_model2.py \
  --input dataset/nexar/train/NAMA_VIDEO.mp4 \
  --detector yolo11s.pt \
  --allow-pretrained-detector-fallback
```

Catatan: mode ini belum Model 2 penuh karena detector-nya belum fine-tuned.

Train risk model ML dari CSV fitur berlabel:

```bash
python src/train_model2_risk.py \
  --features dataset/model2_features.csv \
  --output models/model2_risk_model.npz
```

CSV training membutuhkan kolom fitur seperti `events.csv`:
`ttc_seconds`, `lane_distance`, `relative_distance`, `speed_px_s`,
`horizontal_speed_px_s`, `vertical_speed_px_s`, `acceleration_px_s2`,
`bbox_growth_rate`, `bbox_width_growth_rate`, `bbox_area_growth_rate`,
`trajectory_intersection`, `impact_zone_intersection`, `in_ego_corridor`,
`in_impact_zone`, `near_enough`, `moving_toward_ego_center`,
`approaching_camera`, dan `collision_candidate`.

Label bisa memakai salah satu kolom: `label`, `risk_label`, `target`,
`collision`, `is_collision`, `status`, atau `risk_score`. Nilai positif:
`DANGER`, `HIGH RISK`, `RISK`, `NEAR-MISS`, `1`, `TRUE`. Nilai negatif:
`SAFE`, `WARNING`, `NORMAL`, `0`, `FALSE`.

Output Model 2 tetap memakai kategori visual yang sama:
`SAFE`, `WARNING`, `HIGH RISK`, dan `DANGER`.

## 9. Integrasi Dataset untuk Training YOLO

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
  project=runs/detect \
  name=model2_nexar \
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
- risk score 0-100;
- status `SAFE`, `WARNING`, `HIGH RISK`, atau `DANGER`.

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
	│   ├── evaluate_model1.py
	│   ├── infer.py
	│   ├── infer_model2.py
	│   ├── model2_risk_engine.py
	│   ├── risk_engine.py
	│   └── train_model2_risk.py
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
