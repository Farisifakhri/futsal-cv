# 📊 Laporan Analisis Proyek & Rencana Metodologi: Futsal CV

## 📑 Ringkasan Proyek
Proyek **Futsal Computer Vision (`futsal-cv`)** adalah sistem otomatis berbasis *Computer Vision* dan *Deep Learning* yang dirancang untuk mendeteksi, mengklasifikasikan, dan mengolah entitas pada pertandingan futsal dari rekaman video.

---

## 🎯 Fokus & Target Deteksi (4 Kelas Utama)
Sesuai dengan keputusan pengembangan terbaru, proyek ini menyederhanakan dan memfokuskan deteksi objek ke dalam **4 kelas utama**:

| No | Nama Kelas | Kode Label | Deskripsi |
|---|---|---|---|
| 1 | **`player`** | `0` | Pemain lapangan dari kedua tim |
| 2 | **`goalkeeper`** | `1` | Penjaga gawang / Kiper (biasanya memiliki jersey berbeda) |
| 3 | **`referee`** | `2` | Wasit pertandingan |
| 4 | **`ball`** | `3` | Bola futsal |

> **Catatan Perubahan**: Kelas sebelumnya seperti `outfield-players`, `players`, dan `official` dirampingkan menjadi `player` & `referee` untuk meningkatkan konsistensi annotation, menurunkan *false positive*, serta menambahkan kelas krusial **`ball`** yang sebelumnya belum dilatih secara khusus.

---

## 🔄 Metodologi Pengembangan: CRISP-DM & CRISP-ML(Q)

Metodologi utama yang digunakan adalah **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*), disempurnakan dengan **CRISP-ML(Q)** (*Quality Assurance for Machine Learning*) untuk memastikan kualitas data dan model Machine Learning.

```
+-----------------------------------------------------------------------+
|                         CRISP-DM / CRISP-ML(Q)                        |
|                                                                       |
|  [1. Business] --> [2. Data Analysis] --> [3. Data Prep & Labeling]  |
|         ^                                            |                |
|         |                                            v                |
|  [6. Deployment] <-- [5. Evaluation]   <-- [4. Modeling YOLO]         |
+-----------------------------------------------------------------------+
```

### Detail Tahapan:

1. **Business Understanding (Pemahaman Masalah)**
   - **Tujuan**: Membangun sistem otomatis analisis taktis pertandingan futsal.
   - **Output**: Deteksi real-time 4 kelas entitas & ekstraksi posisi objek.

2. **Data Understanding (Pemahaman Data)**
   - Eksplorasi karakteristik video futsal (kamera sudut tinggi / *tactical view*, pencahayaan lapangan indoor/outdoor, resolusi video).
   - Identifikasi tantangan utama: ukuran bola futsal yang kecil (*small object*), gerakan cepat (*motion blur*), dan tumpuk tindih antar pemain (*occlusion*).

3. **Data Preparation (Persiapan Data & Re-annotation)**
   - Ekstraksi frame video menggunakan script `src/preprocess.py`.
   - **Re-labeling 4 kelas** menggunakan Roboflow / LabelImg.
   - Augmentasi data: Mosaic, Mixup, Random Flip, Rotation, HSV contrast adjustment.

4. **Modeling (Pelatihan YOLO)**
   - Menggunakan transfer learning arsitektur **YOLOv8** / **YOLOv11** (Nano / Small / Medium).
   - Konfigurasi parameter training pada `train.py`: `imgsz=640` (atau `1280` khusus deteksi bola presisi tinggi), `epochs=50-100`, `batch=16`.

5. **Evaluation (Evaluasi Performa)**
   - Mengukur metrik standar *Object Detection*:
     - `mAP@50` (Mean Average Precision pada IoU 0.5)
     - `mAP@50-95`
     - `Precision` & `Recall` per kelas (terutama mengevaluasi Recall untuk kelas `ball`).
   - Analisis *Confusion Matrix* untuk memastikan tidak ada salah deteksi antara `player` dan `referee`/`goalkeeper`.

6. **Deployment & Quality Assurance (CRISP-ML(Q))**
   - Integrasi model `best.pt` pada `main.py`.
   - Penerapan **ByteTrack / BoT-SORT** untuk *multi-object tracking* (memberikan ID konsisten pada pemain & bola).
   - Pengujian kualitas inferences (FPS speed, latensi, error handling).

---

## 💡 Saran & Rekomendasi Tambahan dari AI Agent

### 1. Rekomendasi Metodologi Tambahan: **CRISP-ML(Q) & MLOps Lifecycle**
- **Mengapa CRISP-ML(Q)?** CRISP-DM standar dibuat pada era data mining tradisional. Untuk proyek AI/CV modern, CRISP-ML(Q) menambahkan aspek **Data Quality Checks** (memastikan label tidak *corrupt*) dan **Model Monitoring** (memastikan model tidak *drift* saat diuji pada video dengan kondisi pencahayaan/lapangan berbeda).

### 2. Strategi Khusus Deteksi Bola (`ball`)
- **Tantangan**: Bola futsal adalah objek yang sangat kecil (*small object*) dan cepat.
- **Solusi**:
  - Gunakan `imgsz=1280` saat training & inference jika memungkinkan, ATAU gunakan arsitektur YOLO yang memiliki *extra P2 detection head* (layer khusus objek sangat kecil).
  - Lakukan augmentasi khusus bola (misal: *random cropping* fokus ke area sekitar kaki pemain).

### 3. Pemisahan Warna Jersey / Tim (Future Enhancement)
- Setelah 4 kelas terdeteksi (`player`, `goalkeeper`, `referee`, `ball`), tambahkan modul **K-Means Clustering (Color Extraction)** pada bounding box `player` untuk membedakan Tim A dan Tim B secara otomatis berdasarkan warna jersey.

### 4. Pelacakan Objek (*Multi-Object Tracking*)
- Menggabungkan deteksi YOLO dengan **ByteTrack** (yang sudah ada file konfigurasinya `bytetrack_futsal.yaml`) agar setiap pemain mendapatkan `track_id` unik. Ini memungkinkan perhitungan jarak tempuh pemain, *heatmap*, dan penguasaan bola (*ball possession*).

---
*Laporan ini disimpan di `.agents/AGENTS.md` dan `docs/project_analysis.md` sebagai acuan konteks seluruh agent.*
