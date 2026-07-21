# Project Context & Rules: Futsal Computer Vision (futsal-cv)

## 📌 Project Overview
Sistem deteksi objek berbasis Computer Vision dan Deep Learning untuk menganalisis pertandingan futsal dari rekaman video.

## 🎯 Target & Focus (4 Main Classes)
Proyek ini berfokus pada **Object Detection** dengan **4 kelas utama**:
1. `player` (Pemain lapangan)
2. `goalkeeper` (Kiper)
3. `referee` (Wasit)
4. `ball` (Bola futsal)

> *Note*: Kelas lama (`outfield-players`, `players`, `official`) dirampingkan dan disesuaikan menjadi 4 kelas standar di atas agar deteksi lebih fokus dan presisi.

## 🔄 Methodology: CRISP-DM & CRISP-ML(Q)
Proyek ini mengadopsi metodologi **CRISP-DM** (Cross-Industry Standard Process for Data Mining) yang diperkaya dengan **CRISP-ML(Q)** (Quality Assurance for Machine Learning):

1. **Business/Problem Understanding**:
   - Deteksi otomatis entitas futsal (pemain, kiper, wasit, dan bola) untuk analisis taktis/statistik pertandingan.
2. **Data Understanding**:
   - Mengumpulkan video pertandingan futsal (rekaman kamera *tactical view*).
   - Mengidentifikasi kendala data: ukuran bola yang kecil, *occlusion* (pemain tumpuk tindih), dan *motion blur*.
3. **Data Preparation**:
   - Ekstraksi frame video (`src/preprocess.py`).
   - Labelling / Re-annotation 4 kelas melalui Roboflow.
   - Data Augmentation (Mosaic, HSV adjust, scaling, random crop).
4. **Modeling**:
   - Fine-tuning arsitektur YOLOv8/YOLOv11 menggunakan transfer learning (`train.py`).
   - Eksperimen hiperparameter (learning rate, image size `imgsz=640` atau `1280`).
5. **Evaluation**:
   - Pengukuran metrik: `mAP50`, `mAP50-95`, `Precision`, `Recall`, dan *Confusion Matrix* per kelas.
   - Evaluasi khusus deteksi objek kecil (*small object detection*) untuk bola (`ball`).
6. **Deployment & Quality Assurance (CRISP-ML(Q))**:
   - Inference pipeline (`main.py` & `src/inference.py`).
   - Integrasi tracking (ByteTrack / BoT-SORT).
   - Antarmuka pengguna (Dashboard Gradio/Streamlit).

## 🛠️ Technical Stack
- **Language**: Python 3.11
- **Framework**: Ultralytics YOLO (v8 / v11), PyTorch
- **Computer Vision**: OpenCV, NumPy
- **Tracking**: ByteTrack / Lapx
- **UI**: Gradio / Streamlit
