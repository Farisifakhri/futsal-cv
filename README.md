<div align="center">
  <h1>⚽ Futsal Video Analysis with YOLOv8</h1>
  <p>
    <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/YOLO-v8-yellow?style=for-the-badge&logo=ultralytics&logoColor=black" alt="YOLOv8" />
    <img src="https://img.shields.io/badge/OpenCV-4.8.0-green?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
    <img src="https://img.shields.io/badge/Status-Development-orange?style=for-the-badge" alt="Status" />
  </p>
  <p><i>Sistem deteksi otomatis untuk pemain, bola, dan wasit futsal menggunakan AI berbasis Computer Vision.</i></p>
</div>

---

## 🎯 Target Fitur & Portofolio

| Komponen Analisis | Status Pengerjaan | Deskripsi |
| :--- | :---: | :--- |
| **Deteksi Objek Dasar** | ✅ Selesai | Mendeteksi entitas dasar (orang, bola) menggunakan pre-trained YOLOv8. |
| **Ekstraksi Dataset** | ✅ Selesai | Memotong video futsal menjadi *frame* gambar siap *labeling*. |
| **Custom Model Futsal** | ✅ Selesai | Melatih model kustom untuk mengenali Wasit, Pemain, Official, dan Kiper. |
| **Logika Pelanggaran** | ⏳ *In Progress* | Mendeteksi benturan (*foul*) dan *gesture* wasit di lapangan. |
| **Dashboard UI** | ⏳ *In Progress* | Antarmuka interaktif berbasis Web (Gradio/Streamlit). |

---

## 📊 Hasil Validasi Custom Model (YOLOv8n)
Model dilatih selama **50 Epochs** menggunakan dataset berlabel mandiri melalui Roboflow. Berikut adalah hasil metrik performa riil berdasarkan evaluasi **Precision-Recall Curve (mAP@0.5)**:

*   **Pemain (`players`)**: `mAP50 = 0.841 (84.1%)` 🏃‍♂️ *(Akurasi tertinggi!)*
*   **Wasit (`referee`)**: `mAP50 = 0.799 (79.9%)` 🏁 *(Sangat krusial untuk logika deteksi pelanggaran)*
*   **Petugas Meja (`official`)**: `mAP50 = 0.665 (66.5%)` 🪑
*   **Kiper (`keeper`)**: `mAP50 = 0.386 (38.6%)` 🧤 *(Catatan: Perlu penambahan variasi data visual baju kiper)*
*   **Rata-rata Keseluruhan (`all classes`)**: `mAP50 = 0.673 (67.3%)`

---

## 📂 Struktur Proyek
```text
C:\futsal-cv\
├── data/                  # Folder dataset (images & labels untuk train/valid)
├── models/                # Tempat menyimpan bobot (weights) model YOLO standar
├── notebook/              # Coret-coretan eksperimen (.ipynb)
├── runs/                  # Hasil output training model kustom (.pt) dan log performa
├── src/                   # Kumpulan script logika utama (preprocess, inference)
├── .gitignore             # File untuk mengabaikan file gajah dari Git
├── main.py                # Script utama untuk menjalankan deteksi pada video
├── train.py               # Script untuk melatih/melanjutkan training model kustom
└── requirements.txt       # Daftar pustaka (library) Python
⚙️ Persiapan dan Instalasi  
Pastikan menggunakan Python 3.11 agar ekosistem Machine Learning stabil.  

1. Buat & Aktifkan Virtual Environment  
Bash
py -3.11 -m venv venv
.\venv\Scripts\activate   # Untuk Windows
2. Install Dependencies  
Bash
pip install -r requirements.txt
🚀 Cara Penggunaan  
Uji Coba Model Custom pada Video
Untuk menjalankan deteksi pertandingan menggunakan model kustom terbaik hasil training:

Bash
python main.py
Melatih Ulang / Melanjutkan Latihan Model
Jika ingin melakukan resume training dari checkpoint terakhir:

Bash
python train.py

---

### 💡 Poin Utama yang Diperbaiki:
1. **Pembaruan Data Metrik Akurasi**: Menyelaraskan angka mAP50 dengan hasil grafik asli `PR_curve.png`. Akurasi kelas `players` Aa ternyata tembus **84.1%** dan kelas `referee` tembus **79.9%** (ini tinggi banget untuk ukuran model nano dengan 50 epochs!).
2. **Penambahan Kelas `keeper`**: Ditambahkan ke daftar hasil validasi beserta catatan objektifnya agar pembaca/reviewer tahu kalau Aa paham cara melakukan evaluasi model yang jujur.
3. **Format Code Block**: Merapikan blok teks perintah instalasi dan struktur direktori agar pembacaan kode (`text` dan `bash`) ter-render dengan sempurna di halaman utama GitHub Aa.

Tinggal masukkan perintah Git kemarin ke terminal kanan VS Code, lalu silakan di-push ke GitHub, Aa!
