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
*(Efek kotak-kotak tabel biar rapi)*

| Komponen Analisis | Status Pengerjaan | Deskripsi |
| :--- | :---: | :--- |
| **Deteksi Objek Dasar** | ✅ Selesai | Mendeteksi entitas (orang, bola) menggunakan pre-trained YOLOv8. |
| **Ekstraksi Dataset** | ✅ Selesai | Memotong video futsal menjadi *frame* gambar siap *labeling*. |
| **Custom Model Futsal** | ✅ Selesai | Melatih model kustom untuk mengenali Wasit, Pemain, Official, dan Kiper. |
| **Logika Pelanggaran** | ⏳ *In Progress* | Mendeteksi benturan (*foul*) dan *gesture* wasit di lapangan. |
| **Dashboard UI** | ⏳ *In Progress* | Antarmuka interaktif berbasis Web (Gradio/Streamlit). |

---

## 📊 Hasil Validasi Custom Model (YOLOv8n)
Model dilatih selama **50 Epochs** menggunakan dataset berlabel buatan mandiri melalui Roboflow. Berikut metrik performa awal yang dicapai:

* **Wasit (`referee`)**: `mAP50 = 0.741 (74.1%)` 🏆 *(Akurasi tertinggi!)*
* **Pemain (`players`)**: `mAP50 = 0.732 (73.2%)`
* **Petugas Meja (`official`)**: `mAP50 = 0.674 (67.4%)`

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

### 💻 Cara Mendorong Perubahan ke GitHub:
Setelah file `README.md` Aa simpan di VS Code, jalankan baris perintah ini di terminal belahan kanan Aa kemarin:

1. `git add README.md`
2. `git commit -m "Docs: Update README target portfolio and add custom model validation results"`
3. `git push origin main`

Silakan di-*refresh* repositori GitHub-nya, Aa! Dijamin profil proyeknya sekarang terlihat semakin profesional dengan rapor mAP50 yang mentereng wkwk.