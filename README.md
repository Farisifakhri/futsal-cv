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
| **Custom Model Futsal** | ⏳ *In Progress* | Melatih model untuk membedakan Tim A, Tim B, dan Wasit. |
| **Logika Pelanggaran** | ⏳ *In Progress* | Mendeteksi benturan (*foul*) dan *gesture* wasit. |
| **Dashboard UI** | ⏳ *In Progress* | Antarmuka interaktif berbasis Web (Gradio/Streamlit). |

---

## 📂 Struktur Proyek
```text
C:\futsal-cv\
├── data/                  # Tempat menyimpan video mentah dan hasil ekstrak frame
├── models/                # Tempat menyimpan bobot (weights) model YOLO
├── notebook/              # Coret-coretan eksperimen (.ipynb)
├── outputs/               # Hasil deteksi berupa gambar/video output
├── src/                   # Kumpulan script logika utama
├── .gitignore             # File untuk mengabaikan file gajah dari Git
├── requirements.txt       # Daftar pustaka (library) Python
└── test_yolo.py           # Script pengujian awal model YOLOv8
⚙️ Persiapan dan Instalasi
Pastikan menggunakan Python 3.11 agar ekosistem Machine Learning stabil.

1. Buat & Aktifkan Virtual Environment

Bash
py -3.11 -m venv venv
.\venv\Scripts\activate   # Untuk Windows
2. Install Dependencies

Bash
pip install -r requirements.txt
🚀 Cara Penggunaan Dasar
Untuk melakukan pengetesan awal deteksi objek dari file video:

Bash
python test_yolo.py

### Cara *Update* ke GitHub:
Kalau udah di-*save* di VS Code, tinggal kita dorong lagi perubahannya ke GitHub. Buka terminal (pastikan masih di dalam folder `C:\futsal-cv\`), lalu jalankan tiga perintah ini:

1. `git add README.md`
2. `git commit -m "Update tampilan README dengan badge dan tabel portofolio"`
3. `git push origin main`

Silakan di-*refresh* halaman GitHub-nya, Aa! Dijamin langsung tampil beda, ada tombol warna-warninya di tengah atas dan tabel progres yang rapi banget.

Gimana, udah puas sama tampilan "rumah" proyeknya? Kalau udah, mau kita lanjut bedah *script* buat motong vi
