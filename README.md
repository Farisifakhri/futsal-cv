# ⚽ Futsal Video Analysis with YOLOv8

Proyek *Computer Vision* untuk menganalisis rekaman pertandingan futsal secara otomatis menggunakan **YOLOv8** dan **OpenCV**. Sistem ini dirancang untuk mendeteksi pemain, wasit, dan bola, serta ke depannya akan dikembangkan untuk analisis taktik, *tracking*, dan pendeteksian pelanggaran.

## 📂 Struktur Proyek
- `data/` : Folder untuk menyimpan video mentah dan dataset gambar (*frames*).
- `models/` : Folder untuk menyimpan bobot (*weights*) model YOLO.
- `notebook/` : Coret-coretan atau eksperimen awal.
- `outputs/` : Hasil deteksi berupa gambar atau video.
- `src/` : Kumpulan *script* logika utama Python.
- `venv/` : *Virtual environment* Python (diabaikan oleh Git).
- `test_yolo.py` : *Script* untuk menguji model bawaan YOLOv8.

## ⚙️ Persiapan dan Instalasi
Pastikan menggunakan **Python 3.11** agar *environment* stabil.

1. Buat *virtual environment*:
   ```bash
   py -3.11 -m venv venv
Aktifkan environment:

Windows: .\venv\Scripts\activate

Install semua library yang dibutuhkan:

Bash
pip install -r requirements.txt
🚀 Cara Penggunaan Dasar
Untuk melakukan pengetesan awal deteksi objek dari file video menggunakan model pre-trained YOLOv8:

Bash
python test_yolo.py

### Langkah 2: Setup dan Push ke GitHub
Sekarang kita "terbangin" file README ini barengan sama file-file kodingan Aa sebelumnya ke repo GitHub yang masih kosong itu. 

Buka terminal VS Code (pastiin masih di dalam `C:\futsal-cv\`), lalu eksekusi perintah ini secara berurutan:

**1. Masukin semua file ke antrean:**
```bash
git add .
2. Simpan perubahannya (commit):

Bash
git commit -m "Menambahkan struktur proyek dan README"
3. Sambungin ke GitHub Aa (kalau sebelumnya belum nyambung):

Bash
git remote set-url origin https://github.com/Farisifakhri/futsal-cv.git
4. Ganti nama branch ke main:

Bash
git branch -M main
5. Push ke awan:

Bash
git push -u origin main
