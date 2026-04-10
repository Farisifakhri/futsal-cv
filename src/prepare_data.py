import os
from src.preprocess import extract_frames

# 1. Pastikan folder tujuan ada sesuai .gitignore
os.makedirs('data/images/train', exist_ok=True)

# 2. Jalankan ekstraksi
# gap=30 berarti ambil 1 frame tiap detik (asumsi video 30fps)
print("Sedang mengekstrak frame dari video...")
extract_frames(
    video_path='data/tes_futsal.mp4', 
    output_folder='data/images/train', 
    gap=30
)
print("Selesai! Frame tersimpan di data/images/train")
