import cv2
import numpy as np
import os
import shutil
from pathlib import Path

def apply_clahe_contrast(image):
    """
    Meningkatkan kontras pencahayaan citra futsal indoor menggunakan CLAHE
    (Contrast Limited Adaptive Histogram Equalization) pada channel Luminance (LAB color space).
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_image

def prepare_frame(frame, width=640, height=640, apply_clahe=True):
    """
    Melakukan preprocessing pada single frame (resize + CLAHE contrast enhancement).
    """
    resized = cv2.resize(frame, (width, height))
    if apply_clahe:
        resized = apply_clahe_contrast(resized)
    return resized

def extract_frames(video_path, output_folder, gap=30):
    """
    Mengambil frame dari video untuk dijadikan dataset.
    """
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % gap == 0:
            cv2.imwrite(os.path.join(output_folder, f"frame_{saved_count:04d}.jpg"), frame)
            saved_count += 1
        count += 1
    cap.release()

def preprocess_dataset(src_root="data", dst_root="data_processed", apply_clahe=True):
    """
    Mengeksekusi preprocessing (CLAHE contrast enhancement & copy labels)
    pada seluruh split dataset (train, valid, test).
    """
    src_path = Path(src_root)
    dst_path = Path(dst_root)
    
    splits = ['train', 'valid', 'test']
    print("==================================================")
    print("[Step 1/3] Memulai Preprocessing Dataset Lokal")
    print(f"Source : {src_path.resolve()}")
    print(f"Target : {dst_path.resolve()}")
    print("==================================================")
    
    total_images_processed = 0
    
    for split in splits:
        src_img_dir = src_path / split / "images"
        src_lbl_dir = src_path / split / "labels"
        
        dst_img_dir = dst_path / split / "images"
        dst_lbl_dir = dst_path / split / "labels"
        
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)
        
        if not src_img_dir.exists():
            print(f"[WARN] Directory {src_img_dir} tidak ditemukan, skip split '{split}'.")
            continue
            
        img_files = [f for f in src_img_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        print(f"🔄 Processing split '{split}': {len(img_files)} gambar...")
        
        for img_file in img_files:
            # 1. Baca Gambar
            img = cv2.imread(str(img_file))
            if img is None:
                continue
                
            # 2. Preprocessing CLAHE / Resize
            processed_img = prepare_frame(img, apply_clahe=apply_clahe)
            
            # 3. Simpan Gambar Preprocessed
            cv2.imwrite(str(dst_img_dir / img_file.name), processed_img)
            total_images_processed += 1
            
            # 4. Salin Label .txt yang Sesuai
            lbl_file = src_lbl_dir / f"{img_file.stem}.txt"
            if lbl_file.exists():
                shutil.copy2(lbl_file, dst_lbl_dir / lbl_file.name)
                
    print("==================================================")
    print(f"✅ Preprocessing selesai! Total {total_images_processed} gambar diproses.")
    print("==================================================")
    return dst_path

if __name__ == "__main__":
    preprocess_dataset()