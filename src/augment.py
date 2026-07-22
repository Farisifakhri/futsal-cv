import cv2
import numpy as np
import os
import shutil
import random
from pathlib import Path

def flip_horizontal(image, bboxes):
    """
    Membalik gambar secara horizontal (left-right flip)
    dan menyesuaikan koordinat bounding box YOLO (x_center_new = 1.0 - x_center).
    """
    flipped_image = cv2.flip(image, 1)
    flipped_bboxes = []
    
    for bbox in bboxes:
        cls_id, x_center, y_center, width, height = bbox
        new_x_center = round(1.0 - x_center, 6)
        flipped_bboxes.append([cls_id, new_x_center, y_center, width, height])
        
    return flipped_image, flipped_bboxes

def adjust_brightness_contrast(image, alpha=1.15, beta=15):
    """
    Menyesuaikan brightness dan contrast secara acak pada citra.
    """
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def adjust_hsv_jitter(image, h_shift=5, s_scale=1.2, v_scale=1.1):
    """
    Melakukan variasi warna (Hue, Saturation, Value) pada HSV color space.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    
    h = (h + h_shift) % 180
    s = np.clip(s * s_scale, 0, 255)
    v = np.clip(v * v_scale, 0, 255)
    
    hsv_merged = cv2.merge([h, s, v]).astype(np.uint8)
    jittered_image = cv2.cvtColor(hsv_merged, cv2.COLOR_HSV2BGR)
    return jittered_image

def read_yolo_labels(label_path):
    """
    Membaca file label YOLO format .txt.
    """
    bboxes = []
    if not os.path.exists(label_path):
        return bboxes
        
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                coords = [float(p) for p in parts[1:5]]
                bboxes.append([cls_id] + coords)
    return bboxes

def write_yolo_labels(label_path, bboxes):
    """
    Menulis bounding box ke file label YOLO format .txt.
    """
    with open(label_path, 'w') as f:
        for bbox in bboxes:
            cls_id, x_c, y_c, w, h = bbox
            f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

def augment_dataset(src_root="data_processed", dst_root="data_augmented", augment_train_only=True):
    """
    Mengeksekusi pipeline augmentasi data offline pada dataset.
    Secara default, augmentasi hanya diterapkan pada folder 'train' untuk menjaga integritas validasi.
    """
    src_path = Path(src_root)
    dst_path = Path(dst_root)
    
    splits = ['train', 'valid', 'test']
    print("==================================================")
    print("[Step 2/3] Memulai Data Augmentation Offline Lokal")
    print(f"Source : {src_path.resolve()}")
    print(f"Target : {dst_path.resolve()}")
    print("==================================================")
    
    total_images_generated = 0
    
    for split in splits:
        src_img_dir = src_path / split / "images"
        src_lbl_dir = src_path / split / "labels"
        
        dst_img_dir = dst_path / split / "images"
        dst_lbl_dir = dst_path / split / "labels"
        
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)
        
        if not src_img_dir.exists():
            continue
            
        img_files = [f for f in src_img_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        print(f"🔄 Processing & Augmenting split '{split}': {len(img_files)} original images...")
        
        for img_file in img_files:
            img = cv2.imread(str(img_file))
            if img is None:
                continue
                
            lbl_file = src_lbl_dir / f"{img_file.stem}.txt"
            bboxes = read_yolo_labels(str(lbl_file))
            
            # 1. Simpan gambar asli ke destination
            cv2.imwrite(str(dst_img_dir / img_file.name), img)
            write_yolo_labels(str(dst_lbl_dir / f"{img_file.stem}.txt"), bboxes)
            total_images_generated += 1
            
            # Jika bukan train dan set augment_train_only True, jangan buat variasi tambahan
            if augment_train_only and split != 'train':
                continue
                
            # 2. Augmentasi 1: Horizontal Flip
            flipped_img, flipped_boxes = flip_horizontal(img, bboxes)
            flip_name = f"{img_file.stem}_aug_flip{img_file.suffix}"
            cv2.imwrite(str(dst_img_dir / flip_name), flipped_img)
            write_yolo_labels(str(dst_lbl_dir / f"{img_file.stem}_aug_flip.txt"), flipped_boxes)
            total_images_generated += 1
            
            # 3. Augmentasi 2: Brightness & Contrast Adjustment
            alpha_val = random.uniform(1.1, 1.3)
            beta_val = random.randint(10, 25)
            bc_img = adjust_brightness_contrast(img, alpha=alpha_val, beta=beta_val)
            bc_name = f"{img_file.stem}_aug_bc{img_file.suffix}"
            cv2.imwrite(str(dst_img_dir / bc_name), bc_img)
            write_yolo_labels(str(dst_lbl_dir / f"{img_file.stem}_aug_bc.txt"), bboxes)
            total_images_generated += 1

            # 4. Augmentasi 3: HSV Color Jitter
            hsv_img = adjust_hsv_jitter(img, h_shift=random.randint(2, 8))
            hsv_name = f"{img_file.stem}_aug_hsv{img_file.suffix}"
            cv2.imwrite(str(dst_img_dir / hsv_name), hsv_img)
            write_yolo_labels(str(dst_lbl_dir / f"{img_file.stem}_aug_hsv.txt"), bboxes)
            total_images_generated += 1

    print("==================================================")
    print(f"✅ Data Augmentation selesai! Total {total_images_generated} gambar dalam dataset akhir.")
    print("==================================================")
    return dst_path

if __name__ == "__main__":
    augment_dataset()
