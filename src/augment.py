import cv2
import numpy as np
import os
import shutil
import random
from pathlib import Path

# ============================================================
# Kelas minoritas yang perlu oversampling lebih banyak
# ============================================================
MINORITY_CLASSES = {0, 1}  # 0=ball, 1=goalkeeper

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

def random_scale_crop(image, bboxes, scale_range=(0.6, 0.9)):
    """
    Melakukan random crop dengan zoom-in pada area yang mengandung objek.
    Ini meningkatkan ukuran relatif objek kecil (seperti bola) di beberapa sampel training.
    Hanya bbox yang masih visible (>50% area) setelah crop yang dipertahankan.
    """
    h, w = image.shape[:2]
    scale = random.uniform(*scale_range)
    
    crop_h = int(h * scale)
    crop_w = int(w * scale)
    
    # Random posisi crop
    max_y = h - crop_h
    max_x = w - crop_w
    start_y = random.randint(0, max(0, max_y))
    start_x = random.randint(0, max(0, max_x))
    
    # Crop gambar
    cropped = image[start_y:start_y+crop_h, start_x:start_x+crop_w]
    # Resize kembali ke ukuran asli agar konsisten
    cropped = cv2.resize(cropped, (w, h))
    
    # Sesuaikan bounding boxes
    new_bboxes = []
    for bbox in bboxes:
        cls_id, x_c, y_c, bw, bh = bbox
        
        # Konversi ke pixel coordinates
        px_cx = x_c * w
        px_cy = y_c * h
        px_w = bw * w
        px_h = bh * h
        
        # Sesuaikan relatif terhadap crop area
        new_cx = (px_cx - start_x) / crop_w
        new_cy = (px_cy - start_y) / crop_h
        new_w = px_w / crop_w
        new_h = px_h / crop_h
        
        # Clamp ke [0, 1]
        x1 = max(0, new_cx - new_w / 2)
        y1 = max(0, new_cy - new_h / 2)
        x2 = min(1, new_cx + new_w / 2)
        y2 = min(1, new_cy + new_h / 2)
        
        clipped_w = x2 - x1
        clipped_h = y2 - y1
        
        # Hanya simpan jika bbox masih cukup visible (>50% area asli)
        original_area = new_w * new_h
        clipped_area = clipped_w * clipped_h
        
        if original_area > 0 and (clipped_area / original_area) > 0.5 and clipped_w > 0.005 and clipped_h > 0.005:
            final_cx = round((x1 + x2) / 2, 6)
            final_cy = round((y1 + y2) / 2, 6)
            final_w = round(clipped_w, 6)
            final_h = round(clipped_h, 6)
            new_bboxes.append([cls_id, final_cx, final_cy, final_w, final_h])
    
    return cropped, new_bboxes

def read_yolo_labels(label_path):
    """
    Membaca file label YOLO format .txt.
    Mendukung format Bounding Box standar (5 elemen) maupun Polygon/Segmentation (>5 elemen).
    Jika label berformat Polygon, otomatis dikonversi menjadi Bounding Box presisi (min/max).
    """
    bboxes = []
    if not os.path.exists(label_path):
        return bboxes
        
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                # Format Bounding Box Standar: class_id x_center y_center width height
                cls_id = int(parts[0])
                coords = [float(p) for p in parts[1:5]]
                bboxes.append([cls_id] + coords)
            elif len(parts) > 5:
                # Format Polygon/Segmentation: class_id x1 y1 x2 y2 x3 y3 ...
                cls_id = int(parts[0])
                points = [float(p) for p in parts[1:]]
                xs = points[0::2]
                ys = points[1::2]
                if xs and ys:
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    w = round(x_max - x_min, 6)
                    h = round(y_max - y_min, 6)
                    x_c = round(x_min + w / 2.0, 6)
                    y_c = round(y_min + h / 2.0, 6)
                    bboxes.append([cls_id, x_c, y_c, w, h])
    return bboxes

def write_yolo_labels(label_path, bboxes):
    """
    Menulis bounding box ke file label YOLO format .txt.
    """
    with open(label_path, 'w') as f:
        for bbox in bboxes:
            cls_id, x_c, y_c, w, h = bbox
            f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

def has_minority_class(bboxes):
    """
    Mengecek apakah gambar mengandung kelas minoritas (ball atau goalkeeper).
    """
    return any(int(bbox[0]) in MINORITY_CLASSES for bbox in bboxes)

def augment_dataset(src_root="data_processed", dst_root="data_augmented", augment_train_only=True):
    """
    Mengeksekusi pipeline augmentasi data offline dengan CLASS-AWARE OVERSAMPLING.
    
    Strategi:
    - Gambar yang mengandung kelas minoritas (ball/goalkeeper): 4 variasi augmentasi
      → flip + brightness/contrast + HSV jitter + random scale crop
    - Gambar tanpa kelas minoritas: 2 variasi augmentasi
      → flip + brightness/contrast saja
    
    Ini membantu mengatasi class imbalance dimana ball (414) dan goalkeeper (316)
    jauh lebih sedikit dibanding player (2910).
    """
    src_path = Path(src_root)
    dst_path = Path(dst_root)
    
    splits = ['train', 'valid', 'test']
    print("==================================================")
    print("[Step 2/3] Memulai Data Augmentation Offline (Class-Aware)")
    print(f"Source : {src_path.resolve()}")
    print(f"Target : {dst_path.resolve()}")
    print("==================================================")
    
    total_images_generated = 0
    minority_count = 0
    majority_count = 0
    
    for split in splits:
        src_img_dir = src_path / split / "images"
        src_lbl_dir = src_path / split / "labels"
        
        dst_img_dir = dst_path / split / "images"
        dst_lbl_dir = dst_path / split / "labels"
        
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)
        
        if not src_img_dir.exists():
            continue
            
        img_files = sorted([f for f in src_img_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
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
            
            is_minority = has_minority_class(bboxes)
            
            # 2. Augmentasi 1: Horizontal Flip (semua gambar)
            flipped_img, flipped_boxes = flip_horizontal(img, bboxes)
            flip_name = f"{img_file.stem}_aug_flip{img_file.suffix}"
            cv2.imwrite(str(dst_img_dir / flip_name), flipped_img)
            write_yolo_labels(str(dst_lbl_dir / f"{img_file.stem}_aug_flip.txt"), flipped_boxes)
            total_images_generated += 1
            
            # 3. Augmentasi 2: Brightness & Contrast Adjustment (semua gambar)
            alpha_val = random.uniform(1.1, 1.3)
            beta_val = random.randint(10, 25)
            bc_img = adjust_brightness_contrast(img, alpha=alpha_val, beta=beta_val)
            bc_name = f"{img_file.stem}_aug_bc{img_file.suffix}"
            cv2.imwrite(str(dst_img_dir / bc_name), bc_img)
            write_yolo_labels(str(dst_lbl_dir / f"{img_file.stem}_aug_bc.txt"), bboxes)
            total_images_generated += 1

            # === AUGMENTASI TAMBAHAN UNTUK KELAS MINORITAS ===
            if is_minority:
                minority_count += 1
                
                # 4. Augmentasi 3: HSV Color Jitter (hanya gambar dengan ball/goalkeeper)
                hsv_img = adjust_hsv_jitter(img, h_shift=random.randint(2, 8))
                hsv_name = f"{img_file.stem}_aug_hsv{img_file.suffix}"
                cv2.imwrite(str(dst_img_dir / hsv_name), hsv_img)
                write_yolo_labels(str(dst_lbl_dir / f"{img_file.stem}_aug_hsv.txt"), bboxes)
                total_images_generated += 1
                
                # 5. Augmentasi 4: Random Scale Crop (hanya gambar dengan ball/goalkeeper)
                # Zoom-in meningkatkan ukuran relatif bola di frame
                crop_img, crop_boxes = random_scale_crop(img, bboxes, scale_range=(0.6, 0.85))
                if len(crop_boxes) > 0:  # Pastikan masih ada bbox setelah crop
                    crop_name = f"{img_file.stem}_aug_crop{img_file.suffix}"
                    cv2.imwrite(str(dst_img_dir / crop_name), crop_img)
                    write_yolo_labels(str(dst_lbl_dir / f"{img_file.stem}_aug_crop.txt"), crop_boxes)
                    total_images_generated += 1
            else:
                majority_count += 1

    print("==================================================")
    print(f"✅ Data Augmentation selesai! Total {total_images_generated} gambar dalam dataset akhir.")
    print(f"Gambar train dengan kelas minoritas (ball/goalkeeper): {minority_count} → 5 variasi/gambar")
    print(f"Gambar train tanpa kelas minoritas: {majority_count} → 3 variasi/gambar")
    print("==================================================")
    return dst_path

if __name__ == "__main__":
    augment_dataset()
