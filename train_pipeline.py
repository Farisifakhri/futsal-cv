import os
import shutil
from pathlib import Path
import torch
import numpy as np

# --- Fix untuk numpy 2.0+ compatibility dengan ultralytics ---
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

# --- Fix untuk PyTorch 2.6+ weights_only default ---
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from ultralytics import YOLO
from src.preprocess import preprocess_dataset
from src.augment import augment_dataset

def create_augmented_yaml(output_yaml_path="data_augmented/data.yaml"):
    """
    Membuat file konfigurasi data.yaml khusus untuk dataset yang sudah di-preprocess dan di-augmentasi.
    """
    abs_aug_dir = os.path.abspath(os.path.dirname(output_yaml_path))
    yaml_content = f"""# Futsal-CV Dataset (Preprocessed & Augmented 4-Classes)
path: {abs_aug_dir}
train: train/images
val: valid/images
test: test/images

nc: 4
names:
  0: ball
  1: goalkeeper
  2: player
  3: referee
"""
    os.makedirs(os.path.dirname(output_yaml_path), exist_ok=True)
    with open(output_yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"Config created at: {os.path.abspath(output_yaml_path)}")

def run_pipeline():
    """
    Menjalankan alur lengkap:
    Step 1: PreprocessingCitra (CLAHE Contrast Enhancement)
    Step 2: Augmentasi Data Offline (Flip, Brightness/Contrast, HSV Shift)
    Step 3: Modeling & Training YOLOv8n
    """
    # print("==================================================")
    # print("Pipeline Training Custom Model")
    # print("==================================================")

    # 1. Preprocessing Data
    processed_dir = preprocess_dataset(src_root="data", dst_root="data_processed")

    # 2. Augmentasi Data
    augmented_dir = augment_dataset(src_root="data_processed", dst_root="data_augmented", augment_train_only=True)

    # 3. Buat data.yaml untuk dataset ter-augmentasi
    yaml_path = os.path.join(augmented_dir, "data.yaml")
    create_augmented_yaml(yaml_path)

    # 4. Modeling menggunakan YOLOv8n
    model_path = 'models/yolov8n.pt'
    if not os.path.exists(model_path):
        os.makedirs('models', exist_ok=True)

    stable_path = os.path.join('models', 'best_futsal.pt')

    print("==================================================")
    print("[Step 3/3] Memulai Training Model YOLOv8")
    print("==================================================")

    model = YOLO(model_path)

    results = model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        device=0 if torch.cuda.is_available() else 'cpu',
        project='futsal',
        name='yolov8n_futsal_pipeline',
        exist_ok=True,
        patience=15,
        mosaic=1.0,     # Augmentasi Mosaic untuk objek kecil (bola)
        mixup=0.15,     # Augmentasi Mixup untuk occlusion pemain
        hsv_h=0.015,    # Hue jitter
        hsv_s=0.7,      # Saturation jitter
        hsv_v=0.4,      # Value jitter
        degrees=10.0,   # Rotasi acak
        fliplr=0.5,     # Flip left-right
    )

    print("==================================================")
    print("Training Pipeline Finished!")
    print("==================================================")

    possible_best_paths = [
        os.path.join('futsal', 'yolov8n_futsal_pipeline', 'weights', 'best.pt'),
        os.path.join('runs', 'futsal', 'yolov8n_futsal_pipeline', 'weights', 'best.pt'),
    ]

    trained_best = None
    for p in possible_best_paths:
        if os.path.exists(p):
            trained_best = p
            break

    if trained_best and os.path.exists(trained_best):
        shutil.copy(trained_best, stable_path)
        print(f"Model terbaik berhasil disalin ke: {os.path.abspath(stable_path)}")
    else:
        print(f"[WARN] File model terbaik tidak ditemukan di: {possible_best_paths}")

if __name__ == "__main__":
    run_pipeline()
