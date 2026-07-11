import torch
import os
import shutil
from datetime import datetime
import numpy as np

# --- FIX: numpy 2.0+ menghapus np.trapz, ultralytics 8.1.0 masih memanggilnya ---
# np.trapezoid adalah pengganti resmi dengan fungsi yang identik.
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
# ---------------------------------------------------------------------

# --- FIX: PyTorch 2.6+ mengubah default weights_only=True yang terlalu ketat ---
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
# ---------------------------------------------------------------------

from ultralytics import YOLO

def backup_previous_model(stable_path, backup_dir='models/history'):
    """Backup model lama sebelum ketimpa training baru."""
    if os.path.exists(stable_path):
        os.makedirs(backup_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f'best_futsal_{timestamp}.pt'
        backup_path = os.path.join(backup_dir, backup_name)
        shutil.copy(stable_path, backup_path)
        print(f"[BACKUP] Model lama disimpan ke: {backup_path}")
        return backup_path
    else:
        print("[BACKUP] Tidak ada model lama untuk di-backup (training pertama kali).")
        return None

def train_custom_model():
    model_path = 'models/yolov8n.pt'
    if not os.path.exists(model_path):
        os.makedirs('models', exist_ok=True)

    stable_path = os.path.join('models', 'best_futsal.pt')
    backup_previous_model(stable_path)

    model = YOLO(model_path)

    print("--- Memulai Proses Training Custom Model Futsal ---")

    results = model.train(
        data='data/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device='0' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',
        project='runs/futsal',
        name='yolov8n_futsal',
        exist_ok=True,
        patience=15,
    )

    print("--- Training Selesai! ---")

    trained_best = os.path.join('runs', 'futsal', 'yolov8n_futsal', 'weights', 'best.pt')

    if os.path.exists(trained_best):
        shutil.copy(trained_best, stable_path)
        print(f"Model terbaik disalin ke path stabil: {os.path.abspath(stable_path)}")
    else:
        print(f"[WARNING] File model tidak ditemukan di: {trained_best}")

if __name__ == "__main__":
    train_custom_model()