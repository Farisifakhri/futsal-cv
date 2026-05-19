import torch
import os

# --- Tambahkan penanganan keamanan PyTorch 2.6+ di bagian paling atas ---
if hasattr(torch.serialization, 'add_safe_globals'):
    try:
        from ultralytics.nn.tasks import DetectionModel
        torch.serialization.add_safe_globals([DetectionModel])
    except ImportError:
        pass
# ---------------------------------------------------------------------

from ultralytics import YOLO

def train_custom_model():
    # 1. Inisialisasi model YOLOv8 Nano (Pre-trained)
    model_path = 'models/yolov8n.pt'
    if not os.path.exists(model_path):
        os.makedirs('models', exist_ok=True)
        
    model = YOLO(model_path)

    print("--- Memulai Proses Training Custom Model Futsal ---")
    
    # 2. Jalankan Training
    results = model.train(
        data='data/data.yaml',      # File konfigurasi dataset
        epochs=50,                  # Jumlah iterasi latihan
        imgsz=640,                  # Ukuran gambar standar
        batch=16,                   # Jumlah gambar per batch
        device='0' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu', # Gunakan GPU jika ada
        project='runs/futsal',      # Folder penyimpanan hasil training
        name='yolov8n_futsal'
    )
    
    print("--- Training Selesai! ---")
    print("Model terbaik disimpan di: runs/futsal/yolov8n_futsal/weights/best.pt")

if __name__ == "__main__":
    train_custom_model()
    