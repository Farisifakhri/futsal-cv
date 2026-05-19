import torch
import os

if hasattr(torch.serialization, 'add_safe_globals'):
    try:
        from ultralytics.nn.tasks import DetectionModel
        torch.serialization.add_safe_globals([DetectionModel])
    except ImportError:
        pass

from ultralytics import YOLO

def resume_training():
    # KUNCI UTAMA: Kita tembak langsung ke folder yolov8n_futsal7
    last_model_path = r'C:\futsal-cv\runs\detect\runs\futsal\yolov8n_futsal7\weights\last.pt'
    
    if os.path.exists(last_model_path):
        print("--- Checkpoint Ke-7 Ditemukan! Melanjutkan Training ---")
        model = YOLO(last_model_path)
        
        # Perintah sakti untuk melanjutkan sisa epoch dari titik terakhir
        model.train(resume=True)
    else:
        print(f"Error: File checkpoint masih tidak ditemukan di:\n -> {last_model_path}")
        print("\nCoba Aa cek sekali lagi ke dalam folder:")
        print("C:\\futsal-cv\\runs\\detect\\runs\\futsal\\")
        print("Apakah nama foldernya benar 'yolov8n_futsal7' atau ada angka lain, Aa?")

if __name__ == "__main__":
    resume_training()