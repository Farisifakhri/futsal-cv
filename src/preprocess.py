import cv2
import numpy as np

def prepare_frame(frame, width=640, height=640):
    """
    Melakukan resize dan normalisasi ringan pada frame video.
    """
    # Resize frame ke ukuran standar YOLO (640x640)
    resized = cv2.resize(frame, (width, height))
    
    # Opsional: Jika video pertandingan agak gelap, bisa tambah kontras di sini
    # enhancement = cv2.convertScaleAbs(resized, alpha=1.2, beta=10)
    
    return resized

def extract_frames(video_path, output_folder, gap=30):
    """
    Mengambil frame dari video untuk dijadikan dataset.
    gap=30 artinya mengambil 1 frame setiap 30 frame (sekitar 1 detik).
    """
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % gap == 0:
            cv2.imwrite(f"{output_folder}/frame_{saved_count}.jpg", frame)
            saved_count += 1
        count += 1
    cap.release()