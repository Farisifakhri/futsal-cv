import cv2
import os
from src.inference import FutsalDetector
from src.preprocess import prepare_frame

def main():
    # 1. Konfigurasi Path
    video_path = 'data/tes_futsal.mp4'
    model_path = 'models/yolov8n.pt'
    output_dir = 'outputs'
    
    # Pastikan file video tersedia
    if not os.path.exists(video_path):
        print(f"Error: Video tidak ditemukan di {video_path}")
        return

    # 2. Inisialisasi Detector
    detector = FutsalDetector(model_path=model_path)
    
    # 3. Buka Video
    cap = cv2.VideoCapture(video_path)
    
    print("Memulai pemrosesan... Tekan 'q' untuk berhenti.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 4. Preprocessing (dari src/preprocess.py)
        processed_frame = prepare_frame(frame)
        
        # 5. Deteksi (dari src/inference.py)
        results = detector.detect_referee(processed_frame)
        
        # 6. Visualisasi
        annotated_frame = detector.plot_results(processed_frame, results)
        
        # Tampilkan Hasil
        cv2.imshow("Hasil Deteksi", annotated_frame)
        
        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Pemrosesan selesai.")

if __name__ == "__main__":
    main()