from ultralytics import YOLO
import cv2

class FutsalDetector:
    def __init__(self, model_path='models/yolov8n.pt'):
        # Load model YOLO (bisa Nano atau hasil training sendiri nanti)
        self.model = YOLO('runs/futsal/yolov8n_futsal/weights/best.pt')

    def detect_referee(self, frame):
        """
        Menjalankan deteksi pada satu frame hanya untuk objek 'person'.
        """
        # Tambahkan argumen classes=[0] agar hanya mendeteksi orang (pemain, kiper, & wasit)
        results = self.model(frame, classes=[0])
        return results

    def plot_results(self, frame, results):
        """
        Menggambar kotak (bounding box) pada frame.
        """
        return results[0].plot()