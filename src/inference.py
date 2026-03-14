from ultralytics import YOLO
import cv2

class FutsalDetector:
    def __init__(self, model_path='models/yolov8n.pt'):
        # Load model YOLO (bisa Nano atau hasil training sendiri nanti)
        self.model = YOLO(model_path)

    def detect_referee(self, frame):
        """
        Menjalankan deteksi pada satu frame.
        """
        results = self.model(frame)
        return results

    def plot_results(self, frame, results):
        """
        Menggambar kotak (bounding box) pada frame.
        """
        return results[0].plot()