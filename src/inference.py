from ultralytics import YOLO
import cv2

class FutsalDetector:
    def __init__(self, model_path=r'C:\futsal-cv\runs\detect\runs\futsal\yolov8n_futsal7\weights\best.pt'):
        """
        Load model YOLO menggunakan custom weights hasil training ke-7 Aa yang sukses.
        """
        self.model = YOLO(model_path)

    def detect_referee(self, frame):
        """
        Menjalankan deteksi pada satu frame untuk seluruh kelas custom futsal 
        (keeper, official, outfield-players, players, referee).
        """
        # KUNCI UTAMA: Hapus parameter classes agar semua objek custom Aa langsung terdeteksi
        results = self.model(frame)
        return results

    def plot_results(self, frame, results):
        """
        Menggambar kotak (bounding box) beserta nama kelas custom langsung pada frame.
        """
        return results[0].plot()