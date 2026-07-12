from ultralytics import YOLO
from src.config import MODEL_PATH


class FutsalDetector:
    """
    Kelas utama untuk melakukan inferensi model YOLOv8 Futsal.
    """

    def __init__(self, model_path=MODEL_PATH):
        self.model = YOLO(str(model_path))

    def detect(self, frame, conf=0.35):
        """
        Deteksi objek pada satu frame.
        """
        results = self.model(
            frame,
            conf=conf,
            verbose=False
        )
        return results

    def track(self, frame, conf=0.35):
        """
        Tracking objek menggunakan ByteTrack.
        """
        results = self.model.track(
            frame,
            conf=conf,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )
        return results

    def plot_results(self, results):
        """
        Menggambar bounding box hasil deteksi.
        """
        return results[0].plot()

    def count_objects(self, results):
        """
        Menghitung jumlah objek tiap kelas.
        """

        counts = {}

        if len(results) == 0:
            return counts

        for box in results[0].boxes:

            cls = int(box.cls.item())

            class_name = self.model.names[cls]

            counts[class_name] = counts.get(class_name, 0) + 1

        return counts