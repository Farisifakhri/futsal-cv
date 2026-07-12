from ultralytics import YOLO

from .config import MODEL_PATH

class FutsalDetector:

    def __init__(self):

        self.model = YOLO(str(MODEL_PATH))

    def detect(self, frame):

        return self.model.predict(
            frame,
            conf=0.35,
            verbose=False
        )

    def track(self, frame):

        return self.model.track(
            frame,
            conf=0.35,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )