import cv2
import os
import time

from src.inference import FutsalDetector
from src.preprocess import prepare_frame


VIDEO_PATH = "data/clips/klip1.mp4"
OUTPUT_PATH = "outputs/hasil_deteksi.mp4"


def main():

    if not os.path.exists(VIDEO_PATH):
        print(f"Video tidak ditemukan : {VIDEO_PATH}")
        return

    detector = FutsalDetector()

    cap = cv2.VideoCapture(VIDEO_PATH)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)

    os.makedirs("outputs", exist_ok=True)

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_video,
        (width, height)
    )

    prev_time = time.time()

    print("=== Memulai Deteksi ===")
    print("Tekan tombol Q untuk keluar.")

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        processed = prepare_frame(frame)

        # Kalau mau tracking tinggal ganti ini
        # results = detector.track(processed)
        results = detector.detect(processed)

        annotated = detector.plot_results(results)

        counts = detector.count_objects(results)

        current = time.time()
        fps = 1 / (current - prev_time)
        prev_time = current

        y = 30

        cv2.putText(
            annotated,
            f"FPS : {fps:.1f}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        y += 30

        for name, total in counts.items():

            cv2.putText(
                annotated,
                f"{name} : {total}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            y += 30

        writer.write(annotated)

        cv2.imshow("FutsalLens AI", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print()
    print("===================================")
    print("Deteksi selesai.")
    print(f"Hasil tersimpan di : {OUTPUT_PATH}")
    print("===================================")


if __name__ == "__main__":
    main()