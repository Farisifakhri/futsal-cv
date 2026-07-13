import cv2
import os

from src.inference import FutsalDetector
from src.homography import HomographyTransformer
from src.heatmap import HeatmapCollector
from src.calibration_tool import get_calibration_points_from_video, find_calibration_frame


# --- konfigurasi ---
VIDEO_PATH = "data/clips/klip3.mp4"
# 1. UBAH: Nama file output disesuaikan untuk hasil heatmap pemain
OUTPUT_HEATMAP = "outputs/heatmap_players_final.png"

# 2. UBAH: Kembalikan target class ke "players" agar mendeteksi pemain (Indeks 2)
TARGET_CLASS_NAME = "players"

# 3. UBAH: Turunkan sedikit confidence tracking & koleksi 
# agar pemain berukuran kecil (far-court) tidak terbuang oleh YOLOv8
TRACK_CONF = 0.20      # Nilai optimal untuk suplai ByteTrack
COLLECT_CONF = 0.25    # Nilai ambang batas lolos ke titik heatmap

# 4. KONFIGURASI LAPANGAN (Tetap sesuai ukuran bidang parsial Aa)
COURT_WIDTH = 26.0
COURT_LENGTH = 20.0
IS_PARTIAL_COURT = True

SEGMENT_START_SEC = 0.0
SEGMENT_END_SEC = 20.0

SCENE_CHANGE_THRESHOLD = 999.0
SKIP_FRAMES = 2
MAX_CALIBRATIONS = 1

CALIBRATION_SEARCH_STEP = 10

DEBUG_HEATMAP = True
DEBUG_LIMIT = 200

def detect_scene_change(prev_gray, curr_gray, threshold=SCENE_CHANGE_THRESHOLD):
    diff = cv2.absdiff(prev_gray, curr_gray)
    return diff.mean() > threshold


def CalibrationToolFromFrame(frame):
    from src.calibration_tool import CalibrationTool
    tool = CalibrationTool(frame)
    return tool.run()


def run_pipeline():
    os.makedirs("outputs", exist_ok=True)

    if not os.path.exists(VIDEO_PATH):
        print(f"Video tidak ditemukan: {VIDEO_PATH}")
        return

    probe = cv2.VideoCapture(VIDEO_PATH)
    fps = probe.get(cv2.CAP_PROP_FPS)
    total_frames_video = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    probe.release()

    start_frame = int(SEGMENT_START_SEC * fps)
    end_frame = min(int(SEGMENT_END_SEC * fps), total_frames_video)

    print(f"=== Segmen yang diproses: {SEGMENT_START_SEC}s - {SEGMENT_END_SEC}s "
          f"(frame {start_frame} - {end_frame}, fps={fps:.1f}) ===")
    print(f"=== Target class: {TARGET_CLASS_NAME} ===\n")

    print("=== Cari frame yang menampilkan area lapangan dengan baik (dalam segmen ini) ===")
    print(">>> Pencet 'n'/'p' buat browsing, 'q'/Enter kalau area lapangan sudah keliatan luas.\n")

    calibration_frame_num = find_calibration_frame(
        VIDEO_PATH,
        start_frame=start_frame,
        step=CALIBRATION_SEARCH_STEP,
    )

    print("\n=== Kalibrasi ===")
    points = get_calibration_points_from_video(VIDEO_PATH, frame_number=calibration_frame_num)

    transformer = HomographyTransformer(COURT_WIDTH, COURT_LENGTH)
    transformer.calibrate(points)

    detector = FutsalDetector()
    print(f"[debug] model.names lengkap: {detector.model.names}")

    target_class_id = [
        i for i, n in detector.model.names.items() if n == TARGET_CLASS_NAME
    ]
    print(f"[debug] target_class_id yang dipakai ({TARGET_CLASS_NAME}): {target_class_id}")

    if not target_class_id:
        raise ValueError(
            f"Class '{TARGET_CLASS_NAME}' tidak ditemukan di model.names: {detector.model.names}"
        )

    collector = HeatmapCollector(
        target_class_id[0], COLLECT_CONF, debug=DEBUG_HEATMAP, debug_limit=DEBUG_LIMIT
    )

    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    prev_gray = None
    frame_idx = start_frame
    processed_idx = 0
    calibrations_done = 0
    scene_warning_count = 0

    print(f"\n=== Memulai tracking + pengumpulan posisi (frame {start_frame} - {end_frame}) ===")
    while cap.isOpened() and frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % (SKIP_FRAMES + 1) != 0:
            frame_idx += 1
            continue

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None and detect_scene_change(prev_gray, curr_gray):
            if calibrations_done < MAX_CALIBRATIONS:
                print(f"\nPerubahan scene terdeteksi di frame {frame_idx}.")
                new_points = CalibrationToolFromFrame(frame)
                transformer.calibrate(new_points)
                calibrations_done += 1
            else:
                scene_warning_count += 1
                if scene_warning_count % 50 == 1:
                    print(f"[warning] Scene change di frame {frame_idx}, limit kalibrasi tercapai.")

        prev_gray = curr_gray

        results = detector.track(frame, conf=TRACK_CONF)
        collector.collect(results, transformer)

        frame_idx += 1
        processed_idx += 1

        if processed_idx % 50 == 0:
            print(f"Frame diproses: {processed_idx} (source frame: {frame_idx})")

    cap.release()

    print(f"\nTotal processed frame: {processed_idx}")
    collector.print_summary()
    print(f"Jumlah track ID unik: {len(collector.positions)}")

    if len(collector.all_points) == 0:
        print(
            "\n[diagnosa] Tidak ada titik terkumpul.\n"
            "  Cek baris [summary] di atas:\n"
            "  - Kalau 'Total deteksi target class' = 0 -> window video ini nyaris tidak\n"
            "    menampilkan class tersebut sama sekali, ganti SEGMENT_START_SEC/END_SEC.\n"
            "  - Kalau > 0 tapi 'Total titik masuk heatmap' = 0 -> masalah di\n"
            "    COLLECT_CONF terlalu tinggi atau kalibrasi/margin bermasalah."
        )

    collector.render(
        COURT_WIDTH, COURT_LENGTH,
        save_path=OUTPUT_HEATMAP,
        is_partial=IS_PARTIAL_COURT,
        label=TARGET_CLASS_NAME,
    )


if __name__ == "__main__":
    run_pipeline()