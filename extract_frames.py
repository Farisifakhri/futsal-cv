import cv2
import os

def extract_frames(video_path, output_dir, frame_interval=30):
    """
    Ekstrak frame dari video untuk kebutuhan labelling.
    
    Args:
        video_path     : Path ke file video
        output_dir     : Folder output untuk menyimpan frame
        frame_interval : Ambil 1 frame setiap N frame (default: 30)
                         Untuk video 30fps → 1 gambar per detik
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Video tidak bisa dibuka: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    duration_sec = total_frames / fps

    print(f"[INFO] Total frame  : {total_frames}")
    print(f"[INFO] FPS video    : {fps:.1f}")
    print(f"[INFO] Durasi       : {duration_sec:.1f} detik")
    print(f"[INFO] Interval     : setiap {frame_interval} frame")
    print(f"[INFO] Estimasi img : ~{total_frames // frame_interval} gambar")
    print("-" * 45)

    frame_count  = 0
    saved_count  = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
            print(f"[SAVED] {filename}")

        frame_count += 1

    cap.release()
    print("-" * 45)
    print(f"[DONE] Total frame tersimpan: {saved_count} gambar")
    print(f"[DONE] Lokasi: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    extract_frames(
        video_path     = "data/tes_futsal.mp4",
        output_dir     = "data/raw_frames",       # Folder hasil ekstrak
        frame_interval = 30                        # Ganti ke 15 jika ingin lebih banyak gambar
    )