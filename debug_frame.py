"""
Diagnostik visual: ambil beberapa frame dari klip3, jalankan detect (bukan track),
gambar box + label class-nya, simpan sebagai gambar biar bisa dicek langsung
apakah class '3' (asumsi referee) itu beneran wasit atau salah label pemain.

Jalankan dari root project: python debug_visual.py
"""
import cv2
from src.inference import FutsalDetector
from src.config import VIDEO_PATH

detector = FutsalDetector()
print("model.names:", detector.model.names)

cap = cv2.VideoCapture(str(VIDEO_PATH))
fps = cap.get(cv2.CAP_PROP_FPS)

# ambil beberapa titik waktu berbeda di 20 detik pertama
sample_seconds = [2, 5, 8, 11, 14, 17]

for sec in sample_seconds:
    frame_num = int(sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        print(f"gagal baca frame di detik {sec}")
        continue

    results = detector.detect(frame, conf=0.20)  # conf rendah biar semua kandidat kelihatan
    annotated = results[0].plot()  # otomatis gambar box + nama class + conf

    out_path = f"outputs/debug_frame_{sec}s.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"disimpan: {out_path}")

cap.release()
print("\nSelesai. Buka semua outputs/debug_frame_*.jpg dan cek manual:")
print("- Yang dilabel 'referee' itu beneran wasit (baju beda, biasanya hitam/kuning stripe)?")
print("- Atau itu sebenarnya pemain yang salah diklasifikasi?")