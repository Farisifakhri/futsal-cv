import cv2
from ultralytics import YOLO

# Ubah path model
model = YOLO('models/yolov8n.pt')

# Ubah path video
cap = cv2.VideoCapture('data/tes_futsal.mp4')

print("Kamera/Video berhasil dibuka. Tekan 'q' pada keyboard untuk keluar.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Gagal membaca frame atau video sudah habis.")
        break

    # Lakukan deteksi objek pada frame
    results = model(frame)

    # Gambarkan bounding box hasil deteksi ke atas frame
    annotated_frame = results[0].plot()

    # Tampilkan videonya
    cv2.imshow("Test YOLO Futsal", annotated_frame)

    # Berhenti kalau tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan memori setelah selesai
cap.release()
cv2.destroyAllWindows()