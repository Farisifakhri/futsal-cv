import cv2


class CalibrationTool:
    """
    Buka satu frame, klik 4 sudut lapangan secara berurutan:
    kiri-atas -> kanan-atas -> kanan-bawah -> kiri-bawah
    Tekan 'r' untuk reset klik, 'q' atau ENTER untuk selesai (butuh 4 titik).
    """

    def __init__(self, frame):
        self.frame = frame.copy()
        self.display = frame.copy()
        self.points = []

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            self._redraw()

    def _redraw(self):
        self.display = self.frame.copy()
        labels = ["kiri-atas", "kanan-atas", "kanan-bawah", "kiri-bawah"]
        for i, (x, y) in enumerate(self.points):
            cv2.circle(self.display, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(
                self.display, f"{i+1}:{labels[i]}", (x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
        if len(self.points) > 1:
            for i in range(len(self.points) - 1):
                cv2.line(self.display, self.points[i], self.points[i + 1], (255, 0, 0), 2)
        if len(self.points) == 4:
            cv2.line(self.display, self.points[3], self.points[0], (255, 0, 0), 2)

    def run(self):
        window_name = "Kalibrasi Lapangan - klik 4 sudut urut, r=reset, q=selesai"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        while True:
            cv2.imshow(window_name, self.display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):
                self.points = []
                self.display = self.frame.copy()

            if key == ord("q") or key == 13:
                if len(self.points) == 4:
                    break
                else:
                    print(f"Butuh 4 titik, baru ada {len(self.points)}. Klik lagi atau 'r' untuk reset.")

        cv2.destroyWindow(window_name)
        return self.points


def get_calibration_points_from_video(video_path, frame_number=0):
    """
    Ambil 1 frame dari video, buka tool kalibrasi, return 4 titik pixel.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Gagal membaca frame {frame_number} dari {video_path}")

    tool = CalibrationTool(frame)
    points = tool.run()
    print("Titik kalibrasi terpilih:", points)
    return points


def find_calibration_frame(video_path, start_frame=0, step=30):
    """
    Browse video frame-by-frame (dengan step) untuk cari frame yang
    menampilkan area lapangan paling baik (idealnya semua 4 sudut kelihatan,
    atau minimal area yang ingin dikalibrasi terlihat penuh).
    Kontrol: 'n' = next frame, 'p' = previous frame, 'ENTER'/'q' = pilih frame ini
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current = start_frame

    window_name = "Cari frame full-court - n=next, p=prev, ENTER=pilih"
    cv2.namedWindow(window_name)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current)
        ret, frame = cap.read()
        if not ret:
            print(f"Frame {current} gagal dibaca, mentok di batas video.")
            current = max(0, current - step)
            continue

        display = frame.copy()
        cv2.putText(
            display, f"Frame: {current}/{total_frames}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.imshow(window_name, display)

        key = cv2.waitKey(0) & 0xFF

        if key == ord("n"):
            current = min(total_frames - 1, current + step)
        elif key == ord("p"):
            current = max(0, current - step)
        elif key == ord("q") or key == 13:
            break

    cap.release()
    cv2.destroyWindow(window_name)
    print(f"Frame terpilih untuk kalibrasi: {current}")
    return current