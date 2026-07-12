import cv2
import numpy as np


class HomographyTransformer:
    """
    Transformasi koordinat pixel frame -> koordinat lapangan futsal asli (meter).
    Lapangan futsal standar: 40m x 20m (sesuaikan dengan lapangan di video kamu).
    """

    def __init__(self, court_width=40.0, court_length=20.0):
        self.court_width = court_width
        self.court_length = court_length
        self.H = None

    def calibrate(self, pixel_points):
        """
        pixel_points: list 4 titik (x, y) sudut lapangan di frame.
        Urutan WAJIB konsisten: kiri-atas, kanan-atas, kanan-bawah, kiri-bawah
        """
        if len(pixel_points) != 4:
            raise ValueError("Butuh tepat 4 titik untuk kalibrasi homography.")

        src = np.array(pixel_points, dtype=np.float32)
        dst = np.array([
            [0, 0],
            [self.court_width, 0],
            [self.court_width, self.court_length],
            [0, self.court_length],
        ], dtype=np.float32)

        self.H, status = cv2.findHomography(src, dst)

        if self.H is None:
            raise ValueError("Gagal menghitung homography, cek titik kalibrasi.")

    def transform_point(self, x, y):
        if self.H is None:
            raise ValueError("Homography belum dikalibrasi. Panggil calibrate() dulu.")
        point = np.array([[[x, y]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(point, self.H)
        return float(mapped[0][0][0]), float(mapped[0][0][1])

    def is_calibrated(self):
        return self.H is not None