from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


class HeatmapCollector:
    """
    Mengumpulkan posisi target class (dalam koordinat area kalibrasi/meter) per track_id,
    lalu render sebagai heatmap 2D di atas diagram area lapangan yang terkalibrasi.
    """

    def __init__(self, target_class_id, conf_threshold=0.5, debug=False, debug_limit=200):
        self.target_class_id = target_class_id
        self.conf_threshold = conf_threshold
        self.positions = defaultdict(list)  # track_id -> list of (court_x, court_y)
        self.all_points = []
        self.debug = debug
        self._debug_printed = 0
        self._debug_limit = debug_limit
        self._total_target_seen = 0  # counter total deteksi target class sepanjang video

    def collect(self, results, transformer):
        if not transformer.is_calibrated():
            return

        boxes = results[0].boxes
        n_total = len(boxes) if boxes is not None else 0

        cls_list = [int(c.item()) for c in boxes.cls] if n_total > 0 else []
        n_target_in_frame = cls_list.count(self.target_class_id)
        self._total_target_seen += n_target_in_frame

        if self.debug and self._debug_printed < self._debug_limit:
            conf_list = [round(c.item(), 2) for c in boxes.conf] if n_total > 0 else []
            has_id = boxes.id is not None if boxes is not None else False
            print(
                f"[debug] total: {n_total}, classes: {cls_list}, confs: {conf_list}, "
                f"has_track_id: {has_id}, TARGET di frame ini: {n_target_in_frame}"
            )
            self._debug_printed += 1

        if boxes is None or n_total == 0:
            return
        if boxes.id is None:
            return

        for box, track_id, cls, conf in zip(
            boxes.xyxy, boxes.id, boxes.cls, boxes.conf
        ):
            if int(cls.item()) != self.target_class_id:
                continue
            if conf.item() < self.conf_threshold:
                continue

            x1, y1, x2, y2 = box.tolist()
            pixel_x = (x1 + x2) / 2
            pixel_y = y2  # bottom-center = posisi kaki

            court_x, court_y = transformer.transform_point(pixel_x, pixel_y)

            # =========================================================================
            # PERBAIKAN UTAMA UNTUK UAS:
            # Mengubah margin dari 2.0 menjadi 50.0 meter agar koordinat pemain 
            # yang meleset akibat distorsi homografi kamera tidak langsung dibuang!
            # =========================================================================
            margin = 50.0 
            in_bounds = (
                -margin <= court_x <= transformer.court_width + margin and
                -margin <= court_y <= transformer.court_length + margin
            )

            if self.debug and self._debug_printed < self._debug_limit:
                status = "DITERIMA" if in_bounds else "DIBUANG"
                print(
                    f"[debug] pixel=({pixel_x:.0f},{pixel_y:.0f}) -> "
                    f"court=({court_x:.1f},{court_y:.1f}) [{status}]"
                )
                self._debug_printed += 1

            if in_bounds:
                tid = int(track_id.item())
                self.positions[tid].append((court_x, court_y))
                self.all_points.append((court_x, court_y))

    def print_summary(self):
        print(f"\n[summary] Total deteksi target class terlihat sepanjang video: "
              f"{self._total_target_seen}")
        print(f"[summary] Total titik masuk heatmap (lolos conf & margin): {len(self.all_points)}")

    def render(self, court_width, court_length, save_path=None, bins=50,
               is_partial=False, label=""):
        if not self.all_points:
            print("Tidak ada titik terkumpul, heatmap kosong.")
            return

        xs = [p[0] for p in self.all_points]
        ys = [p[1] for p in self.all_points]

        fig, ax = plt.subplots(figsize=(10, 5))

        self._draw_court(ax, court_width, court_length, is_partial)

        heatmap, xedges, yedges = np.histogram2d(
            xs, ys, bins=bins,
            range=[[0, court_width], [0, court_length]]
        )

        ax.imshow(
            heatmap.T,
            extent=[0, court_width, 0, court_length],
            origin="lower",
            cmap="hot",
            alpha=0.65,
            aspect="auto",
        )

        title = f"Heatmap Posisi {label}" if label else "Heatmap Posisi"
        if is_partial:
            title += " (area terkalibrasi sebagian — bukan lapangan penuh)"
        ax.set_title(title, fontsize=11)

        ax.set_xlim(0, court_width)
        ax.set_ylim(0, court_length)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Heatmap disimpan di: {save_path}")

        plt.show()

    def _draw_court(self, ax, w, l, is_partial=False):
        ax.plot([0, w, w, 0, 0], [0, 0, l, l, 0], color="white", linewidth=2)

        if not is_partial:
            ax.plot([w / 2, w / 2], [0, l], color="white", linewidth=1)
            circle = plt.Circle((w / 2, l / 2), 3, color="white", fill=False, linewidth=1)
            ax.add_patch(circle)
        else:
            ax.text(
                0.02, 0.95, "AREA PARSIAL", transform=ax.transAxes,
                fontsize=9, color="yellow", weight="bold",
                verticalalignment="top"
            )

        ax.set_facecolor("#1a6b2f")