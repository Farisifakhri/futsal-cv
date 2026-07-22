import os
import shutil
from pathlib import Path
import torch
import numpy as np

# --- Fix untuk numpy 2.0+ compatibility dengan ultralytics ---
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

# --- Fix untuk PyTorch 2.6+ weights_only default (Aman dari RecursionError di Notebook) ---
if not getattr(torch, '_is_patched_for_weights_only', False):
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
    torch._is_patched_for_weights_only = True

from ultralytics import YOLO
from src.preprocess import preprocess_dataset
from src.augment import augment_dataset

def create_augmented_yaml(output_yaml_path="data_augmented/data.yaml"):
    """
    Membuat file konfigurasi data.yaml khusus untuk dataset yang sudah di-preprocess dan di-augmentasi.
    """
    abs_aug_dir = os.path.abspath(os.path.dirname(output_yaml_path))
    yaml_content = f"""# Futsal-CV Dataset V3 (Preprocessed & Augmented 4-Classes)
path: {abs_aug_dir}
train: train/images
val: valid/images
test: test/images

nc: 4
names:
  0: ball
  1: goalkeeper
  2: player
  3: referee
"""
    os.makedirs(os.path.dirname(output_yaml_path), exist_ok=True)
    with open(output_yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"Config created at: {os.path.abspath(output_yaml_path)}")

def plot_training_results(save_dir):
    """
    Membuat visualisasi grafik performa training model.
    Grafik mencakup: Loss curves, mAP curves, dan per-class performance.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import csv
    except ImportError:
        print("[WARN] matplotlib tidak tersedia, skip plotting.")
        return
    
    results_csv = os.path.join(save_dir, 'results.csv')
    if not os.path.exists(results_csv):
        print(f"[WARN] File {results_csv} tidak ditemukan, skip plotting.")
        return
    
    # Parse results.csv
    data = {}
    with open(results_csv, 'r') as f:
        reader = csv.reader(f)
        headers = [h.strip() for h in next(reader)]
        for h in headers:
            data[h] = []
        for row in reader:
            for i, val in enumerate(row):
                try:
                    data[headers[i]].append(float(val.strip()))
                except (ValueError, IndexError):
                    data[headers[i]].append(0.0)
    
    epochs = list(range(1, len(data.get('epoch', [])) + 1))
    if not epochs:
        print("[WARN] Tidak ada data epoch di results.csv, skip plotting.")
        return
    
    # --- Konfigurasi style ---
    plt.style.use('default')
    colors = {
        'train_box': '#2196F3',
        'train_cls': '#FF5722', 
        'train_dfl': '#4CAF50',
        'val_box': '#1565C0',
        'val_cls': '#D84315',
        'val_dfl': '#2E7D32',
        'precision': '#9C27B0',
        'recall': '#FF9800',
        'mAP50': '#E91E63',
        'mAP50-95': '#00BCD4',
        'lr': '#607D8B',
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('📊 Futsal-CV YOLOv8m Training Performance (Dataset V3)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # --- Plot 1: Box Loss ---
    ax = axes[0, 0]
    if 'train/box_loss' in data:
        ax.plot(epochs, data['train/box_loss'], color=colors['train_box'], label='Train', linewidth=2)
    if 'val/box_loss' in data:
        ax.plot(epochs, data['val/box_loss'], color=colors['val_box'], label='Val', linewidth=2, linestyle='--')
    ax.set_title('Box Loss', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # --- Plot 2: Classification Loss ---
    ax = axes[0, 1]
    if 'train/cls_loss' in data:
        ax.plot(epochs, data['train/cls_loss'], color=colors['train_cls'], label='Train', linewidth=2)
    if 'val/cls_loss' in data:
        ax.plot(epochs, data['val/cls_loss'], color=colors['val_cls'], label='Val', linewidth=2, linestyle='--')
    ax.set_title('Classification Loss', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # --- Plot 3: DFL Loss ---
    ax = axes[0, 2]
    if 'train/dfl_loss' in data:
        ax.plot(epochs, data['train/dfl_loss'], color=colors['train_dfl'], label='Train', linewidth=2)
    if 'val/dfl_loss' in data:
        ax.plot(epochs, data['val/dfl_loss'], color=colors['val_dfl'], label='Val', linewidth=2, linestyle='--')
    ax.set_title('Distribution Focal Loss (DFL)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # --- Plot 4: Precision & Recall ---
    ax = axes[1, 0]
    if 'metrics/precision(B)' in data:
        ax.plot(epochs, data['metrics/precision(B)'], color=colors['precision'], label='Precision', linewidth=2)
    if 'metrics/recall(B)' in data:
        ax.plot(epochs, data['metrics/recall(B)'], color=colors['recall'], label='Recall', linewidth=2)
    ax.set_title('Precision & Recall', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # --- Plot 5: mAP50 & mAP50-95 ---
    ax = axes[1, 1]
    if 'metrics/mAP50(B)' in data:
        ax.plot(epochs, data['metrics/mAP50(B)'], color=colors['mAP50'], label='mAP50', linewidth=2.5)
    if 'metrics/mAP50-95(B)' in data:
        ax.plot(epochs, data['metrics/mAP50-95(B)'], color=colors['mAP50-95'], label='mAP50-95', linewidth=2.5)
    ax.set_title('mAP Performance', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Tambahkan annotasi best mAP50 value
    if 'metrics/mAP50(B)' in data and data['metrics/mAP50(B)']:
        best_map50 = max(data['metrics/mAP50(B)'])
        best_epoch = data['metrics/mAP50(B)'].index(best_map50) + 1
        ax.annotate(f'Best: {best_map50:.3f}\n(Epoch {best_epoch})', 
                    xy=(best_epoch, best_map50),
                    xytext=(best_epoch + len(epochs)*0.1, best_map50 - 0.08),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    fontsize=10, fontweight='bold', color='red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # --- Plot 6: Learning Rate ---
    ax = axes[1, 2]
    if 'lr/pg0' in data:
        ax.plot(epochs, data['lr/pg0'], color=colors['lr'], label='LR (pg0)', linewidth=2)
    if 'lr/pg1' in data:
        ax.plot(epochs, data['lr/pg1'], color='#90A4AE', label='LR (pg1)', linewidth=1.5, linestyle='--')
    ax.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plot_path = os.path.join(save_dir, 'training_performance.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"📊 Grafik performa training disimpan di: {plot_path}")
    
    return plot_path

def print_final_summary(save_dir):
    """
    Mencetak ringkasan akhir metrik performa model setelah training selesai.
    """
    results_csv = os.path.join(save_dir, 'results.csv')
    if not os.path.exists(results_csv):
        return
    
    try:
        import csv
        with open(results_csv, 'r') as f:
            reader = csv.reader(f)
            headers = [h.strip() for h in next(reader)]
            last_row = None
            for row in reader:
                last_row = row
        
        if last_row:
            print("\n" + "=" * 60)
            print("📈 RINGKASAN PERFORMA MODEL (Epoch Terakhir)")
            print("=" * 60)
            metrics_map = {
                'metrics/precision(B)': 'Precision',
                'metrics/recall(B)': 'Recall',
                'metrics/mAP50(B)': 'mAP@50',
                'metrics/mAP50-95(B)': 'mAP@50-95',
            }
            for key, name in metrics_map.items():
                if key in headers:
                    idx = headers.index(key)
                    val = float(last_row[idx].strip())
                    emoji = "✅" if val >= 0.8 else "⚠️" if val >= 0.6 else "❌"
                    print(f"  {emoji} {name:15s}: {val:.4f}")
            print("=" * 60)
    except Exception as e:
        print(f"[WARN] Gagal membaca ringkasan: {e}")

def run_pipeline(model_type="yolov8m.pt", imgsz=1280, epochs=80):
    """
    Pipeline training YOLOv8m optimal untuk dataset futsal v3.
    
    Strategi optimasi:
    - imgsz=1280: Pertahankan detail objek kecil (bola)
    - copy_paste=0.3: Auto-balancing kelas minoritas
    - cls=1.5: Naikkan bobot classification loss
    - multi_scale=True: Variasi ukuran input per batch
    - close_mosaic=15: Mosaic lebih lama untuk small object
    - cos_lr=True: Smooth convergence
    """
    # 1. Preprocessing Data (CLAHE saja, TANPA resize)
    processed_dir = preprocess_dataset(src_root="data", dst_root="data_processed")

    # 2. Augmentasi Data (Class-Aware Oversampling)
    augmented_dir = augment_dataset(src_root="data_processed", dst_root="data_augmented", augment_train_only=True)

    # 3. Buat data.yaml untuk dataset ter-augmentasi
    yaml_path = os.path.join(augmented_dir, "data.yaml")
    create_augmented_yaml(yaml_path)

    # 4. Modeling menggunakan YOLOv8m
    model_name_clean = Path(model_type).stem
    model_path = os.path.join('models', model_type if model_type.endswith('.pt') else f"{model_type}.pt")
    
    # Cek apakah pretrained weights lokal sudah ada di folder models/
    if os.path.exists(model_path):
        model_to_load = model_path
    else:
        model_to_load = model_type

    os.makedirs('models', exist_ok=True)
    stable_path = os.path.join('models', 'best_futsal.pt')

    print("==================================================")
    print(f"🚀 [Step 3/3] Memulai Training Model {model_name_clean.upper()}")
    print(f"📌 imgsz={imgsz} | epochs={epochs} | batch=auto-calculated")
    print("==================================================")

    model = YOLO(model_to_load)

    # ============================================================
    # Hyperparameter optimal untuk Dataset V3 (class imbalance fix)
    # ============================================================
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=4,                # imgsz=1280 pada T4 (15GB) perlu batch kecil
        device=0 if torch.cuda.is_available() else 'cpu',
        project='futsal',
        name=f'{model_name_clean}_futsal_v3',
        exist_ok=True,
        
        # --- Convergence & Regularization ---
        patience=20,            # Lebih sabar menunggu improvement kelas minoritas
        optimizer='AdamW',      # Eksplisit agar lr0 tidak di-override
        lr0=0.001,              # Learning rate awal yang stabil
        lrf=0.01,               # Final LR ratio
        cos_lr=True,            # Cosine LR schedule → smooth convergence
        weight_decay=0.001,     # Regularisasi sedikit lebih kuat
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        label_smoothing=0.05,   # Sedikit label smoothing
        
        # --- Augmentasi Online (KUNCI anti-imbalance) ---
        mosaic=1.0,             # Mosaic penuh — penting untuk small object
        close_mosaic=15,        # Matikan mosaic di 15 epoch terakhir (bukan 10)
        mixup=0.2,              # Mixup untuk regularisasi occlusion
        copy_paste=0.3,         # ⚡ KUNCI: Copy-paste augmentation → auto-balance kelas minoritas
        
        # --- Augmentasi Geometris ---
        degrees=10.0,           # Rotasi acak
        translate=0.2,          # Translasi lebih besar
        scale=0.9,              # Multi-scale lebih agresif
        fliplr=0.5,             # Flip horizontal
        flipud=0.0,             # Tidak flip vertikal (futsal tidak masuk akal)
        perspective=0.0001,     # Sedikit perspektif transform
        
        # --- Augmentasi Warna ---
        hsv_h=0.015,            # Hue jitter
        hsv_s=0.7,              # Saturation jitter
        hsv_v=0.4,              # Value jitter
        erasing=0.4,            # Random erasing untuk robustness
        
        # --- Loss Weights ---
        box=7.5,                # Box loss weight (default)
        cls=1.5,                # ⚡ KUNCI: Naikkan cls loss → fokus membedakan kelas
        dfl=1.5,                # Distribution Focal Loss (default)
        
        # --- Multi-Scale Training ---
        # NOTE: multi_scale=True dinonaktifkan karena bug di ultralytics 8.1.0
        # (random.randrange() menerima float dari imgsz*0.5 → TypeError)
        # Variasi skala sudah ter-cover oleh parameter scale=0.9 di augmentasi geometris
        multi_scale=False,
    )

    print("==================================================")
    print("✅ Training Pipeline Finished!")
    print("==================================================")

    # 5. Cari dan simpan model terbaik
    save_dir = os.path.join('futsal', f'{model_name_clean}_futsal_v3')
    possible_best_paths = [
        os.path.join(save_dir, 'weights', 'best.pt'),
        os.path.join('runs', 'futsal', f'{model_name_clean}_futsal_v3', 'weights', 'best.pt'),
    ]

    trained_best = None
    for p in possible_best_paths:
        if os.path.exists(p):
            trained_best = p
            break

    if trained_best and os.path.exists(trained_best):
        shutil.copy(trained_best, stable_path)
        print(f"🏆 Model terbaik disalin ke: {os.path.abspath(stable_path)}")
    else:
        print(f"[WARN] File model terbaik tidak ditemukan di: {possible_best_paths}")

    # 6. Plot grafik performa training
    actual_save_dir = save_dir if os.path.exists(save_dir) else None
    if not actual_save_dir:
        alt_dir = os.path.join('runs', 'futsal', f'{model_name_clean}_futsal_v3')
        if os.path.exists(alt_dir):
            actual_save_dir = alt_dir
    
    if actual_save_dir:
        plot_training_results(actual_save_dir)
        print_final_summary(actual_save_dir)

if __name__ == "__main__":
    run_pipeline()
