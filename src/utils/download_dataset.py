import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow

def download_futsal_dataset(download_dir="data", model_format="yolov8"):
    """
    Mengunduh dataset futsal dari Roboflow secara aman menggunakan kredensial dari file .env.
    
    Args:
        download_dir (str): Folder tujuan ekstraksi dataset (default: "data")
        model_format (str): Format ekspor dataset (default: "yolov8")
        
    Returns:
        dataset: Objek dataset dari Roboflow
    """
    # 1. Muat environment variables dari .env
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"
    load_dotenv(dotenv_path=env_path)

    api_key = os.getenv("ROBOFLOW_API_KEY")
    workspace_name = os.getenv("ROBOFLOW_WORKSPACE", "risifar")
    project_name = os.getenv("ROBOFLOW_PROJECT", "futsal-detection-ncfs")
    version_num = int(os.getenv("ROBOFLOW_VERSION", "1"))

    if not api_key:
        raise ValueError(
            "[ERROR] ROBOFLOW_API_KEY tidak ditemukan di file .env!\n"
            "Pastikan Anda telah mengisi file .env dengan kredensial yang benar."
        )

    target_path = project_root / download_dir
    os.makedirs(target_path, exist_ok=True)

    print("==================================================")
    print("Start Downloading Dataset from Roboflow")
    print(f"Workspace : {workspace_name}")
    print(f"Project   : {project_name}")
    print(f"Version   : V{version_num}")
    print(f"Path      : {target_path}")
    print("==================================================")

    # 2. Inisialisasi Roboflow client
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace_name).project(project_name)
    version = project.version(version_num)

    # 3. Eksekusi Download Dataset
    dataset = version.download(model_format)

    # 4. Jika Roboflow mendownload ke subfolder (misal futsal-detection-ncfs-1), pindahkan ke download_dir
    downloaded_loc = Path(dataset.location)
    if downloaded_loc.resolve() != target_path.resolve():
        print(f"[INFO] Memindahkan dataset dari {downloaded_loc} ke {target_path}...")
        for item in downloaded_loc.iterdir():
            dest = target_path / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
        shutil.rmtree(downloaded_loc)

    print("==================================================")
    print("✅ Unduh dataset selesai!")
    print(f"📂 Dataset tersimpan di : {target_path}")
    print("==================================================")

    return dataset

if __name__ == "__main__":
    download_futsal_dataset()
