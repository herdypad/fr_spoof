"""
Spoof Detection — cvpr2024-face-anti-spoofing-challenge
========================================================
Paper  : Joint Physical-Digital Facial Attack Detection Via Simulating
         Spoofing Clues (CVPR Workshop 2024)
Repo   : https://github.com/Xianhua-He/cvpr2024-face-anti-spoofing-challenge

Alur kerja
----------
1. InsightFace FaceAnalysis  → deteksi wajah di gambar
2. CVPR2024 FAS model        → klasifikasi Real (live=0) vs Spoof (spoof=1)

Model yang tersedia (diunduh otomatis via gdown):
┌─────────────────────────┬────────────────┬──────────────────────────────────┐
│ Model                   │ Input          │ Google Drive ID                  │
├─────────────────────────┼────────────────┼──────────────────────────────────┤
│ mobilenet_v3_small.pth  │ full image     │ 1UhCrC2VQCz4zE1UFc-lnziivqB3bSIpP│
│ shufflenetv2_1.0x.pth   │ full image     │ 18e23EW2ncsnOqET4jCSEskHVNvFYAF1j │
│ shufflenetv2_0.5x.pth   │ full image     │ 16VzZSYcVTNErFoLF87urdrzQbGbKvqjX │
│ full_resnet50.pth        │ full image     │ 1VpWN8CXdVVLTwyTPABeFXmr3UnnenjYe │
│ face_swin_v2_base.pth   │ face area only │ 1E4UD8UK_KzjhpAvR6hYInlteOEaxDZbZ │
└─────────────────────────┴────────────────┴──────────────────────────────────┘

Label: spoof=1, live=0  →  spoof_score = softmax(output)[:, 1]
"""

from __future__ import annotations
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # suppress timm deprecation
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from insightface.app import FaceAnalysis

# ─────────────────────────────────────────────
# Tambahkan cvpr2024_fas ke sys.path agar bisa
# import dari nets/ repo tersebut.
# ─────────────────────────────────────────────
ROOT         = Path(__file__).parent
REPO_DIR     = ROOT / "cvpr2024_fas"
MODELS_DIR   = ROOT / "models"

if REPO_DIR.exists() and str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

try:
    from nets.utils import get_model, load_pretrain   # noqa: E402
except ImportError as e:
    raise ImportError(
        f"Tidak bisa import dari cvpr2024_fas/nets/utils.py: {e}\n"
        f"Pastikan repo sudah ada di: {REPO_DIR}\n"
        "Jalankan: git clone https://github.com/Xianhua-He/cvpr2024-face-anti-spoofing-challenge.git cvpr2024_fas"
    ) from e


# ─────────────────────────────────────────────
# Konfigurasi
# ─────────────────────────────────────────────
TEST_FOLDER = ROOT / "test"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Pilih model di sini ─────────────────────
# "mobilenet_v3_small" → ringan, ~2MB, input full image
# "shufflenetv2_1.0x"  → sedikit lebih akurat
# "face_swin_v2_base"  → paling akurat, tapi perlu crop wajah
ACTIVE_MODEL = "mobilenet_v3_small"

# Mapping nama model → (arch_name, gdrive_id, input_mode)
# input_mode: "full" = gambar penuh, "face" = crop wajah (pakai InsightFace)
MODEL_CONFIG: dict[str, tuple[str, str, str]] = {
    "mobilenet_v3_small": (
        "mobilenet_v3_small",
        "1UhCrC2VQCz4zE1UFc-lnziivqB3bSIpP",
        "full",
    ),
    "shufflenetv2_1.0x": (
        "shufflenetv2_1.0x",
        "18e23EW2ncsnOqET4jCSEskHVNvFYAF1j",
        "full",
    ),
    "shufflenetv2_0.5x": (
        "shufflenetv2_0.5x",
        "16VzZSYcVTNErFoLF87urdrzQbGbKvqjX",
        "full",
    ),
    "full_resnet50": (
        "resnet50",
        "1VpWN8CXdVVLTwyTPABeFXmr3UnnenjYe",
        "full",
    ),
    "face_swin_v2_base": (
        "swin_v2_base",
        "1E4UD8UK_KzjhpAvR6hYInlteOEaxDZbZ",
        "face",
    ),
}

INPUT_SIZE    = 224          # ukuran input model
SPOOF_THRESH  = 0.5          # skor spoof > nilai ini → dikategorikan SPOOF
DEVICE        = "cpu"        # ganti "cuda" jika punya GPU

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

LABEL = {True: "REAL  [OK]", False: "SPOOF [!!]"}



# ─────────────────────────────────────────────
# Download otomatis bobot model via gdown
# ─────────────────────────────────────────────
def _download_weights(model_name: str, gdrive_id: str) -> Path:
    """Download .pth dari Google Drive jika belum ada."""
    MODELS_DIR.mkdir(exist_ok=True)
    weight_path = MODELS_DIR / f"{model_name}.pth"
    if weight_path.exists():
        print(f"[+] Bobot model sudah ada: {weight_path.name}")
        return weight_path

    print(f"[↓] Mengunduh {weight_path.name} dari Google Drive…")
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        gdown.download(url, str(weight_path), quiet=False)
        print(f"[+] Tersimpan di: {weight_path}")
    except Exception as exc:
        raise RuntimeError(
            f"Gagal mengunduh {weight_path.name}: {exc}\n"
            f"Download manual dari: https://drive.google.com/file/d/{gdrive_id}\n"
            f"Simpan ke: {weight_path}"
        ) from exc
    return weight_path


# ─────────────────────────────────────────────
# Load model CVPR2024 FAS
# ─────────────────────────────────────────────
def _load_fas_model(model_name: str = ACTIVE_MODEL) -> tuple[torch.nn.Module, str]:
    """
    Muat model CVPR2024 FAS.

    Returns
    -------
    (model, input_mode)  — input_mode: "full" atau "face"
    """
    if model_name not in MODEL_CONFIG:
        raise ValueError(f"Model '{model_name}' tidak dikenal. Pilih dari: {list(MODEL_CONFIG)}")

    arch_name, gdrive_id, input_mode = MODEL_CONFIG[model_name]
    weight_path = _download_weights(model_name, gdrive_id)

    print(f"[+] Memuat model: {model_name}  (arch={arch_name}, input={input_mode})")

    import logging
    logging.basicConfig(level=logging.WARNING)   # suppress load_pretrain verbose

    model = get_model(arch_name, num_classes=2)
    model = load_pretrain(str(weight_path), model)
    model.eval()
    model.to(DEVICE)
    return model, input_mode


# ─────────────────────────────────────────────
# Load InsightFace (untuk deteksi wajah)
# ─────────────────────────────────────────────
def _load_face_detector() -> FaceAnalysis:
    app = FaceAnalysis(
        name="buffalo_sc",
        allowed_modules=["detection"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


# ─────────────────────────────────────────────
# Preprocessing gambar → tensor
# ─────────────────────────────────────────────
def _preprocess(img_bgr: np.ndarray) -> torch.Tensor:
    """
    BGR → RGB → resize (224×224) → ImageNet normalisasi → tensor NCHW.
    """
    img_rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE))
    img_float   = img_resized.astype(np.float32) / 255.0

    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    img_norm = (img_float - mean) / std

    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0)
    return tensor.to(DEVICE)


# ─────────────────────────────────────────────
# Prediksi anti-spoof satu patch gambar
# ─────────────────────────────────────────────
def _predict(model: torch.nn.Module, arch_name: str, img_bgr: np.ndarray) -> tuple[bool, float]:
    """
    Returns
    -------
    (is_real: bool, spoof_score: float)
    spoof_score ∈ [0,1]  — mendekati 1 → spoof, mendekati 0 → real
    """
    tensor = _preprocess(img_bgr)

    with torch.no_grad():
        # mobilenet & shufflenet → langsung output logits
        # resnet & swin          → (feat, logits) tuple
        out = model(tensor)
        if isinstance(out, (tuple, list)):
            out = out[-1]           # ambil logits

        probs       = F.softmax(out, dim=1)
        spoof_score = float(probs[0, 1].item())   # class 1 = spoof

    is_real = spoof_score < SPOOF_THRESH
    return is_real, spoof_score


# ─────────────────────────────────────────────
# Fungsi utama: test semua foto di folder test/
# ─────────────────────────────────────────────
def test_photos(
    test_folder: str | Path = TEST_FOLDER,
    model_name:  str        = ACTIVE_MODEL,
) -> list[dict]:
    """
    Baca semua gambar di `test_folder`, jalankan deteksi spoofing,
    cetak hasil, dan kembalikan sebagai list.

    Parameters
    ----------
    test_folder : folder berisi gambar-gambar uji
    model_name  : salah satu key dari MODEL_CONFIG

    Returns
    -------
    list[dict]
        [{
            "file"      : str   — nama file,
            "face_count": int   — jumlah wajah terdeteksi,
            "results"   : [{"face_id", "is_real", "spoof_score"}]
        }]

    Catatan label
    -------------
    spoof=1, live=0  (sesuai repo asli)
    spoof_score mendekati 1.0 → SPOOF, mendekati 0.0 → REAL
    """
    test_folder = Path(test_folder)
    if not test_folder.exists():
        raise FileNotFoundError(f"Folder tidak ditemukan: {test_folder}")

    image_files = [
        f for f in sorted(test_folder.iterdir())
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_files:
        print(f"[!] Tidak ada gambar di folder: {test_folder}")
        return []

    print(f"\n{'='*65}")
    print(f"  CVPR2024 Face Anti-Spoofing  |  {len(image_files)} gambar")
    print(f"  Model : {model_name}")
    print(f"{'='*65}")

    # Muat model & detektor
    fas_model, input_mode = _load_fas_model(model_name)
    face_app = _load_face_detector()

    all_results: list[dict] = []

    for img_path in image_files:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"\n[!] Gagal membaca: {img_path.name}")
            continue

        # ── Deteksi wajah ────────────────────────────────────────────
        faces = face_app.get(img_bgr)

        print(f"\n[File] {img_path.name}  →  {len(faces)} wajah terdeteksi")

        file_result: dict = {
            "file":       img_path.name,
            "face_count": len(faces),
            "results":    [],
        }

        if not faces:
            print("       Tidak ada wajah — lewati.")
            all_results.append(file_result)
            continue

        for idx, face in enumerate(faces):
            # Pilih area untuk prediksi:
            # "full"  → gambar asli penuh (model dilatih dengan full image)
            # "face"  → crop wajah (model dilatih dengan face region)
            if input_mode == "face":
                bbox = face.bbox.astype(int)
                x1 = max(0, bbox[0]);  y1 = max(0, bbox[1])
                x2 = min(img_bgr.shape[1], bbox[2])
                y2 = min(img_bgr.shape[0], bbox[3])
                patch = img_bgr[y1:y2, x1:x2]
                if patch.size == 0:
                    continue
            else:
                patch = img_bgr    # full image

            is_real, spoof_score = _predict(fas_model, MODEL_CONFIG[model_name][0], patch)
            real_score = 1.0 - spoof_score

            face_info = {
                "face_id":     idx + 1,
                "is_real":     is_real,
                "spoof_score": round(spoof_score, 4),
            }
            file_result["results"].append(face_info)

            print(
                f"       Wajah #{idx+1}: {LABEL[is_real]}"
                f"  |  spoof={spoof_score:.4f}  real={real_score:.4f}"
            )

            # Full image mode: satu prediksi sudah mewakili seluruh gambar
            if input_mode == "full":
                break

        all_results.append(file_result)

    # ── Ringkasan ─────────────────────────────────────────────────────
    total_faces = sum(r["face_count"] for r in all_results)
    total_real  = sum(1 for r in all_results for f in r["results"] if f["is_real"])
    total_spoof = sum(1 for r in all_results for f in r["results"] if not f["is_real"])

    print(f"\n{'='*65}")
    print(f"  RINGKASAN")
    print(f"  Gambar diproses : {len(image_files)}")
    print(f"  Total wajah     : {total_faces}")
    print(f"  Real            : {total_real}")
    print(f"  Spoof           : {total_spoof}")
    print(f"{'='*65}\n")

    return all_results


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    MODELS_DIR.mkdir(exist_ok=True)
    test_photos()
