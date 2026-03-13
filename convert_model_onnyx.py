"""
Konversi model CVPR2024 Face Anti-Spoofing → ONNX (→ TFLite opsional)
======================================================================
Output model:
  - Input  : [batch, 3, 224, 224]  float32, ImageNet-normalized
  - Output : [batch, 2]            logits  [live_score, spoof_score]

Cara pakai:
  python convert_model.py                          # konversi mobilenet_v3_small (default)
  python convert_model.py --model shufflenetv2_1.0x

File hasil:
  models/antispoof.onnx      ← untuk Flutter via onnxruntime atau tflite_flutter
  models/antispoof.tflite    ← (jika flag --tflite diberikan)
"""

import argparse
import sys
import warnings
import os
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).parent
REPO_DIR = ROOT / "cvpr2024_fas"

if REPO_DIR.exists() and str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

# ─────────────────────────────────────────────
# Konfigurasi model yang tersedia
# ─────────────────────────────────────────────
MODEL_CONFIG = {
    "mobilenet_v3_small": ("mobilenet_v3_small", "1UhCrC2VQCz4zE1UFc-lnziivqB3bSIpP"),
    "shufflenetv2_1.0x":  ("shufflenetv2_1.0x",  "18e23EW2ncsnOqET4jCSEskHVNvFYAF1j"),
    "shufflenetv2_0.5x":  ("shufflenetv2_0.5x",  "16VzZSYcVTNErFoLF87urdrzQbGbKvqjX"),
    "full_resnet50":      ("resnet50",             "1VpWN8CXdVVLTwyTPABeFXmr3UnnenjYe"),
    "face_swin_v2_base":  ("swin_v2_base",         "1E4UD8UK_KzjhpAvR6hYInlteOEaxDZbZ"),
}

INPUT_SIZE = 224


# ─────────────────────────────────────────────
# Load model dari checkpoint
# ─────────────────────────────────────────────
def load_model(model_name: str):
    import torch
    import logging
    logging.basicConfig(level=logging.WARNING)

    from nets.utils import get_model, load_pretrain  # dari cvpr2024_fas/

    arch_name, gdrive_id = MODEL_CONFIG[model_name]
    weight_path = ROOT / "models" / f"{model_name}.pth"

    # Download jika belum ada
    if not weight_path.exists():
        print(f"[↓] Mengunduh {model_name}.pth ...")
        import gdown
        gdown.download(
            f"https://drive.google.com/uc?id={gdrive_id}",
            str(weight_path), quiet=False,
        )

    print(f"[+] Memuat model: {model_name} ({arch_name})")
    model = get_model(arch_name, num_classes=2)
    model = load_pretrain(str(weight_path), model)
    model.eval()
    return model


# ─────────────────────────────────────────────
# Konversi → ONNX
# ─────────────────────────────────────────────
def convert_to_onnx(model, output_path: Path) -> Path:
    import torch

    print(f"\n[→] Konversi ke ONNX ...")
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=18,      # opset 18 kompatibel dengan torch >= 2.0
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch"},
            "output": {0: "batch"},
        },
        do_constant_folding=True,
    )

    # torch 2.x kadang membuat file .data terpisah — gabungkan ke satu file
    data_file = Path(str(output_path) + ".data")
    if data_file.exists():
        import onnx
        from onnx.external_data_helper import convert_model_to_external_data, load_external_data_for_model
        onnx_model_tmp = onnx.load(str(output_path))
        load_external_data_for_model(onnx_model_tmp, str(output_path.parent))
        # Simpan ulang sebagai satu file (tanpa external data)
        onnx.save_model(
            onnx_model_tmp,
            str(output_path),
            save_as_external_data=False,
        )
        data_file.unlink(missing_ok=True)
        print("    → merged external data to single .onnx file")

    # Verifikasi
    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"[✓] ONNX tersimpan : {output_path}  ({size_mb:.2f} MB)")

    # Quick runtime test
    import onnxruntime as ort
    import numpy as np
    sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    dummy_np = np.random.randn(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
    out = sess.run(None, {"input": dummy_np})
    print(f"    → test inference OK  |  output shape: {out[0].shape}")

    return output_path


# ─────────────────────────────────────────────
# Konversi → TFLite (opsional, butuh tensorflow)
# ─────────────────────────────────────────────
def convert_to_tflite(onnx_path: Path, output_path: Path) -> Path | None:
    print(f"\n[→] Konversi ke TFLite ...")

    try:
        import onnx
        import onnx2tf
    except ImportError:
        print("[!] Butuh: pip install onnx2tf tensorflow")
        print("    Lewati konversi TFLite, gunakan ONNX di Flutter.")
        return None

    # onnx → tf savedmodel → tflite
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(output_path.parent / "tf_savedmodel"),
        non_verbose=True,
    )

    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_saved_model(
        str(output_path.parent / "tf_savedmodel")
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    output_path.write_bytes(tflite_model)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"[✓] TFLite tersimpan: {output_path}  ({size_mb:.2f} MB)")
    return output_path


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Konversi model CVPR2024 FAS ke ONNX / TFLite"
    )
    parser.add_argument(
        "--model", default="mobilenet_v3_small",
        choices=list(MODEL_CONFIG.keys()),
        help="Nama model yang akan dikonversi (default: mobilenet_v3_small)",
    )
    parser.add_argument(
        "--tflite", action="store_true",
        help="Konversi juga ke TFLite setelah ONNX (butuh tensorflow + onnx2tf)",
    )
    parser.add_argument(
        "--output-dir", default="models",
        help="Folder output (default: models/)",
    )
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(exist_ok=True)

    onnx_path   = out_dir / "antispoof.onnx"
    tflite_path = out_dir / "antispoof.tflite"

    print("=" * 60)
    print(f"  CVPR2024 FAS Model Converter")
    print(f"  Model  : {args.model}")
    print(f"  Output : {out_dir}")
    print("=" * 60)

    # 1. Load
    model = load_model(args.model)

    # 2. → ONNX
    convert_to_onnx(model, onnx_path)

    # 3. → TFLite (opsional)
    if args.tflite:
        convert_to_tflite(onnx_path, tflite_path)

    print("\n" + "=" * 60)
    print("  Selesai! Copy file berikut ke Flutter:")
    print(f"  {onnx_path}")
    if args.tflite and tflite_path.exists():
        print(f"  {tflite_path}")
    print("  → flutter_antispoof/assets/models/")
    print("=" * 60)


if __name__ == "__main__":
    main()
