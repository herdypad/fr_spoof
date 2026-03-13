"""
spoof_image.py — Production-ready Face Anti-Spoof Detector
===========================================================
Stack  : ONNX Runtime (CPU) + InsightFace face detection
Model  : models/antispoof.onnx  (MobileNetV3-Small, input 224×224)
Label  : spoof_score → 1 = SPOOF,  0 = REAL

Fitur untuk production
----------------------
• Singleton — model & detektor dimuat sekali, dipakai berkali-kali
• ONNX Runtime + ORT_ENABLE_ALL → inferensi jauh lebih cepat dari PyTorch
• Thread-safe  — inference lock agar aman di multi-thread/asyncio env
• Warm-up      — 3 dummy forward pass saat init agar kernel JIT sudah siap
• Batch inference — semua wajah dalam satu gambar diproses satu kali jalan
• Face crop + margin — crop wajah (bukan full image) untuk akurasi lebih baik
• Latency tracking per gambar

Cara pakai
----------
# API sederhana:
    from spoof_image import SpoofDetector
    det = SpoofDetector()           # load model once
    result = det.predict_file("test/foto.jpg")
    for face in result.faces:
        print(face.is_real, face.spoof_score)

# CLI:
    python spoof_image.py                         # test/ folder (default)
    python spoof_image.py --image foto.jpg
    python spoof_image.py --folder test/
    python spoof_image.py --folder test/ --benchmark
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

# ──────────────────────────────────────────────────────────────
# Konstanta global
# ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

ROOT            = Path(__file__).parent
MODELS_DIR      = ROOT / "models"
ONNX_MODEL_PATH = MODELS_DIR / "antispoof.onnx"

INPUT_SIZE     = 224
SPOOF_THRESH   = 0.5
FACE_DET_SIZE  = (320, 320)   # lebih kecil = lebih cepat, cukup untuk face det

# Pre-compute sebagai array agar preprocessing tidak re-alloc tiap frame
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ──────────────────────────────────────────────────────────────
# Data classes hasil prediksi
# ──────────────────────────────────────────────────────────────
@dataclass
class FaceResult:
    """Hasil anti-spoof satu wajah."""
    face_id:     int
    is_real:     bool
    spoof_score: float         # mendekati 1 → SPOOF
    real_score:  float         # mendekati 1 → REAL
    bbox:        list[int]     # [x1, y1, x2, y2] dalam piksel gambar asli


@dataclass
class ImageResult:
    """Hasil anti-spoof satu gambar (bisa banyak wajah)."""
    file:       str
    face_count: int
    faces:      list[FaceResult] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def is_real(self) -> Optional[bool]:
        """True jika SEMUA wajah real, False jika ada yang spoof, None jika kosong."""
        if not self.faces:
            return None
        return all(f.is_real for f in self.faces)


# ──────────────────────────────────────────────────────────────
# SpoofDetector — kelas utama
# ──────────────────────────────────────────────────────────────
class SpoofDetector:
    """
    Production-ready Face Anti-Spoof Detector.

    Singleton — hanya satu instance per proses, model tidak dibebani
    load ulang.  Thread-safe via `_infer_lock`.

    Parameters
    ----------
    onnx_path    : path ke file .onnx (default: models/antispoof.onnx)
    spoof_thresh : threshold skor spoof (default 0.5)
    num_threads  : ONNX intra-op threads (default 4, turun jika single-core)
    face_margin  : padding wajah relatif terhadap lebar/tinggi bbox (default 0.2)
    enable_warmup: jalankan 3 dummy pass saat init (default True)
    """

    _instance: Optional["SpoofDetector"] = None
    _class_lock = threading.Lock()

    # ── Singleton ──────────────────────────────────────────────
    def __new__(cls, *args, **kwargs) -> "SpoofDetector":
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    obj._initialized = False
                    cls._instance = obj
        return cls._instance

    def __init__(
        self,
        onnx_path:     str | Path  = ONNX_MODEL_PATH,
        spoof_thresh:  float       = SPOOF_THRESH,
        num_threads:   int         = 4,
        face_margin:   float       = 0.2,
        enable_warmup: bool        = True,
    ) -> None:
        # Guard agar __init__ tidak dijalankan ulang pada singleton yang sama
        if self._initialized:
            return

        self.spoof_thresh = spoof_thresh
        self.face_margin  = face_margin
        self._onnx_path   = Path(onnx_path)
        self._infer_lock  = threading.Lock()

        logger.info("Memuat ONNX session…")
        self._session    = self._build_ort_session(num_threads)
        self._input_name = self._session.get_inputs()[0].name

        logger.info("Memuat InsightFace face detector…")
        self._face_app = self._build_face_detector()

        if enable_warmup:
            self._warmup()

        self._initialized = True
        logger.info(
            "SpoofDetector siap — model: %s | thresh: %.2f",
            self._onnx_path.name, self.spoof_thresh,
        )

    # ── Internal: build ORT session ────────────────────────────
    def _build_ort_session(self, num_threads: int) -> ort.InferenceSession:
        if not self._onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model tidak ditemukan: {self._onnx_path}\n"
                "Jalankan dulu: python convert_model_onnyx.py"
            )

        opts = ort.SessionOptions()
        # Aktifkan semua optimisasi graf (constant folding, op fusion, dll.)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = max(1, num_threads // 2)
        # SEQUENTIAL lebih efisien untuk model kecil (MobileNet/ShuffleNet)
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # Matikan spinner log ORT agar output bersih
        opts.log_severity_level = 3

        session = ort.InferenceSession(
            str(self._onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        logger.info("ORT session OK — providers: %s", session.get_providers())
        return session

    # ── Internal: build InsightFace detector ───────────────────
    def _build_face_detector(self) -> FaceAnalysis:
        app = FaceAnalysis(
            name="buffalo_sc",
            allowed_modules=["detection"],   # hanya deteksi, tidak butuh landmark/embedding
            providers=["CPUExecutionProvider"],
        )
        # det_size kecil = lebih cepat, masih cukup untuk sebagian besar use-case
        app.prepare(ctx_id=0, det_size=FACE_DET_SIZE)
        return app

    # ── Internal: warm-up ──────────────────────────────────────
    def _warmup(self) -> None:
        dummy = np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        for _ in range(3):
            self._session.run(None, {self._input_name: dummy})
        logger.info("Warm-up selesai (3 pass).")

    # ── Internal: preprocessing BGR patch → NCHW float32 ──────
    @staticmethod
    def _preprocess_single(img_bgr: np.ndarray) -> np.ndarray:
        """
        BGR → RGB → resize 224×224 → normalisasi ImageNet → NCHW float32.
        Semua operasi numpy in-place untuk meminimalkan alokasi memori.
        """
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)
        img *= 1.0 / 255.0
        img -= _MEAN
        img /= _STD
        # HWC → CHW → 1×CHW
        return np.ascontiguousarray(img.transpose(2, 0, 1))[np.newaxis]

    # ── Internal: softmax ──────────────────────────────────────
    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def predict_patch(self, img_bgr: np.ndarray) -> tuple[bool, float]:
        """
        Inferensi anti-spoof langsung pada satu patch gambar (tanpa deteksi wajah).

        Parameters
        ----------
        img_bgr : gambar BGR (dari cv2.imread atau frame kamera)

        Returns
        -------
        (is_real, spoof_score)
        """
        tensor = self._preprocess_single(img_bgr)
        with self._infer_lock:
            logits = self._session.run(None, {self._input_name: tensor})[0]
        probs = self._softmax(logits)
        spoof_score = float(probs[0, 1])
        return spoof_score < self.spoof_thresh, spoof_score

    def predict_batch(self, patches: list[np.ndarray]) -> list[tuple[bool, float]]:
        """
        Batch inferensi — proses banyak patch sekaligus dalam satu ORT call.
        Lebih cepat dari memanggil predict_patch N kali.

        Parameters
        ----------
        patches : list gambar BGR (ukuran boleh berbeda, akan di-resize)

        Returns
        -------
        list[(is_real, spoof_score)]
        """
        if not patches:
            return []

        # Stack semua patch jadi satu tensor batch
        tensors = [self._preprocess_single(p) for p in patches]
        batch   = np.concatenate(tensors, axis=0)   # shape: [N, 3, 224, 224]

        with self._infer_lock:
            logits = self._session.run(None, {self._input_name: batch})[0]

        probs   = self._softmax(logits)
        results = []
        for i in range(len(patches)):
            sc = float(probs[i, 1])
            results.append((sc < self.spoof_thresh, sc))
        return results

    def predict_image(
        self,
        img_bgr:  np.ndarray,
        filename: str = "unknown",
    ) -> ImageResult:
        """
        Pipeline lengkap: deteksi wajah → crop → batch anti-spoof.

        Parameters
        ----------
        img_bgr  : gambar BGR (cv2.imread)
        filename : label untuk logging / result

        Returns
        -------
        ImageResult  dengan detail per wajah (FaceResult)
        """
        t0 = time.perf_counter()

        # 1. Deteksi wajah
        faces = self._face_app.get(img_bgr)

        patches: list[np.ndarray] = []
        bboxes:  list[list[int]]  = []
        h, w = img_bgr.shape[:2]

        for face in faces:
            bbox = face.bbox.astype(int)
            bw   = bbox[2] - bbox[0]
            bh   = bbox[3] - bbox[1]
            mx   = int(bw * self.face_margin)
            my   = int(bh * self.face_margin)

            x1 = max(0, bbox[0] - mx)
            y1 = max(0, bbox[1] - my)
            x2 = min(w, bbox[2] + mx)
            y2 = min(h, bbox[3] + my)

            crop = img_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            patches.append(crop)
            bboxes.append([x1, y1, x2, y2])

        # 2. Batch inferensi anti-spoof
        face_results: list[FaceResult] = []
        if patches:
            preds = self.predict_batch(patches)
            for idx, ((is_real, sc), bbox) in enumerate(zip(preds, bboxes)):
                face_results.append(FaceResult(
                    face_id     = idx + 1,
                    is_real     = is_real,
                    spoof_score = round(sc, 4),
                    real_score  = round(1.0 - sc, 4),
                    bbox        = bbox,
                ))

        latency_ms = (time.perf_counter() - t0) * 1000

        return ImageResult(
            file       = filename,
            face_count = len(faces),
            faces      = face_results,
            latency_ms = round(latency_ms, 2),
        )

    def predict_file(self, image_path: str | Path) -> ImageResult:
        """
        Baca file gambar dari disk lalu jalankan predict_image.

        Parameters
        ----------
        image_path : path absolut atau relatif ke file gambar

        Returns
        -------
        ImageResult
        """
        img_path = Path(image_path)
        img      = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Gagal membaca gambar: {img_path}")
        return self.predict_image(img, filename=img_path.name)


# ──────────────────────────────────────────────────────────────
# Utilitas: benchmark latency
# ──────────────────────────────────────────────────────────────
def run_benchmark(
    detector:  SpoofDetector,
    image_dir: str | Path,
    n_runs:    int = 10,
) -> None:
    """
    Ukur rata-rata latency & throughput di semua gambar dalam folder.

    Parameters
    ----------
    detector  : instance SpoofDetector
    image_dir : folder berisi gambar uji
    n_runs    : berapa kali setiap gambar diproses (untuk rata-rata yang stabil)
    """
    image_dir = Path(image_dir)
    images = [
        cv2.imread(str(p))
        for p in sorted(image_dir.iterdir())
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    images = [img for img in images if img is not None]

    if not images:
        print(f"[!] Tidak ada gambar di {image_dir}")
        return

    latencies: list[float] = []
    for _ in range(n_runs):
        for img in images:
            r = detector.predict_image(img)
            latencies.append(r.latency_ms)

    arr = np.array(latencies)
    print(f"\n{'='*55}")
    print(f"  BENCHMARK  [{len(images)} gambar × {n_runs} run]")
    print(f"  avg latency : {arr.mean():.1f} ms/gambar")
    print(f"  p50 latency : {np.percentile(arr, 50):.1f} ms")
    print(f"  p95 latency : {np.percentile(arr, 95):.1f} ms")
    print(f"  min / max   : {arr.min():.1f} / {arr.max():.1f} ms")
    print(f"  throughput  : {1000 / arr.mean():.1f} gambar/detik")
    print(f"{'='*55}\n")


# ──────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────
def _print_result(result: ImageResult) -> None:
    label_map = {True: "REAL  [OK]", False: "SPOOF [!!]"}
    if not result.faces:
        print(f"  [{result.file}]  Tidak ada wajah terdeteksi  ({result.latency_ms} ms)")
        return
    for face in result.faces:
        label = label_map[face.is_real]
        print(
            f"  [{result.file}]  Wajah #{face.face_id}: {label}"
            f"  spoof={face.spoof_score:.4f}  real={face.real_score:.4f}"
            f"  [{result.latency_ms} ms]"
        )


def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Production Face Anti-Spoof Detector (ONNX Runtime)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image",     type=str,   help="Path ke satu gambar")
    parser.add_argument("--folder",    type=str,   help="Path ke folder gambar")
    parser.add_argument("--thresh",    type=float, default=SPOOF_THRESH,
                        help="Threshold skor spoof")
    parser.add_argument("--threads",   type=int,   default=4,
                        help="Jumlah ORT intra-op threads")
    parser.add_argument("--margin",    type=float, default=0.2,
                        help="Padding wajah (fraksi dari ukuran bbox)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Jalankan benchmark latency (butuh --folder)")
    parser.add_argument("--runs",      type=int,   default=10,
                        help="Jumlah run untuk benchmark")
    args = parser.parse_args()

    # Inisialisasi detector (singleton, model load sekali)
    detector = SpoofDetector(
        spoof_thresh  = args.thresh,
        num_threads   = args.threads,
        face_margin   = args.margin,
    )

    # ── Mode benchmark ──────────────────────────────────────────
    if args.benchmark:
        folder = args.folder or str(ROOT / "test")
        run_benchmark(detector, folder, n_runs=args.runs)
        return

    # ── Single image ────────────────────────────────────────────
    if args.image:
        result = detector.predict_file(args.image)
        print()
        _print_result(result)
        print()
        return

    # ── Folder mode ─────────────────────────────────────────────
    folder = Path(args.folder) if args.folder else ROOT / "test"
    if not folder.exists():
        parser.print_help()
        return

    files = [f for f in sorted(folder.iterdir()) if f.suffix.lower() in IMAGE_EXTENSIONS]
    if not files:
        print(f"[!] Tidak ada gambar di folder: {folder}")
        return

    print(f"\n{'='*65}")
    print(f"  Production Anti-Spoof  |  {len(files)} gambar  |  model: antispoof.onnx")
    print(f"{'='*65}")

    total_real = total_spoof = 0
    for img_path in files:
        result = detector.predict_file(img_path)
        _print_result(result)
        for f in result.faces:
            if f.is_real:
                total_real += 1
            else:
                total_spoof += 1

    print(f"\n{'='*65}")
    print(f"  RINGKASAN  →  REAL: {total_real}   SPOOF: {total_spoof}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
