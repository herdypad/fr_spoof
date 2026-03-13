"""
api_test.py — Simple REST API for Face Anti-Spoof Detector
===========================================================
Jalankan:
    python api_test.py
    # atau
    uvicorn api_test:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /            → health check
    POST /predict     → upload gambar, return hasil spoof detection
    POST /predict/url → kirim URL gambar, return hasil spoof detection
"""

from __future__ import annotations

import io
import logging
import urllib.request

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl

from spoof_image import SpoofDetector

# ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(
    title="Face Anti-Spoof API",
    description="Deteksi gambar real vs spoof menggunakan ONNX model.",
    version="1.0.0",
)

# Singleton — model hanya di-load sekali saat startup
detector = SpoofDetector()


# ──────────────────────────────────────────────────────────────
# Schema response
# ──────────────────────────────────────────────────────────────
class FaceOut(BaseModel):
    face_id:     int
    is_real:     bool
    spoof_score: float
    real_score:  float
    bbox:        list[int]   # [x1, y1, x2, y2]

class PredictResponse(BaseModel):
    filename:    str
    face_count:  int
    faces:       list[FaceOut]
    is_real:     bool | None   # True = semua wajah real, False = ada spoof, None = tidak ada wajah
    latency_ms:  float


class UrlRequest(BaseModel):
    url: HttpUrl


# ──────────────────────────────────────────────────────────────
# Helper: bytes → cv2 image
# ──────────────────────────────────────────────────────────────
def _bytes_to_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="File bukan gambar atau format tidak didukung.")
    return img


def _result_to_response(result) -> PredictResponse:
    return PredictResponse(
        filename   = result.file,
        face_count = result.face_count,
        faces      = [
            FaceOut(
                face_id     = f.face_id,
                is_real     = f.is_real,
                spoof_score = f.spoof_score,
                real_score  = f.real_score,
                bbox        = f.bbox,
            )
            for f in result.faces
        ],
        is_real    = result.is_real,
        latency_ms = result.latency_ms,
    )


# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def health():
    return {"status": "ok", "model": "antispoof.onnx"}


@app.post("/predict", response_model=PredictResponse, tags=["Predict"])
async def predict(file: UploadFile = File(..., description="File gambar (jpg/png/bmp)")):
    """
    Upload gambar → deteksi wajah → return skor spoof tiap wajah.
    """
    content = await file.read()
    img     = _bytes_to_bgr(content)
    result  = detector.predict_image(img, filename=file.filename or "upload")
    return _result_to_response(result)


@app.post("/predict/url", response_model=PredictResponse, tags=["Predict"])
def predict_url(body: UrlRequest):
    """
    Kirim URL gambar → deteksi wajah → return skor spoof tiap wajah.
    """
    url = str(body.url)
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            content = resp.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Gagal mengunduh gambar: {exc}") from exc

    img    = _bytes_to_bgr(content)
    result = detector.predict_image(img, filename=url.split("/")[-1] or "url")
    return _result_to_response(result)


# ──────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api_test:app", host="0.0.0.0", port=8000, reload=False)
