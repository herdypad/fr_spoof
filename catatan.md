# Cara buat environment dan run (macOS)

## 1) Buat virtual environment
```bash
cd /Users/herdypad/Kantor/FR_ONE
python3 -m venv .venv
source .venv/bin/activate
```

## 2) Install dependency
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3) Siapkan data test
Taruh gambar uji (`.jpg/.jpeg/.png/.bmp/.webp`) ke folder:

```bash
test/
```

## 4) Run spoof detector
```bash
python spoof_detector.py
```

Saat pertama jalan, bobot model akan terunduh otomatis ke folder `models/`.

## 5) Kalau error module belum ada
Jalankan lagi:

```bash
pip install -r requirements.txt
```

Lalu ulangi:

```bash
python spoof_detector.py


```



api
http://localhost:8000/docs.