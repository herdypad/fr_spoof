Letakkan model anti-spoof ONNX di sini dengan nama: antispoof.onnx

Rekomendasi model:
─────────────────────────────────────────────────────
MiniFASNetV2 (CVPR 2020) — akurasi tinggi, input 128×128
  Repo  : https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
  Konversi ke ONNX menggunakan script export dari repo di atas.

Setelah file antispoof.onnx ada di sini, jalankan:
  python spoof_detector.py

Jika file ONNX tidak ada, script otomatis fallback ke
metode tekstur LBP (akurasi lebih rendah, tapi tetap berjalan).
