Taruh foto-foto yang ingin diuji di sini.
Format yang didukung: .jpg  .jpeg  .png  .bmp  .webp

Contoh struktur:
  test/
  ├── real_person.jpg      ← wajah asli
  ├── printed_photo.jpg    ← foto dari kertas (spoof)
  └── screen_capture.png   ← foto dari layar (spoof)



# Step 1: Konversi model (Python)
pip install onnx onnx-tf tensorflow
python convert_model.py

# Step 2: Flutter
cd flutter_antispoof
flutter pub get
flutter run