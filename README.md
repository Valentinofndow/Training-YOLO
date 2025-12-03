# Object Detection Project (YOLOv11)

Project ini berisi implementasi model YOLOv11 untuk mendeteksi objek (bola dan gawang) menggunakan Python dan OpenCV. Repo ini mencakup script untuk prediksi gambar, video, webcam, serta konfigurasi dataset dan contoh hasil.

## Struktur Folder
```
project/
│
├── data/
│   ├── images/
│   └── labels/
│
├── model/
│   └── best.pt
│
├── test/
│   ├── *.mp4
│   ├── *_detected.jpg
│   └── *.jpg
│
├── config.yml
├── google_colab_config.yaml
├── main.py
├── predict_video.py
├── predict_webcam.py
└── requirements.txt
```
## Fitur
- Deteksi objek (ball dan goalpost) menggunakan YOLOv11
- Prediksi gambar
- Prediksi video
- Prediksi webcam real-time
- Contoh input dan output tersedia di folder test

## Model
Model tersimpan pada:
model/best.pt  
Model ini merupakan hasil training YOLOv11 menggunakan dataset custom.

## Cara Menjalankan

### 1. Clone Repository
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo


### 2. Install Dependencies
pip install -r requirements.txt


### 3. Prediksi Gambar
python main.py


### 4. Prediksi Video
python predict_video.py


### 5. Prediksi Webcam
python predict_webcam.py


## Dataset
Struktur dataset mengikuti format YOLO standar (images + labels + file config.yml).

## Catatan
- Gunakan Python 3.8 atau lebih baru
- Pastikan library ultralytics sudah ter-install

## Lisensi
Repo ini digunakan untuk keperluan penelitian atau proyek pribadi.
