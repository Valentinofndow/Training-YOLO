import os
from ultralytics import YOLO
import cv2

# Path video input
video_path = "D:/UMN/Kegiatan/Proyek/KRI/test02/test/bola1.mp4"
video_path_out = "{}_out.mp4".format(video_path)

# Buka video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Tidak bisa membuka video {video_path}")
    exit()

# Baca frame pertama untuk mendapatkan ukuran video
ret, frame = cap.read()
if not ret:
    print("Error: Tidak bisa membaca frame dari video")
    exit()

H, W, _ = frame.shape

# Buat VideoWriter untuk menyimpan hasil
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'XVID'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# Path model
model_path = "D:/UMN/Kegiatan/Proyek/runs/detect/train2/weights/last.pt"

# Load model
if not os.path.exists(model_path):
    print(f"Error: File model {model_path} tidak ditemukan!")
    exit()

model = YOLO(model_path)  # Load model custom

# Threshold untuk deteksi
threshold = 0.5

# Proses video
while ret:
    # Deteksi objek
    results = model(frame)[0]

    # Gambar bounding box dan label
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Tulis frame ke video output
    out.write(frame)

    # Baca frame berikutnya
    ret, frame = cap.read()

# Tutup video
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video hasil disimpan di: {video_path_out}")