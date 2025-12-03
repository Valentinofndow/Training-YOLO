from ultralytics import YOLO
import cv2
import time

# Inisialisasi webcam
cap = cv2.VideoCapture(0)  # 0 untuk kamera bawaan laptop
cap.set(3, 1280)
cap.set(4, 720)
if not cap.isOpened():
    print("Error: Tidak bisa membuka webcam!")
    exit()

# Baca frame pertama untuk mendapatkan ukuran video
ret, frame = cap.read()
if not ret:
    print("Error: Tidak bisa membaca frame dari webcam")
    exit()

# Path model
model_path = "model/best.pt"

# Load model YOLO
model = YOLO(model_path)  # Load model custom

# Threshold untuk deteksi
threshold = 0.5

# Variabel untuk menghitung FPS
prev_time = 0  # Waktu sebelumnya
curr_time = 0  # Waktu sekarang

# Parameter konversi jarak
real_width_cm = 6.2  # Lebar sebenarnya objek (contoh: bola)
focal_length_px = 967.75  # Estimasi focal length, perlu kalibrasi

# Loop untuk live webcam
while True:
    # Baca frame dari webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Tidak bisa membaca frame dari webcam")
        break

    # Dapatkan ukuran frame
    height, width, _ = frame.shape

    # Tentukan titik tengah layar
    center = (width // 2, height // 2)

    # Gambar titik tengah layar (DOT tengah)
    cv2.circle(frame, center, 5, (255, 255, 255), -1)  # Titik putih di tengah

    # Deteksi objek dengan YOLO
    results = model(frame)[0]

    # Gambar bounding box dan label
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            # Gambar bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            # Tambahkan label
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

            # Hitung titik tengah objek
            obj_center = (int((x1 + x2) // 2), int((y1 + y2) // 2))

            # Gambar garis dari titik tengah layar ke titik tengah objek
            cv2.line(frame, center, obj_center, (0, 0, 255), 2)

            # Hitung ukuran bounding box
            object_width_px = int(x2 - x1)
            object_height_px = int(y2 - y1)
            object_length_cm = round(((obj_center[0] - center[0]) ** 2 + (obj_center[1] - center[1]) ** 2) ** 0.5 / focal_length_px * real_width_cm, 2)

            # Hitung jarak dari kamera ke objek dalam cm
            if object_width_px > 0:
                distance_cm = (real_width_cm * focal_length_px) / object_width_px
                distance_cm = round(distance_cm, 2)
                
                # Tampilkan informasi di bawah objek
                text_position = (int(x1), int(y2) + 30)
                cv2.putText(frame, f"Width: {object_width_px} px", text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(frame, f"Height: {object_height_px} px", (int(x1), int(y2) + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(frame, f"Length: {object_length_cm} cm", (int(x1), int(y2) + 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(frame, f"Distance: {distance_cm} cm", (int(x1), int(y2) + 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(frame, f"Confidence: {score:.2f}", (int(x1), int(y2) + 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3, cv2.LINE_AA)

    # Hitung FPS
    curr_time = time.time()  # Waktu sekarang
    fps = 1 / (curr_time - prev_time)  # Rumus FPS
    prev_time = curr_time  # Update waktu sebelumnya

    # Tampilkan FPS di frame (warna biru)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Tampilkan frame hasil deteksi
    cv2.imshow('YOLO Live Webcam Detection', frame)

    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup webcam dan jendela
cap.release()
cv2.destroyAllWindows()
