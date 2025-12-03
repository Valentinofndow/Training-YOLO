from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Train model
results = model.train(data="D:/UMN/Kegiatan/Proyek/KRI/test02/config.yml", epochs=200)