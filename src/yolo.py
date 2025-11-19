from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="../data/object_detection_Dataset/data.yaml",
    epochs=40,
    imgsz=640,
    batch=16,
    device=0
)

