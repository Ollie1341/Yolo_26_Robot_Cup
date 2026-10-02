from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.train(
    data="yolo_detection/data.yaml",
    epochs=100,
    imgsz=640,
    device="mps",
)