from ultralytics import YOLO

model = YOLO("yolo26l.pt")

model.train(
    data="yolo_detection_split/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device="mps",
    workers=16,
    patience=0,
    name="yolo26n-all-classes-clean-model-x",
)

print("\nPer-class validation results:")
for i, name in model.names.items():
    print(
        f"{name:20} "
        f"P={metrics.box.p[i]:.3f} "
        f"R={metrics.box.r[i]:.3f} "
        f"mAP50={metrics.box.ap50[i]:.3f} "
        f"mAP50-95={metrics.box.ap[i]:.3f}"
    )