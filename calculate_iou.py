from pathlib import Path
from ultralytics import YOLO
import numpy as np

MODEL = "runs/detect/train-3/weights/best.pt"
IMAGE_DIR = Path("yolo_detection/images/train")
LABEL_DIR = Path("yolo_detection/labels/train")

CLASS_NAMES = {
    0: "ball",
    1: "robot",
    2: "goalpost",
    3: "L-Intersection",
    4: "T-Intersection",
    5: "X-Intersection",
}

model = YOLO(MODEL)

iou_by_class = {name: [] for name in CLASS_NAMES.values()}
all_ious = []

def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


image_paths = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
)

print(f"Images: {len(image_paths)}")

results = model.predict(
    source=str(IMAGE_DIR),
    conf=0.25,
    iou=0.7,
    verbose=False,
)

for image_path, result in zip(image_paths, results):

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        continue

    # Image dimensions
    height, width = result.orig_shape

    # Ground-truth boxes
    ground_truth = []

    with open(label_path) as f:
        for line in f:
            values = line.strip().split()

            if len(values) != 5:
                continue

            cls, xc, yc, w, h = map(float, values)

            x1 = (xc - w / 2) * width
            y1 = (yc - h / 2) * height
            x2 = (xc + w / 2) * width
            y2 = (yc + h / 2) * height

            ground_truth.append({
                "class": int(cls),
                "box": [x1, y1, x2, y2],
            })

    # Predictions
    predictions = []

    if result.boxes is not None:
        for box, cls in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
        ):
            predictions.append({
                "class": int(cls),
                "box": box.tolist(),
            })

    # Match predictions to GT boxes
    matched_gt = set()

    for pred in predictions:

        best_iou = 0
        best_gt = None

        for i, gt in enumerate(ground_truth):

            if i in matched_gt:
                continue

            if pred["class"] != gt["class"]:
                continue

            iou = box_iou(pred["box"], gt["box"])

            if iou > best_iou:
                best_iou = iou
                best_gt = i

        if best_gt is not None:
            matched_gt.add(best_gt)

            class_name = CLASS_NAMES[pred["class"]]

            iou_by_class[class_name].append(best_iou)
            all_ious.append(best_iou)


print("\n========== IoU RESULTS ==========")

if all_ious:
    print(f"Mean IoU: {np.mean(all_ious):.4f}")
    print(f"Mean IoU: {np.mean(all_ious) * 100:.2f}%")
else:
    print("No matched detections found.")

print("\nPer-class IoU:")

for class_name, values in iou_by_class.items():

    if values:
        print(
            f"{class_name:17s} "
            f"{np.mean(values):.4f} "
            f"({np.mean(values) * 100:.2f}%) "
            f"[{len(values)} matches]"
        )
    else:
        print(f"{class_name:17s} No matches")