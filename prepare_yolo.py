from pathlib import Path
import shutil
import yaml

SOURCE = Path.home() / ".cache/huggingface/hub/datasets--NUbots--segmentation/snapshots/164e4ad1a0bbdd54835489f456b9509c80057589/torso21/train"
OUTPUT = Path("yolo_detection")

CLASSES = [
    "ball",
    "robot",
    "goalpost",
    "L-Intersection",
    "T-Intersection",
    "X-Intersection",
]

CLASS_IDS = {name: i for i, name in enumerate(CLASSES)}

# Size of the artificial bounding box around point annotations.
POINT_BOX_SIZE = 10

images_out = OUTPUT / "images" / "train"
labels_out = OUTPUT / "labels" / "train"

images_out.mkdir(parents=True, exist_ok=True)
labels_out.mkdir(parents=True, exist_ok=True)

with open("torso21_train_annotations.yaml") as f:
    annotations = yaml.safe_load(f)["images"]

source_images = SOURCE / "images"

count_images = 0
count_labels = 0
count_skipped = 0

for image_path in source_images.iterdir():
    if not image_path.is_file():
        continue

    name = image_path.name

    if name not in annotations:
        continue

    data = annotations[name]

    width = data["width"]
    height = data["height"]

    labels = []

    for ann in data["annotations"]:
        obj_type = ann["type"]

        if obj_type not in CLASS_IDS:
            continue

        if not ann.get("in_image", False):
            continue

        vector = ann.get("vector", [])

        if len(vector) == 2:
            # Normal bounding box: [[x1,y1], [x2,y2]]
            x1, y1 = vector[0]
            x2, y2 = vector[1]

        elif len(vector) == 4:
            # Goalpost quadrilateral.
            xs = [p[0] for p in vector]
            ys = [p[1] for p in vector]

            x1 = min(xs)
            y1 = min(ys)
            x2 = max(xs)
            y2 = max(ys)

        elif len(vector) == 1:
            # Intersection is a point annotation.
            x, y = vector[0]

            half = POINT_BOX_SIZE / 2

            x1 = max(0, x - half)
            y1 = max(0, y - half)
            x2 = min(width, x + half)
            y2 = min(height, y + half)

        else:
            count_skipped += 1
            continue

        # Make sure coordinates are valid.
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))

        if x2 <= x1 or y2 <= y1:
            count_skipped += 1
            continue

        # Convert to YOLO format:
        # class x_center y_center width height
        xc = ((x1 + x2) / 2) / width
        yc = ((y1 + y2) / 2) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height

        labels.append(
            f"{CLASS_IDS[obj_type]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
        )

    # Copy image.
    destination_image = images_out / image_path.name
    shutil.copy2(image_path, destination_image)

    # Write labels. Empty label files are valid for images with no target
    # objects in the image.
    label_path = labels_out / f"{image_path.stem}.txt"
    label_path.write_text("\n".join(labels))

    count_images += 1
    count_labels += len(labels)

# Dataset configuration.
data = {
    "path": str(OUTPUT.resolve()),
    "train": "images/train",
    "val": "images/train",
    "names": {i: name for i, name in enumerate(CLASSES)},
}

with open(OUTPUT / "data.yaml", "w") as f:
    yaml.safe_dump(data, f, sort_keys=False)

print(f"Images: {count_images}")
print(f"Object labels: {count_labels}")
print(f"Skipped annotations: {count_skipped}")
print(f"Dataset: {OUTPUT.resolve()}")
