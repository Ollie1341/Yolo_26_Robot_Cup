from pathlib import Path
import random
import math
from PIL import Image, ImageDraw

# Paths relative to your VisionModel directory
IMAGE_DIR = Path("yolo_detection/images/train")
LABEL_DIR = Path("yolo_detection/labels/train")
OUTPUT_DIR = Path("intersection_label_inspection")

SAMPLES_PER_CLASS = 20
SEED = 42

CLASS_NAMES = {
    3: "L-Intersection",
    4: "T-Intersection",
    5: "X-Intersection",
}

random.seed(SEED)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_image(stem):
    """Find the image corresponding to a YOLO label file."""
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        path = IMAGE_DIR / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def collect_annotations():
    """Collect all L/T/X annotations."""
    annotations = {3: [], 4: [], 5: []}

    for label_path in LABEL_DIR.glob("*.txt"):
        image_path = find_image(label_path.stem)

        if image_path is None:
            continue

        with label_path.open("r") as f:
            for line_number, line in enumerate(f, start=1):
                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                cls = int(float(parts[0]))

                if cls not in annotations:
                    continue

                xc, yc, w, h = map(float, parts[1:])

                annotations[cls].append(
                    {
                        "image": image_path,
                        "label": label_path,
                        "line": line_number,
                        "xc": xc,
                        "yc": yc,
                        "w": w,
                        "h": h,
                    }
                )

    return annotations


def make_tile(item, cls, tile_size=(420, 320)):
    """Create one contact-sheet tile with the GT box drawn."""
    image_path = item["image"]

    with Image.open(image_path) as src:
        image = src.convert("RGB")

    width, height = image.size

    xc = item["xc"]
    yc = item["yc"]
    bw = item["w"]
    bh = item["h"]

    x1 = max(0, (xc - bw / 2) * width)
    y1 = max(0, (yc - bh / 2) * height)
    x2 = min(width, (xc + bw / 2) * width)
    y2 = min(height, (yc + bh / 2) * height)

    # Add surrounding context so we can judge the annotation.
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    pad_x = max(box_w * 1.5, width * 0.08)
    pad_y = max(box_h * 1.5, height * 0.08)

    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(width, x2 + pad_x)
    crop_y2 = min(height, y2 + pad_y)

    crop = image.crop(
        (
            int(crop_x1),
            int(crop_y1),
            int(crop_x2),
            int(crop_y2),
        )
    )

    # Reserve top area for filename/class text.
    usable_height = tile_size[1] - 40

    scale = min(
        tile_size[0] / crop.width,
        usable_height / crop.height,
    )

    resized_size = (
        max(1, int(crop.width * scale)),
        max(1, int(crop.height * scale)),
    )

    crop = crop.resize(resized_size)

    tile = Image.new("RGB", tile_size, "white")

    offset_x = (tile_size[0] - crop.width) // 2
    offset_y = 40 + (usable_height - crop.height) // 2

    tile.paste(crop, (offset_x, offset_y))

    draw = ImageDraw.Draw(tile)

    bx1 = offset_x + (x1 - crop_x1) * scale
    by1 = offset_y + (y1 - crop_y1) * scale
    bx2 = offset_x + (x2 - crop_x1) * scale
    by2 = offset_y + (y2 - crop_y1) * scale

    draw.rectangle(
        (bx1, by1, bx2, by2),
        outline="red",
        width=4,
    )

    draw.text(
        (8, 8),
        CLASS_NAMES[cls],
        fill="black",
    )

    # Shorten long filenames so they fit reasonably.
    filename = image_path.name
    if len(filename) > 52:
        filename = filename[:49] + "..."

    draw.text(
        (8, 22),
        filename,
        fill="black",
    )

    return tile


def main():
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

    if not LABEL_DIR.exists():
        raise FileNotFoundError(f"Label directory not found: {LABEL_DIR}")

    annotations = collect_annotations()

    print("Intersection annotation counts:")
    for cls, name in CLASS_NAMES.items():
        print(f"  {name}: {len(annotations[cls])}")

    print()

    for cls, class_name in CLASS_NAMES.items():
        available = annotations[cls]

        if not available:
            print(f"No annotations found for {class_name}")
            continue

        count = min(SAMPLES_PER_CLASS, len(available))
        selected = random.sample(available, count)

        columns = 4
        rows = math.ceil(count / columns)

        tile_width = 420
        tile_height = 320

        sheet = Image.new(
            "RGB",
            (
                columns * tile_width,
                rows * tile_height,
            ),
            "white",
        )

        for index, item in enumerate(selected):
            tile = make_tile(
                item,
                cls,
                (tile_width, tile_height),
            )

            x = (index % columns) * tile_width
            y = (index // columns) * tile_height

            sheet.paste(tile, (x, y))

        safe_name = class_name.lower().replace("-", "_")
        output_path = OUTPUT_DIR / f"{safe_name}_labels.jpg"

        sheet.save(output_path, quality=92)

        print(
            f"{class_name}: sampled {count} of {len(available)} "
            f"-> {output_path}"
        )

    print()
    print("Inspection complete.")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
