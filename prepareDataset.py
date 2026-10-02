from datasets import load_dataset
from PIL import Image
import numpy as np

ds = load_dataset("NUbots/segmentation")

images = []
masks = []

for example in ds["train"]:
    if example["label"] == 0:
        images.append(example["image"])
    elif example["label"] == 1:
        masks.append(example["image"])

print("Images:", len(images))
print("Masks:", len(masks))

print("\nImage sizes:")
for image in images[:10]:
    print(image.size)

print("\nMask sizes:")
for mask in masks[:10]:
    arr = np.array(mask)

    print(
        "size:", mask.size,
        "mode:", mask.mode,
        "unique:", np.unique(arr.reshape(-1, 3), axis=0)[:20]
    )