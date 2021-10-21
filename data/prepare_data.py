from pathlib import Path
import json

import numpy as np
import nibabel as nib
from tqdm import tqdm


DATA_PATH = Path("./Task07_Pancreas/")

MAX_IMAGES = 5000

def downscale(img):
    return img[::2, ::2, :].copy()


def load(path):
    return nib.load(path).get_fdata().astype(np.float32)


def main():
    with open(DATA_PATH / "dataset.json", "r") as f:
        dataset = json.load(f)

    imgs = []
    print("loading images...")
    for pair in tqdm(dataset["training"]):
        imgs.append(downscale(load(DATA_PATH / pair["image"])))

    imgs = np.concatenate(imgs, axis=-1)

    # fit image within [0,1]
    imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
    imgs = np.moveaxis(imgs, -1, 0)
    np.save("images.npy", imgs[:MAX_IMAGES].copy())

    masks = []
    print("loading masks...")
    for pair in tqdm(dataset["training"]):
        masks.append(downscale(load(DATA_PATH / pair["label"])))

    masks = np.concatenate(masks, axis=-1)
    masks = np.moveaxis(masks, -1, 0)

    np.save("masks.npy", masks[:MAX_IMAGES].copy())

    boxes = []

    print("creating bounding boxes...")
    for i in range(masks.shape[0]):
        mask = masks[i]
        if (mask > 0).sum():
            a = np.where(mask != 0)
            bbox = np.array([np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])])
            boxes.append(bbox)
        else:
            boxes.append(None)

    boxes = np.array(boxes)
    np.save("bounding_boxes.npy", boxes[:MAX_IMAGES].copy())


if __name__ == "__main__":
    main()
