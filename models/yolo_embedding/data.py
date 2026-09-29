"""Loading annotations, cropping vehicles by bbox and preprocessing crops for YOLO-cls backbones."""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image

DATASET_DIR = Path(os.environ.get("FALCON_DATASET_DIR", "/home/andrey/datasets/lct26-street-falcon-reid/extracted"))
MODULE_DIR = Path(__file__).resolve().parent
REPO_DIR = MODULE_DIR.parents[1]

# Preprocessing modes (applied identically to every model):
#   center_crop - Ultralytics-native classify transform: resize short side to imgsz, center crop imgsz x imgsz
#   squash      - resize crop directly to imgsz x imgsz, aspect ratio is not preserved, nothing is cut off
#   letterbox   - resize long side to imgsz, pad to square with gray (114), nothing is cut off
PREPROCESS_MODES = ("center_crop", "squash", "letterbox")
LETTERBOX_FILL = 114


def read_table(path):
    return pd.read_csv(path, dtype={"image_id": str})


def load_crop(image_id, x, y, w, h, images_dir=None, margin=0.0):
    """Cuts the vehicle out of the frame by bbox as given by the organizers.

    `margin` is our own optional knob (NOT required by the data): relative padding added on every
    side of the bbox, e.g. 0.05 = +5% of w/h. Default 0 means the bbox is used as is.
    """
    images_dir = Path(images_dir or DATASET_DIR / "images")
    with Image.open(images_dir / f"{image_id}.jpg") as im:
        im = im.convert("RGB")
        W, H = im.size
        dx, dy = w * margin, h * margin
        box = (
            max(0, int(round(x - dx))),
            max(0, int(round(y - dy))),
            min(W, int(round(x + w + dx))),
            min(H, int(round(y + h + dy))),
        )
        return im.crop(box)


def load_crops(df, images_dir=None, margin=0.0, workers=16):
    """Loads crops for every row of an annotation table, preserving row order."""
    rows = list(df[["image_id", "x", "y", "w", "h"]].itertuples(index=False, name=None))
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda r: load_crop(*r, images_dir=images_dir, margin=margin), rows))


def preprocess(crop, mode="center_crop", imgsz=224):
    """PIL RGB crop -> float tensor 3 x imgsz x imgsz in [0, 1] (YOLO-cls uses no mean/std normalization)."""
    if mode == "center_crop":
        img = TF.center_crop(TF.resize(crop, imgsz, antialias=True), [imgsz, imgsz])
    elif mode == "squash":
        img = TF.resize(crop, [imgsz, imgsz], antialias=True)
    elif mode == "letterbox":
        w, h = crop.size
        s = imgsz / max(w, h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        img = Image.new("RGB", (imgsz, imgsz), (LETTERBOX_FILL,) * 3)
        img.paste(crop.resize((nw, nh), Image.BILINEAR), ((imgsz - nw) // 2, (imgsz - nh) // 2))
    else:
        raise ValueError(f"unknown preprocess mode {mode!r}, expected one of {PREPROCESS_MODES}")
    return TF.to_tensor(img)


def preprocess_batch(crops, mode="center_crop", imgsz=224, workers=16):
    with ThreadPoolExecutor(workers) as ex:
        return torch.stack(list(ex.map(lambda c: preprocess(c, mode, imgsz), crops)))
