"""Train/val data for fine-tuning: cached preprocessed crops, split loading and GPU augmentations.

Every image of train.csv is cropped by bbox with margin 0 and preprocessed by the SAME function as in the frozen
pipeline (models.yolo_embedding.data.load_crops + preprocess_batch), then stored as uint8. TF.to_tensor
produces uint8/255, so the uint8 cache is lossless: `cache[i] / 255` is bit-identical to what the frozen
pipeline feeds the model. Augmentations are applied on top of it, on the GPU, only during training.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from models.yolo_embedding.data import DATASET_DIR, REPO_DIR, load_crops, preprocess_batch, read_table

CACHE_DIR = REPO_DIR / "runs/yolo_finetune/cache"


def train_table():
    return read_table(DATASET_DIR / "train.csv")


def preprocessed_train(mode="center_crop", chunk=512):
    """uint8 tensor [len(train.csv), 3, 224, 224] in train.csv row order (built once per mode, then cached)."""
    df = train_table()
    path, ids_path = CACHE_DIR / f"train_{mode}.npy", CACHE_DIR / f"train_{mode}.ids"
    if path.exists() and ids_path.read_text().split() == df.image_id.tolist():
        return torch.from_numpy(np.load(path))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(0, len(df), chunk):
        x = preprocess_batch(load_crops(df.iloc[i : i + chunk]), mode)
        out.append((x * 255).round().to(torch.uint8))
    arr = torch.cat(out)
    np.save(path, arr.numpy())
    ids_path.write_text("\n".join(df.image_id))
    return arr


def full_train_split():
    """Every vehicle of train.csv goes to training, there is no validation part (train.py --full-train)."""
    df = train_table()
    return {"split": {"seed": None}, "train_rows": df.index.to_numpy(), "train_vid": df.vehicle_id.to_numpy(),
            "train_cam": df.camera_id.to_numpy(), "dir": None}


def load_split(split_dir):
    """Open-set split produced by models.yolo_embedding.make_split (or converted to its format).

    Returns train-part row indices into train.csv and the val query/gallery with their GT columns.
    camera_id is returned for the sampler and for the junk filter of the metric, never as a model input.
    """
    split_dir = Path(split_dir)
    split = json.loads((split_dir / "mini_val_split.json").read_text())
    df = train_table()
    train_ids, val_ids = set(split["train_vehicle_ids"]), set(split["val_vehicle_ids"])
    assert not train_ids & val_ids, "vehicle_id overlap between train and val"
    row = {img: i for i, img in enumerate(df.image_id)}
    q_df, g_df = read_table(split_dir / "mini_query.csv"), read_table(split_dir / "mini_gallery.csv")
    gt = read_table(split_dir / "mini_gt.csv").set_index("image_id")
    assert set(gt.vehicle_id) <= val_ids
    tr = df[df.vehicle_id.isin(train_ids)]
    return {
        "split": split,
        "train_rows": tr.index.to_numpy(),
        "train_vid": tr.vehicle_id.to_numpy(),
        "train_cam": tr.camera_id.to_numpy(),
        "q_rows": np.array([row[i] for i in q_df.image_id]),
        "g_rows": np.array([row[i] for i in g_df.image_id]),
        "q_ids": q_df.image_id.tolist(),
        "g_ids": g_df.image_id.tolist(),
        "q_vid": gt.loc[q_df.image_id, "vehicle_id"].to_numpy(),
        "q_cam": gt.loc[q_df.image_id, "camera_id"].to_numpy(),
        "g_vid": gt.loc[g_df.image_id, "vehicle_id"].to_numpy(),
        "g_cam": gt.loc[g_df.image_id, "camera_id"].to_numpy(),
        "dir": split_dir,
    }


class GpuAugment:
    """Standard Re-ID augmentations, per sample, on a float batch in [0, 1] (B x 3 x H x W, on GPU).

    - horizontal flip, p = 0.5 (a mirrored car is still the same car; plates are blurred anyway);
    - light color jitter: brightness +-20%, contrast / saturation +-15%, NO hue shift (color is identity);
    - zero pad 10 px + random crop back to H x W (small translation, as in Bag of Tricks);
    - random erasing, p = 0.5, area 2-40%, aspect 0.3-3.3, filled with the ImageNet mean (Zhong et al. 2020).
    """

    def __init__(self, flip=0.5, brightness=0.2, contrast=0.15, saturation=0.15, pad=10, erase_p=0.5):
        self.flip, self.b, self.c, self.s, self.pad, self.erase_p = flip, brightness, contrast, saturation, pad, erase_p
        self.fill = torch.tensor([0.485, 0.456, 0.406])[:, None, None]

    def __call__(self, x, gen):
        """Fully vectorized (no per-sample Python loop, no host syncs)."""
        B, C, H, W = x.shape
        dev = x.device
        u = lambda *s: torch.rand(*s, device=dev, generator=gen)
        x = torch.where((u(B) < self.flip)[:, None, None, None], x.flip(-1), x)

        luma = lambda t: (0.299 * t[:, 0] + 0.587 * t[:, 1] + 0.114 * t[:, 2])[:, None]
        x = x * (1 + (2 * u(B, 1, 1, 1) - 1) * self.b)
        mean = luma(x).mean((2, 3), keepdim=True)
        x = (x - mean) * (1 + (2 * u(B, 1, 1, 1) - 1) * self.c) + mean
        gray = luma(x)
        x = ((x - gray) * (1 + (2 * u(B, 1, 1, 1) - 1) * self.s) + gray).clamp(0, 1)

        p = self.pad
        xp = F.pad(x, (p, p, p, p))
        oy, ox = (u(B) * (2 * p + 1)).long(), (u(B) * (2 * p + 1)).long()
        rows = (oy[:, None] + torch.arange(H, device=dev))[:, None, :, None]
        cols = (ox[:, None] + torch.arange(W, device=dev))[:, None, None, :]
        x = xp[torch.arange(B, device=dev)[:, None, None, None], torch.arange(C, device=dev)[None, :, None, None],
               rows, cols]

        # random erasing: 10 candidate rectangles per sample, the first one that fits inside the image is used
        n = 10
        area = H * W * (0.02 + 0.38 * u(B, n))
        ar = torch.exp(np.log(0.3) + (np.log(1 / 0.3) - np.log(0.3)) * u(B, n))
        h, w = (area * ar).sqrt().round(), (area / ar).sqrt().round()
        ok = (h < H) & (w < W)
        first = torch.where(ok.any(1), ok.float().argmax(1), torch.zeros_like(ok[:, 0], dtype=torch.long))
        pick = lambda t: t.gather(1, first[:, None])[:, 0]
        h, w = pick(h), pick(w)
        do = (u(B) < self.erase_p) & ok.any(1)
        y0, x0 = (u(B) * (H - h)).floor(), (u(B) * (W - w)).floor()
        yy, xx = torch.arange(H, device=dev)[None, :, None], torch.arange(W, device=dev)[None, None, :]
        m = ((yy >= y0[:, None, None]) & (yy < (y0 + h)[:, None, None])
             & (xx >= x0[:, None, None]) & (xx < (x0 + w)[:, None, None]))
        m = (m & do[:, None, None])[:, None]
        return torch.where(m, self.fill.to(dev)[None], x)
