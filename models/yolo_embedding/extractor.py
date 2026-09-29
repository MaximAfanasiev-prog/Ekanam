"""Frozen YOLO-cls backbone as a vehicle embedding extractor (no fine-tuning).

The embedding is the input of the final `Linear(1280 -> 1000)` of the Ultralytics `Classify` head,
i.e. `GAP(Conv1x1(backbone features))` - the last activation before the ImageNet classifier.
"""

import os

os.environ.setdefault("YOLO_OFFLINE", "1")  # never reach the network at inference time

import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from .data import MODULE_DIR

WEIGHTS_DIR = MODULE_DIR / "weights"
MODELS = [f"yolov8{s}-cls" for s in "nsmlx"] + [f"yolo26{s}-cls" for s in "nsmlx"]


def weights_path(name):
    return WEIGHTS_DIR / f"{name}.pt"


class YoloEmbedder:
    def __init__(self, name, device=None):
        self.name = name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        yolo = YOLO(str(weights_path(name)), task="classify")
        self.model = yolo.model.to(self.device).eval().float()
        head = self.model.model[-1]
        assert type(head).__name__ == "Classify", f"{name}: unexpected head {type(head).__name__}"
        self.dim = head.linear.in_features
        self._feat = None
        head.linear.register_forward_hook(lambda m, inp, out: setattr(self, "_feat", inp[0]))

    @torch.inference_mode()
    def __call__(self, x):
        """x: float tensor B x 3 x H x W in [0, 1] -> L2-normalized embeddings B x dim (on device)."""
        self.model(x.to(self.device, non_blocking=True))
        return F.normalize(self._feat.float(), dim=1)

    def embed(self, images, batch_size=128):
        """images: float tensor N x 3 x H x W -> np.float32 N x dim, L2-normalized."""
        out = [self(images[i : i + batch_size]).cpu() for i in range(0, len(images), batch_size)]
        return torch.cat(out).numpy().astype(np.float32)


def weights_size_mb(name):
    return weights_path(name).stat().st_size / 2**20
