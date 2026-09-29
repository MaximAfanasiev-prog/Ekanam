"""YOLO-cls backbone fine-tuned for vehicle Re-ID: BNNeck embedding + margin ID head (training only).

    image -> YOLO-cls layers except Classify -> Classify.conv (1x1, c -> 1280) -> GAP -> feat (1280)
          -> BatchNorm1d, bias frozen at 0 (BNNeck) -> emb (1280)

`feat` feeds the batch-hard triplet loss, `emb` feeds the CosFace/ArcFace head (Bag of Tricks, Luo et al. 2019).
At inference one of them is L2-normalized and used as the embedding, see FinetunedEmbedder for which one.
Everything up to GAP is initialized from the
ImageNet checkpoint, so before training `feat` is exactly the frozen baseline embedding
(models/yolo_embedding/extractor.py). The ImageNet `Linear(1280 -> 1000)` is dropped.
"""

import math
import os

os.environ.setdefault("YOLO_OFFLINE", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.yolo_embedding.extractor import weights_path
from ultralytics import YOLO


class ReIDNet(nn.Module):
    def __init__(self, name):
        super().__init__()
        cls = YOLO(str(weights_path(name)), task="classify").model  # architecture + ImageNet weights
        head = cls.model[-1]
        assert type(head).__name__ == "Classify", f"{name}: unexpected head {type(head).__name__}"
        self.name = name
        self.backbone = cls.model[:-1]
        self.neck = head.conv
        self.dim = head.linear.in_features
        self.bnneck = nn.BatchNorm1d(self.dim)
        self.float().requires_grad_(True)  # Ultralytics loads checkpoints with requires_grad=False
        self.bnneck.bias.requires_grad_(False)

    def forward(self, x):
        """x: B x 3 x H x W in [0, 1] -> (feat: pre-BN, for triplet; emb: post-BN, for ID loss and retrieval)."""
        feat = self.neck(self.backbone(x)).mean((2, 3))
        return feat, self.bnneck(feat)


class MarginHead(nn.Module):
    """Normalized-softmax classifier with an additive margin on the target class, over the train vehicle_ids.

    cosface: s * (cos(theta) - m)      (Wang et al. 2018, "CosFace")
    arcface: s * cos(theta + m)        (Deng et al. 2019, "ArcFace"), with the usual fallback for theta > pi - m
    """

    def __init__(self, dim, n_classes, kind="cosface", s=30.0, m=0.35):
        super().__init__()
        assert kind in ("cosface", "arcface"), kind
        self.kind, self.s, self.m = kind, s, m
        self.weight = nn.Parameter(torch.randn(n_classes, dim) * 0.01)

    def forward(self, emb, labels):
        cos = F.linear(F.normalize(emb.float()), F.normalize(self.weight.float())).clamp(-1 + 1e-7, 1 - 1e-7)
        if self.kind == "cosface":
            target = cos - self.m
        else:
            sin = (1 - cos**2).sqrt()
            phi = cos * math.cos(self.m) - sin * math.sin(self.m)
            target = torch.where(cos > math.cos(math.pi - self.m), phi, cos - math.sin(math.pi - self.m) * self.m)
        onehot = F.one_hot(labels, cos.size(1)).bool()
        return self.s * torch.where(onehot, target, cos), cos


def save_embedding_net(net, path, meta, half=False):
    """Stores only what inference needs (no ID head, no optimizer): backbone + neck + BNNeck and a meta dict."""
    sd = {k: (v.half() if half and v.is_floating_point() else v).cpu() for k, v in net.state_dict().items()}
    torch.save({"model": net.name, "state_dict": sd, "meta": meta}, path)


def load_embedding_net(path, device="cpu"):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    net = ReIDNet(ckpt["model"])
    net.load_state_dict({k: v.float() for k, v in ckpt["state_dict"].items()})
    return net.to(device).eval(), ckpt["meta"]


FEATURES = ("feat", "emb")


def default_feature(meta):
    """`emb` (post-BNNeck) only if the ID loss trained the BNNeck; with w_id = 0 its affine part gets no gradient
    and it merely standardizes `feat` with running stats, which costs 2.5-3 mAP@10 points on all three mini-val
    splits (runs/yolo_finetune/eval/posteval.json). With w_id > 0 `emb` is 1-4 points better than `feat`."""
    return "feat" if meta["config"]["w_id"] == 0 else "emb"


class FinetunedEmbedder:
    """Drop-in replacement for models.yolo_embedding.extractor.YoloEmbedder (same __call__/embed/dim/name).

    feature: "feat" or "emb", default by the checkpoint's training config (default_feature).
    flip:    horizontal-flip TTA - the image and its mirror go through the net as one batch and their outputs
             are summed before L2-normalization; +1.5-2.5 mAP@10 on every split and every checkpoint tried.
    """

    def __init__(self, checkpoint, device=None, feature=None, flip=True):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net, self.meta = load_embedding_net(checkpoint, self.device)
        self.feature = feature or default_feature(self.meta)
        assert self.feature in FEATURES, self.feature
        self.flip = flip
        self.name, self.dim = self.net.name, self.net.dim

    @torch.inference_mode()
    def __call__(self, x):
        """x: float tensor B x 3 x H x W in [0, 1] -> L2-normalized embeddings B x dim (on device)."""
        x = x.to(self.device, non_blocking=True)
        if self.flip:
            x = torch.cat([x, x.flip(-1)])
        out = self.net(x)[FEATURES.index(self.feature)].float()
        if self.flip:
            out = out[: len(out) // 2] + out[len(out) // 2 :]
        return F.normalize(out, dim=1)

    def embed(self, images, batch_size=128):
        out = [self(images[i : i + batch_size]).cpu() for i in range(0, len(images), batch_size)]
        return torch.cat(out).numpy().astype(np.float32)
