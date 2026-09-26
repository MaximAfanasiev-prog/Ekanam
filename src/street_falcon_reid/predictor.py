from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .dataset import build_transform, crop_vehicle
from .model import ReIDModel
from .prepare import sha256_file
from .records import VehicleRecord


class InvalidBBox(ValueError):
    """The client must supply a rectangle inside the original image."""


@dataclass(frozen=True)
class BBox:
    x: int
    y: int
    w: int
    h: int

    def validate(self, image: Image.Image) -> None:
        if any(type(value) is not int for value in (self.x, self.y, self.w, self.h)):
            raise InvalidBBox("BBox x, y, w, h must be integers in pixels.")
        if self.x < 0 or self.y < 0 or self.w <= 0 or self.h <= 0:
            raise InvalidBBox("BBox requires x,y >= 0 and w,h > 0.")
        if self.x + self.w > image.width or self.y + self.h > image.height:
            raise InvalidBBox("BBox must fit inside the original image.")


def load_model(
    checkpoint_path: str | Path, device: str = "cpu"
) -> tuple[ReIDModel, dict[str, Any], torch.device]:
    target = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["model_config"]
    model = ReIDModel(
        backbone=str(config["backbone"]),
        embedding_dim=int(config["embedding_dim"]),
        num_classes=int(config["num_classes"]),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(target).eval()
    return model, checkpoint, target


class ImagePredictor:
    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        self.model_version = sha256_file(checkpoint_path)
        self.model, checkpoint, self.device = load_model(checkpoint_path, device)
        self.data_config = checkpoint["data_config"]
        for key in ("image_height", "image_width"):
            value = self.data_config[key]
            if type(value) is not int or not 1 <= value <= 4096:
                raise ValueError("Checkpoint image dimensions must be integers in 1..4096.")
        margin = float(self.data_config["crop_margin"])
        if not np.isfinite(margin) or not 0 <= margin <= 1:
            raise ValueError("Checkpoint crop margin must be finite and in 0..1.")
        self.embedding_dim = int(checkpoint["model_config"]["embedding_dim"])
        self.threshold = float(checkpoint["open_set_threshold"])
        if not np.isfinite(self.threshold):
            raise ValueError("Checkpoint threshold must be finite.")
        self.transform = build_transform(
            int(self.data_config["image_height"]),
            int(self.data_config["image_width"]),
            training=False,
        )

    def embed_image(self, image: Image.Image, bbox: BBox) -> np.ndarray:
        bbox.validate(image)
        record = VehicleRecord("upload", bbox.x, bbox.y, bbox.w, bbox.h)
        crop = crop_vehicle(image.convert("RGB"), record, float(self.data_config["crop_margin"]))
        tensor = self.transform(crop).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            vectors, _ = self.model(tensor)
        embedding = vectors[0].float().cpu().numpy()
        if not np.isfinite(embedding).all() or np.linalg.norm(embedding) <= 0:
            raise ValueError("Model produced an invalid embedding.")
        return embedding
