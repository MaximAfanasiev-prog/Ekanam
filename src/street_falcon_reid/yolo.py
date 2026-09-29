"""Online adapter for the reviewed YOLO Re-ID recipe (source revision in bundle)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from PIL import Image
from torchvision.transforms import functional as transforms

from .predictor import BBox
from .prepare import sha256_file
from .search import GalleryIndex, SearchBusy, SearchService
from .yolo_rerank import _rerank_batch


class YoloPredictor:
    def __init__(self, directory: Path, manifest: dict, device: str):
        os.environ.setdefault("YOLO_OFFLINE", "1")
        from ultralytics import YOLO

        self.device = torch.device(device)
        self.model_version = manifest["sha256"]["checkpoint.pt"]
        checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
        config = checkpoint["meta"]["config"]
        if (
            checkpoint["model"] != "yolo26l-cls"
            or config["mode"] != "squash"
            or config["w_id"] != 0
            or checkpoint["meta"]["epoch"] != manifest["epoch"]
        ):
            raise ValueError("Unsupported YOLO checkpoint recipe.")
        # Trusted, checksum-verified Ultralytics architecture checkpoint.
        architecture = YOLO(str(directory / "base.pt"), task="classify").model
        head = architecture.model[-1]
        if type(head).__name__ != "Classify" or head.linear.in_features != 1280:
            raise ValueError("Unexpected YOLO classification architecture.")
        self.net = nn.Module()
        self.net.backbone = architecture.model[:-1]
        self.net.neck = head.conv
        self.net.bnneck = nn.BatchNorm1d(1280)
        self.net.float().load_state_dict(
            {key: value.float() for key, value in checkpoint["state_dict"].items()}
        )
        self.net.to(self.device).eval()
        self.embedding_dim = 1280
        self.threshold = float(manifest["threshold"])
        if not np.isfinite(self.threshold) or not -1 <= self.threshold <= 1:
            raise ValueError("Invalid YOLO threshold.")
        self.data_config = {"image_height": 224, "image_width": 224, "crop_margin": 0.0}
        self.model_name = "YOLO26-L Re-ID"
        self.threshold_source = "fixed test-derived 25% quantile; not online calibrated"

    def embed_image(self, image: Image.Image, bbox: BBox) -> np.ndarray:
        return self.embed_images([image], [bbox])[0]

    @torch.inference_mode()
    def embed_images(self, images: list[Image.Image], boxes: list[BBox]) -> np.ndarray:
        if not images or len(images) != len(boxes) or len(images) > 32:
            raise ValueError("Expected 1..32 images and matching boxes.")
        tensors = []
        for image, bbox in zip(images, boxes, strict=True):
            bbox.validate(image)
            with image.crop((bbox.x, bbox.y, bbox.x + bbox.w, bbox.y + bbox.h)) as crop:
                tensors.append(
                    transforms.to_tensor(
                        transforms.resize(crop.convert("RGB"), [224, 224], antialias=True)
                    )
                )
        tensor = torch.stack(tensors).to(self.device)
        batch = torch.cat([tensor, tensor.flip(-1)])
        feature = self.net.neck(self.net.backbone(batch)).mean((2, 3)).float()
        count = len(images)
        vectors = functional.normalize(feature[:count] + feature[count:], dim=1).cpu().numpy()
        if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) <= 0):
            raise ValueError("Invalid YOLO embeddings.")
        return vectors


class YoloSearchService(SearchService):
    def __init__(self, predictor, gallery, report):
        super().__init__(predictor, gallery)
        # Bound memory: cache small galleries, compute only top-100 distances for large ones.
        self._gallery_tensor = torch.from_numpy(gallery.embeddings.copy())
        self._distances = None
        if len(gallery.ids) <= 2048:
            self._distances = (2 - 2 * self._gallery_tensor @ self._gallery_tensor.T).clamp_(min=0)
        self.metrics_report = report
        self.ranking_method = "k-reciprocal top-100; displayed scores are cosine"

    @classmethod
    def load(cls, directory: Path, device: str = "cpu"):
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["schema_version"] != 1:
            raise ValueError("Unsupported YOLO bundle.")
        for name in ("checkpoint.pt", "base.pt", "gallery.npy", "ids.json", "metrics.json"):
            if sha256_file(directory / name) != manifest["sha256"][name]:
                raise ValueError(f"YOLO bundle checksum mismatch: {name}")
        predictor = YoloPredictor(directory, manifest, device)
        gallery = GalleryIndex(
            json.loads((directory / "ids.json").read_text()),
            np.load(directory / "gallery.npy", allow_pickle=False),
            sha256_file(directory / "manifest.json"),
        )
        if gallery.embeddings.shape[1] != 1280 or len(gallery.ids) < 10:
            raise ValueError("YOLO gallery must have 1280 dimensions and at least ten rows.")
        report = json.loads((directory / "metrics.json").read_text())
        service = cls(predictor, gallery, report)
        service.warmup()
        return service

    def search(self, image: Image.Image, bbox: BBox, top_k: int = 10) -> dict:
        if type(top_k) is not int or not 1 <= top_k <= 10:
            raise ValueError("top_k must be an integer in 1..10.")
        if not self._lock.acquire(blocking=False):
            raise SearchBusy("Search is busy. Retry shortly.")
        try:
            embedding = self.predictor.embed_image(image, bbox)
            matches, _ = self.rank_embedding(embedding, top_k)
            best_cosine = max(match["score"] for match in matches)
            accepted = best_cosine >= self.predictor.threshold
            return {
                "accepted": accepted,
                "threshold": self.predictor.threshold,
                "matches": matches if accepted else [],
                "model_version": self.predictor.model_version,
                "gallery_version": self.gallery.version,
                "decision_score": best_cosine,
                "ranking_method": self.ranking_method,
            }
        finally:
            self._lock.release()

    @torch.inference_mode()
    def rank_embedding(self, embedding: np.ndarray, top_k: int = 10):
        query = torch.from_numpy(np.array(embedding, dtype=np.float32, copy=True)).reshape(1, -1)
        cosine = query @ self._gallery_tensor.T
        count = min(100, len(self.gallery.ids))
        indices = cosine.topk(count, dim=1).indices
        distances = (2 - 2 * cosine).clamp(min=0).gather(1, indices)
        if self._distances is None:
            candidates = self._gallery_tensor[indices]
            neighbours = (2 - 2 * candidates @ candidates.transpose(1, 2)).clamp_(min=0)
        else:
            neighbours = self._distances[indices[:, :, None], indices[:, None, :]]
        reranked = (1 - _rerank_batch(distances, neighbours, min(10, count), 3, 0.5))[0].numpy()
        if not np.isfinite(reranked).all():
            raise ValueError("Non-finite re-ranking scores.")
        scores = np.full(len(self.gallery.ids), -np.inf, dtype=np.float32)
        scores[indices[0].numpy()] = reranked
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [
            {
                "rank": rank,
                "image_id": self.gallery.ids[index],
                "score": float(cosine[0, index]),
                "rerank_score": float(scores[index]),
            }
            for rank, index in enumerate(order, 1)
        ], float(cosine.max())
