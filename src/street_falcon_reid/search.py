from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from .config import load_config
from .metrics import stable_rank
from .predictor import BBox, ImagePredictor
from .prepare import sha256_file
from .records import read_records

logger = logging.getLogger(__name__)


class SearchBusy(RuntimeError):
    pass


class GalleryIndex:
    def __init__(self, ids: list[str], embeddings: np.ndarray, version: str) -> None:
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Gallery IDs must be non-empty and unique.")
        if embeddings.ndim != 2 or embeddings.shape[0] != len(ids):
            raise ValueError("Gallery embeddings do not match gallery IDs.")
        if embeddings.shape[1] == 0 or embeddings.dtype != np.float32:
            raise ValueError("Gallery embeddings must be a non-empty float32 matrix.")
        if not np.isfinite(embeddings).all():
            raise ValueError("Gallery embeddings must be finite.")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("Gallery embeddings must be L2-normalized.")
        self.ids = tuple(ids)
        self.embeddings = embeddings / norms
        self.embeddings.setflags(write=False)
        self.version = version

    @classmethod
    def load(
        cls, data_dir: Path, run_dir: Path, predictor: ImagePredictor
    ) -> GalleryIndex:
        output = run_dir / "submission"
        metadata = json.loads((output / "run-metadata.json").read_text(encoding="utf-8"))
        verification = json.loads((output / "verification.json").read_text(encoding="utf-8"))
        if metadata["checkpoint_sha256"] != predictor.model_version:
            raise ValueError("Gallery and model checkpoint do not match.")
        if metadata["embedding_dim"] != predictor.embedding_dim:
            raise ValueError("Gallery and model embedding dimensions do not match.")
        if not np.isclose(metadata["open_set_threshold"], predictor.threshold, rtol=0, atol=1e-8):
            raise ValueError("Gallery and model thresholds do not match.")
        config = load_config(run_dir / "config.toml")
        for key in ("image_height", "image_width", "crop_margin"):
            if config["data"][key] != predictor.data_config[key]:
                raise ValueError("Gallery and model preprocessing do not match.")
        gallery_path = data_dir / "test_gallery.csv"
        gallery = read_records(gallery_path, require_labels=False)
        query_count = metadata["query_count"]
        if type(query_count) is not int or query_count < 0:
            raise ValueError("Invalid query count.")
        if metadata["gallery_count"] != len(gallery):
            raise ValueError("Gallery count does not match metadata.")
        gallery_checksum = metadata.get("gallery_csv_sha256")
        if gallery_checksum is not None:
            if gallery_checksum != sha256_file(gallery_path):
                raise ValueError("Gallery CSV checksum does not match.")
        else:
            logger.warning(
                "Legacy run has no gallery CSV checksum; use the original dataset CSV order."
            )
        path = output / "embeddings.npy"
        checksum = sha256_file(path)
        if checksum != verification["sha256"]["embeddings.npy"]:
            raise ValueError("Embeddings checksum does not match.")
        values = np.load(path, allow_pickle=False)
        if not isinstance(values, np.ndarray):
            values.close()
            raise ValueError("Expected an embeddings array.")
        expected_shape = (query_count + len(gallery), predictor.embedding_dim)
        if values.shape != expected_shape or list(values.shape) != verification["embedding_shape"]:
            raise ValueError("Combined query/gallery embedding shape does not match.")
        return cls(
            [record.image_id for record in gallery],
            values[query_count:].copy(),
            f"{run_dir.name}:{checksum}",
        )

    def search(self, embedding: np.ndarray, top_k: int = 10) -> list[dict[str, object]]:
        if type(top_k) is not int or not 1 <= top_k <= 10:
            raise ValueError("top_k must be an integer between 1 and 10.")
        query = np.asarray(embedding, dtype=np.float32)
        if query.shape != (self.embeddings.shape[1],) or not np.isfinite(query).all():
            raise ValueError("Query embedding has an invalid shape or values.")
        norm = np.linalg.norm(query)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Query embedding must have a finite, non-zero norm.")
        scores = (query / norm)[None, :] @ self.embeddings.T
        order = stable_rank(scores)[0, :min(top_k, len(self.ids))]
        return [
            {"rank": rank, "image_id": self.ids[index], "score": float(scores[0, index])}
            for rank, index in enumerate(order, start=1)
        ]


class SearchService:
    def __init__(self, predictor: ImagePredictor, gallery: GalleryIndex) -> None:
        self.predictor = predictor
        self.gallery = gallery
        self._lock = threading.Lock()

    @classmethod
    def load(cls, data_dir: Path, run_dir: Path, device: str = "cpu") -> SearchService:
        predictor = ImagePredictor(run_dir / "checkpoint-best.pt", device)
        gallery = GalleryIndex.load(data_dir, run_dir, predictor)
        return cls(predictor, gallery)

    def search(self, image: Image.Image, bbox: BBox, top_k: int = 10) -> dict[str, object]:
        if not self._lock.acquire(blocking=False):
            raise SearchBusy("Search is busy. Retry shortly.")
        try:
            embedding = self.predictor.embed_image(image, bbox)
            matches = self.gallery.search(embedding, top_k)
            return {
                "accepted": matches[0]["score"] >= self.predictor.threshold,
                "threshold": self.predictor.threshold,
                "matches": matches,
                "model_version": self.predictor.model_version,
                "gallery_version": self.gallery.version,
            }
        finally:
            self._lock.release()
