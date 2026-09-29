"""Exact full-frame matching, independent of filename, metadata and target bbox."""

from __future__ import annotations

import hashlib
import json
import re
import struct
from pathlib import Path

from PIL import Image

ALGORITHM = "sha256-rgb-pixels-v1"


def frame_digest(image: Image.Image) -> str:
    """Hash dimensions and decoded RGB pixels, without resizing or EXIF rotation."""
    digest = hashlib.sha256()
    digest.update(ALGORITHM.encode("ascii"))
    digest.update(struct.pack(">II", *image.size))
    with image.convert("RGB") as rgb:
        digest.update(rgb.tobytes())
    return digest.hexdigest()


class FrameIndex:
    def __init__(self, ids, hashes):
        if len(hashes) != len(ids) or any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in hashes
        ):
            raise ValueError("Invalid frame fingerprints.")
        groups = {}
        for position, value in enumerate(hashes):
            groups.setdefault(value, []).append(position)
        self.groups = {key: tuple(value) for key, value in groups.items()}

    @classmethod
    def load(cls, path: Path, ids):
        data = json.loads(path.read_text())
        if (
            data.get("schema_version") != 1
            or data.get("algorithm") != ALGORITHM
            or data.get("ids") != list(ids)
        ):
            raise ValueError("Frame index does not match gallery IDs or algorithm.")
        return cls(ids, data["hashes"])

    def matching(self, image: Image.Image) -> tuple[int, ...]:
        return self.groups.get(frame_digest(image), ())
