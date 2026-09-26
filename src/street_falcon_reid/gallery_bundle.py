"""Create an immutable gallery package outside Git from a verified baseline run."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .predictor import ImagePredictor
from .prepare import sha256_file
from .search import GalleryIndex

PREPROCESS_KEYS = ("image_height", "image_width", "crop_margin")


def load_bundle(bundle: Path, predictor: ImagePredictor) -> GalleryIndex:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported gallery manifest schema.")
    if manifest["checkpoint_sha256"] != predictor.model_version:
        raise ValueError("Gallery bundle checkpoint does not match.")
    if manifest["embedding_dim"] != predictor.embedding_dim:
        raise ValueError("Gallery bundle embedding dimension does not match.")
    if not np.isclose(manifest["threshold"], predictor.threshold, rtol=0, atol=1e-8):
        raise ValueError("Gallery bundle threshold does not match.")
    if manifest["preprocessing"] != {key: predictor.data_config[key] for key in PREPROCESS_KEYS}:
        raise ValueError("Gallery bundle preprocessing does not match.")
    for filename in ("ids.json", "gallery.npy"):
        if sha256_file(bundle / filename) != manifest["sha256"][filename]:
            raise ValueError(f"Gallery bundle checksum failed: {filename}")
    ids = json.loads((bundle / "ids.json").read_text(encoding="utf-8"))
    if not isinstance(ids, list) or any(not isinstance(item, str) or not item for item in ids):
        raise ValueError("Gallery IDs must be non-empty strings.")
    if manifest["gallery_count"] != len(ids):
        raise ValueError("Gallery bundle count does not match.")
    embeddings = np.load(bundle / "gallery.npy", allow_pickle=False)
    if not isinstance(embeddings, np.ndarray):
        embeddings.close()
        raise ValueError("Expected a gallery array.")
    if embeddings.shape != (len(ids), predictor.embedding_dim):
        raise ValueError("Gallery bundle shape does not match.")
    return GalleryIndex(ids, embeddings, sha256_file(bundle / "manifest.json"))


def export_bundle(
    data_dir: Path, run_dir: Path, output: Path, *, allow_legacy: bool = False
) -> Path:
    # Explicit operator action: legacy CSV order cannot be proved from old metadata.
    metadata = json.loads((run_dir / "submission/run-metadata.json").read_text(encoding="utf-8"))
    verified_csv = "gallery_csv_sha256" in metadata
    if not verified_csv and not allow_legacy:
        raise ValueError(
            "Legacy run: explicitly pass --allow-legacy with the original gallery CSV."
        )
    if output.exists():
        raise FileExistsError("Bundle output already exists; use a new version directory.")
    predictor = ImagePredictor(run_dir / "checkpoint-best.pt", "cpu")
    gallery = GalleryIndex.load(data_dir, run_dir, predictor)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gallery-export-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        (stage / "ids.json").write_text(
            json.dumps(gallery.ids, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        np.save(stage / "gallery.npy", gallery.embeddings, allow_pickle=False)
        manifest = {
            "schema_version": 1,
            "checkpoint_sha256": predictor.model_version,
            "embedding_dim": predictor.embedding_dim,
            "gallery_count": len(gallery.ids),
            "threshold": predictor.threshold,
            "preprocessing": {key: predictor.data_config[key] for key in PREPROCESS_KEYS},
            "source_run": run_dir.name,
            "source_gallery_csv_sha256": sha256_file(data_dir / "test_gallery.csv"),
            "source_csv_checksum_verified": verified_csv,
            "sha256": {name: sha256_file(stage / name) for name in ("ids.json", "gallery.npy")},
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        load_bundle(stage, predictor)
        # mkdir is exclusive, so an existing or concurrent export cannot be overwritten.
        output.mkdir(mode=0o700)
        # The manifest is published last: partial output is not loadable.
        for name in ("ids.json", "gallery.npy", "manifest.json"):
            os.replace(stage / name, output / name)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-legacy", action="store_true")
    args = parser.parse_args()
    result = export_bundle(args.data_dir, args.run_dir, args.output, allow_legacy=args.allow_legacy)
    print(f"Gallery bundle ready: {result}")


if __name__ == "__main__":
    main()
