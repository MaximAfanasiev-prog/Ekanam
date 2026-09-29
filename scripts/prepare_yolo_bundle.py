"""Prepare a checked gallery from the team's final 60-epoch YOLO artifacts, outside Git."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.directory
    if (root / "manifest.json").exists():
        raise ValueError("Bundle already prepared; use a new directory.")
    meta = json.loads((root / "run_meta.json").read_text())
    checkpoint = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=True)
    if (
        meta["trained_epochs"] != 60
        or checkpoint["meta"]["epoch"] != 60
        or not checkpoint["meta"]["config"]["full_train"]
        or meta["mode"] != "squash"
        or meta["feature"] != "feat"
        or meta["flip"] is not True
        or meta["margin"] != 0
        or meta["rerank"] != {"k1": 10, "k2": 3, "lam": 0.5, "top_k": 100}
    ):
        raise ValueError("Expected final 60-epoch squash/feat/flip recipe.")
    with (args.data_dir / "test_gallery.csv").open() as stream:
        ids = [row["image_id"] for row in csv.DictReader(stream)]
    values = np.load(root / "embeddings.npy", allow_pickle=False)
    if (
        values.dtype != np.float32
        or values.shape != (meta["n_query"] + len(ids), 1280)
        or len(ids) != meta["n_gallery"]
        or len(set(ids)) != len(ids)
        or not np.isfinite(values).all()
        or not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-3)
    ):
        raise ValueError("Invalid source embeddings.")
    np.save(root / "gallery.npy", values[meta["n_query"] :])
    (root / "ids.json").write_text(json.dumps(ids))
    names = ("checkpoint.pt", "base.pt", "gallery.npy", "ids.json", "metrics.json")
    manifest = {
        "schema_version": 1,
        "epoch": 60,
        "threshold": meta["threshold"],
        "source_revision": (root / "source_revision.txt").read_text().strip(),
        "sha256": {name: digest(root / name) for name in names},
        "source_embeddings_sha256": digest(root / "embeddings.npy"),
        "gallery_csv_sha256": digest(args.data_dir / "test_gallery.csv"),
        "query_csv_sha256": digest(args.data_dir / "test_query.csv"),
        "threshold_source": "fixed test-derived quantile; online calibration pending",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Prepared YOLO bundle:", len(ids), "gallery rows.")


if __name__ == "__main__":
    main()
