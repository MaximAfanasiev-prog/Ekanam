"""Build a demo gallery from supplied CSVs without modifying source data or the active bundle."""

import argparse
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from street_falcon_reid.frames import ALGORITHM, frame_digest
from street_falcon_reid.predictor import BBox
from street_falcon_reid.prepare import sha256_file
from street_falcon_reid.yolo import YoloSearchService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 32:
        parser.error("batch-size must be 1..32")
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    thumbnails = args.output / "thumbnails"
    thumbnails.mkdir(mode=0o700)
    service = YoloSearchService.load(args.source_bundle)
    rows, counts, source_hashes = [], {}, {}
    for split in ("train", "test_query", "test_gallery"):
        path = args.data_dir / (split + ".csv")
        with path.open(encoding="utf-8-sig") as stream:
            part = list(csv.DictReader(stream))
        rows.extend(part)
        counts[split] = len(part)
        source_hashes[path.name] = sha256_file(path)
    ids = [row["image_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Source CSVs contain duplicate IDs.")
    vectors = np.lib.format.open_memmap(
        args.output / "gallery.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), service.predictor.embedding_dim),
    )
    images_dir = (args.data_dir / "images").resolve()
    started = time.monotonic()
    max_batch_error = 0.0
    frame_hashes = []
    for start in range(0, len(rows), args.batch_size):
        images, boxes = [], []
        batch_rows = rows[start : start + args.batch_size]
        try:
            for row in batch_rows:
                path = (images_dir / (row["image_id"] + ".jpg")).resolve()
                if path.parent != images_dir:
                    raise ValueError("Invalid image path.")
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
                frame_hashes.append(frame_digest(images[-1]))
                boxes.append(BBox(*(int(row[key]) for key in ("x", "y", "w", "h"))))
            batch = service.predictor.embed_images(images, boxes)
            if start == 0:
                for i in range(min(3, len(images))):
                    single = service.predictor.embed_image(images[i], boxes[i])
                    max_batch_error = max(max_batch_error, float(np.max(np.abs(single - batch[i]))))
                if max_batch_error > 1e-4:
                    raise ValueError("Batch and single-image extraction disagree.")
            vectors[start : start + len(images)] = batch
            for row, image, box in zip(batch_rows, images, boxes, strict=True):
                with image.crop((box.x, box.y, box.x + box.w, box.y + box.h)) as crop:
                    crop.thumbnail((640, 480), Image.Resampling.LANCZOS)
                    name = hashlib.sha256(row["image_id"].encode()).hexdigest() + ".jpg"
                    crop.save(thumbnails / name, quality=88)
        finally:
            for image in images:
                image.close()
        if start % (args.batch_size * 16) == 0:
            vectors.flush()
            print(
                json.dumps(
                    {
                        "processed": start + len(batch_rows),
                        "total": len(rows),
                        "elapsed_s": round(time.monotonic() - started, 1),
                    }
                ),
                flush=True,
            )
    vectors.flush()
    if not np.isfinite(vectors).all() or not np.allclose(
        np.linalg.norm(vectors, axis=1), 1, atol=1e-3
    ):
        raise ValueError("Invalid gallery vectors.")
    del vectors
    (args.output / "ids.json").write_text(json.dumps(ids))
    (args.output / "frames.json").write_text(json.dumps({
        "schema_version": 1, "algorithm": ALGORITHM, "ids": ids, "hashes": frame_hashes,
    }))
    for filename in ("checkpoint.pt", "base.pt"):
        shutil.copyfile(args.source_bundle / filename, args.output / filename)
    report = service.metrics_report.copy()
    report["gallery_usage"] = {
        "mode": "demo_all_splits",
        "count": len(rows),
        "splits": counts,
        "includes_training_images": True,
        "includes_test_queries": True,
        "evaluated_on_this_gallery": False,
        "self_matches_possible": True,
        "exact_same_frame_excluded": True,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2))
    manifest = json.loads((args.source_bundle / "manifest.json").read_text())
    manifest = {
        "schema_version": 1,
        "epoch": manifest["epoch"],
        "threshold": manifest["threshold"],
        "source_revision": manifest["source_revision"],
        "parent_manifest_sha256": sha256_file(args.source_bundle / "manifest.json"),
        "source_csv_sha256": source_hashes,
        "gallery_usage": report["gallery_usage"],
        "threshold_source": "inherited demo threshold; expanded gallery not calibrated",
        "sha256": {
            name: sha256_file(args.output / name)
            for name in (
                "checkpoint.pt", "base.pt", "gallery.npy", "ids.json", "metrics.json", "frames.json"
            )
        },
    }
    evidence = {
        "count": len(rows),
        "seconds": round(time.monotonic() - started, 1),
        "batch_size": args.batch_size,
        "max_batch_single_error": max_batch_error,
    }
    (args.output / "build-evidence.json").write_text(json.dumps(evidence, indent=2))
    # Publish readiness last: incomplete directories cannot be loaded by the service.
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"status": "complete", **evidence}), flush=True)


if __name__ == "__main__":
    main()
