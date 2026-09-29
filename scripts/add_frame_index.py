"""Create a new YOLO bundle with exact-frame exclusion; never mutate the source."""

import argparse
import json
import shutil
import time
from pathlib import Path

from PIL import Image

from street_falcon_reid.frames import ALGORITHM, FrameIndex, frame_digest
from street_falcon_reid.prepare import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.source_bundle / "manifest.json").read_text())
    for name in ("checkpoint.pt", "base.pt", "gallery.npy", "ids.json", "metrics.json"):
        if sha256_file(args.source_bundle / name) != manifest["sha256"][name]:
            raise ValueError(f"Source checksum mismatch: {name}")
    for name, digest in manifest.get("source_csv_sha256", {}).items():
        if name not in {"train.csv", "test_query.csv", "test_gallery.csv"}:
            raise ValueError("Unexpected source CSV.")
        if sha256_file(args.data_dir / name) != digest:
            raise ValueError(f"Source CSV mismatch: {name}")
    ids = json.loads((args.source_bundle / "ids.json").read_text())
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    images = (args.data_dir / "images").resolve()
    hashes = []
    started = time.monotonic()
    for i, image_id in enumerate(ids, 1):
        path = (images / (image_id + ".jpg")).resolve()
        if path.parent != images:
            raise ValueError("Invalid image path.")
        with Image.open(path) as image:
            hashes.append(frame_digest(image))
        if i % 500 == 0:
            print(json.dumps({"processed": i, "total": len(ids)}), flush=True)
    frames = {
        "schema_version": 1, "algorithm": ALGORITHM, "ids": ids, "hashes": hashes,
    }
    (args.output / "frames.json").write_text(json.dumps(frames))
    FrameIndex.load(args.output / "frames.json", ids)
    for name in ("checkpoint.pt", "base.pt", "gallery.npy", "ids.json"):
        shutil.copyfile(args.source_bundle / name, args.output / name)
    report = json.loads((args.source_bundle / "metrics.json").read_text())
    if "gallery_usage" in report:
        report["gallery_usage"]["exact_same_frame_excluded"] = True
        # Transformed copies can still occur; avoid claiming all self-matches are impossible.
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2))
    manifest["parent_manifest_sha256"] = sha256_file(args.source_bundle / "manifest.json")
    manifest["gallery_usage"] = report.get("gallery_usage", {})
    for name in ("frames.json", "metrics.json"):
        manifest["sha256"][name] = sha256_file(args.output / name)
    evidence = {
        "count": len(ids), "unique_frames": len(set(hashes)),
        "duplicate_frame_entries": len(ids) - len(set(hashes)),
        "seconds": round(time.monotonic() - started, 1),
    }
    (args.output / "frame-index-evidence.json").write_text(json.dumps(evidence, indent=2))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"status": "complete", **evidence}), flush=True)


if __name__ == "__main__":
    main()
