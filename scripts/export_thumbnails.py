"""Export gallery-only preview crops; keep output outside Git and container images."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--gallery-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    ids = json.loads((args.gallery_dir / "ids.json").read_text())
    with (args.data_dir / "test_gallery.csv").open(encoding="utf-8-sig") as stream:
        records = {row["image_id"]: row for row in csv.DictReader(stream)}
    if not set(ids) <= records.keys():
        raise ValueError("Gallery IDs missing from source CSV.")
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    images = (args.data_dir / "images").resolve()
    for image_id in ids:
        path = (images / (image_id + ".jpg")).resolve()
        if path.parent != images:
            raise ValueError("Invalid source image path.")
        row = records[image_id]
        x, y, w, h = (int(row[key]) for key in ("x", "y", "w", "h"))
        with Image.open(path) as source:
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                raise ValueError("Invalid gallery bbox.")
            if x + w > source.width or y + h > source.height:
                raise ValueError("Gallery bbox outside image.")
            with source.crop((x, y, x + w, y + h)).convert("RGB") as crop:
                crop.thumbnail((640, 480), Image.Resampling.LANCZOS)
                filename = hashlib.sha256(image_id.encode()).hexdigest() + ".jpg"
                crop.save(args.output_dir / filename, quality=88)
    print(f"Exported {len(ids)} gallery thumbnails.")


if __name__ == "__main__":
    main()
