"""Compare online adapter to source artifacts without emitting images or gallery IDs."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from street_falcon_reid.predictor import BBox
from street_falcon_reid.yolo import YoloSearchService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    service = YoloSearchService.load(args.bundle)
    with (args.data_dir / "test_query.csv").open() as stream:
        queries = list(csv.DictReader(stream))
    with (args.data_dir / "test_gallery.csv").open() as stream:
        gallery = list(csv.DictReader(stream))
    with (args.bundle / "expected.csv").open() as stream:
        expected = {row[0]: row[1:] for row in csv.reader(stream)}
    stored = np.load(args.bundle / "embeddings.npy", allow_pickle=False)
    indices = np.linspace(0, len(queries) - 1, 12, dtype=int)
    max_error, durations = 0.0, []
    exact_online = 0
    minimum_overlap = 10
    for index in indices:
        row = queries[index]
        bbox = BBox(*(int(row[key]) for key in ("x", "y", "w", "h")))
        with Image.open(args.data_dir / "images" / (row["image_id"] + ".jpg")) as source:
            image = source.convert("RGB")
            start = time.monotonic()
            vector = service.predictor.embed_image(image, bbox)
            max_error = max(max_error, float(np.max(np.abs(vector - stored[index]))))
            matches, _ = service.rank_embedding(vector)
            durations.append(time.monotonic() - start)
            actual = [m["image_id"] for m in matches]
            wanted = expected[row["image_id"]]
            saved = [m["image_id"] for m in service.rank_embedding(stored[index])[0]]
            assert saved == wanted
            assert actual[0] == wanted[0]
            overlap = len(set(actual) & set(wanted))
            assert overlap >= 9
            minimum_overlap = min(minimum_overlap, overlap)
            exact_online += int(actual == wanted)
    for index in (0, len(gallery) // 2, len(gallery) - 1):
        row = gallery[index]
        bbox = BBox(*(int(row[key]) for key in ("x", "y", "w", "h")))
        with Image.open(args.data_dir / "images" / (row["image_id"] + ".jpg")) as source:
            vector = service.predictor.embed_image(source.convert("RGB"), bbox)
        error = float(np.max(np.abs(vector - stored[len(queries) + index])))
        max_error = max(max_error, error)
    assert max_error < 0.0003, max_error
    print(
        json.dumps(
            {
                "status": "ok",
                "real_queries_checked": len(indices),
                "gallery_vectors_checked": 3,
                "saved_embedding_top10_matches_source": True,
                "online_exact_top10_count": exact_online,
                "online_top1_matches": len(indices),
                "minimum_top10_overlap": minimum_overlap,
                "max_embedding_abs_error": max_error,
                "median_seconds": float(np.median(durations)),
                "note": "Integration sample; not a quality or capacity benchmark.",
            }
        )
    )


if __name__ == "__main__":
    main()
