"""Validate a completed demonstration bundle and a running candidate API."""

import argparse
import csv
import io
import json
import time
import urllib.request
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from street_falcon_reid.frames import FrameIndex
from street_falcon_reid.prepare import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--thumbnail-dir", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:27818")
    args = parser.parse_args()
    manifest = json.loads((args.directory / "manifest.json").read_text())
    ids = json.loads((args.directory / "ids.json").read_text())
    assert len(ids) == len(set(ids)) == 11416
    for name, digest in manifest["sha256"].items():
        assert sha256_file(args.directory / name) == digest
    vectors = np.load(args.directory / "gallery.npy", mmap_mode="r", allow_pickle=False)
    assert vectors.shape == (11416, 1280)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3)
    thumbnail_dir = args.thumbnail_dir or args.directory / "thumbnails"
    assert len(list(thumbnail_dir.glob("*.jpg"))) == len(ids)
    frames = (
        FrameIndex.load(args.directory / "frames.json", ids)
        if "frames.json" in manifest["sha256"] else None
    )
    verified_thumbnails = 0
    for filename, digest in manifest["source_csv_sha256"].items():
        assert sha256_file(args.data_dir / filename) == digest
    latencies = []
    for split in ("train", "test_query", "test_gallery"):
        with (args.data_dir / (split + ".csv")).open() as stream:
            row = next(csv.DictReader(stream))
        boundary = uuid.uuid4().hex
        fields = {key: row[key] for key in ("x", "y", "w", "h")}
        fields["top_k"] = "10"
        parts = [
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'
            ).encode()
            for key, value in fields.items()
        ]
        parts.append(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
                'filename="query.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'
            ).encode()
            + (args.data_dir / "images" / (row["image_id"] + ".jpg")).read_bytes()
            + b"\r\n"
        )
        body = b"".join(parts) + f"--{boundary}--\r\n".encode()
        request = urllib.request.Request(
            args.url + "/api/v1/search",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
        latencies.append(round(time.monotonic() - started, 4))
        if frames:
            with Image.open(args.data_dir / "images" / (row["image_id"] + ".jpg")) as image:
                excluded = frames.matching(image)
            assert excluded
            assert result["excluded_same_frame"] == len(excluded)
            excluded_ids = {ids[i] for i in excluded}
            assert not excluded_ids & {item["image_id"] for item in result["matches"]}
            if result["accepted"]:
                assert len(result["matches"]) == 10
                assert result["decision_score"] >= result["threshold"]
            else:
                assert result["matches"] == []
                assert result["decision_score"] < result["threshold"]
        else:
            assert len(result["matches"]) == 10
            assert row["image_id"] in {item["image_id"] for item in result["matches"]}
            assert result["decision_score"] > 0.999
        for match in result["matches"]:
            assert match["image_id"] in ids
            path = args.url + "/api/v1/gallery/" + match["image_id"] + "/thumbnail"
            with urllib.request.urlopen(path, timeout=5) as response:
                with Image.open(io.BytesIO(response.read())) as image:
                    image.verify()
                    verified_thumbnails += 1
    with urllib.request.urlopen(args.url + "/api/v1/info", timeout=5) as response:
        assert json.load(response)["gallery_count"] == 11416
    with urllib.request.urlopen(args.url + "/api/v1/metrics", timeout=5) as response:
        usage = json.load(response)["report"]["gallery_usage"]
        assert usage["count"] == 11416 and usage["evaluated_on_this_gallery"] is False
    print(
        json.dumps(
            {
                "status": "ok",
                "count": len(ids),
                "splits_tested": 3,
                "thumbnails_verified_over_http": verified_thumbnails,
                "same_frame_filter": bool(frames),
                "query_seconds": latencies,
            }
        )
    )


if __name__ == "__main__":
    main()
