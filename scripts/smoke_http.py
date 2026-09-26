"""Exercise a running API with one real query; print no images, IDs or embeddings."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:27812")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.requests <= 100 or not 1 <= args.concurrency <= 8:
        parser.error("Use 1..100 requests and 1..8 clients")
    with (args.data_dir / "test_query.csv").open(encoding="utf-8-sig") as stream:
        query = next(csv.DictReader(stream))
    with (args.run_dir / "submission/submission.csv").open(encoding="utf-8-sig") as stream:
        expected = next(row[1:] for row in csv.reader(stream) if row[0] == query["image_id"])
    boundary = uuid.uuid4().hex
    fields = {key: query[key] for key in ("x", "y", "w", "h")}
    fields["top_k"] = "10"
    parts = [
        (f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n'
         f'{value}\r\n').encode()
        for key, value in fields.items()
    ]
    parts.append(
        (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
         'filename="query.jpg"\r\nContent-Type: image/jpeg\r\n\r\n').encode()
        + (args.data_dir / "images" / f'{query["image_id"]}.jpg').read_bytes() + b"\r\n"
    )
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()

    def post(payload: bytes = body):
        request = urllib.request.Request(
            args.url.rstrip("/") + "/api/v1/search", data=payload,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        started = time.monotonic()
        try:
            response = urllib.request.urlopen(request, timeout=30)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            result = json.load(response)
            request_id = response.headers["x-request-id"]
            assert result["request_id"] == request_id
            return response.status, result, time.monotonic() - started, request_id

    status, result, _, _ = post()
    assert status == 200, result
    assert [row["image_id"] for row in result["matches"]] == expected[:10]
    assert result["accepted"] == (result["matches"][0]["score"] >= result["threshold"])
    bad = body.replace(
        f'name="x"\r\n\r\n{query["x"]}\r\n'.encode(), b'name="x"\r\n\r\n-1\r\n', 1
    )
    assert post(bad)[0] == 422
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        outcomes = list(pool.map(lambda _: post(), range(args.requests)))
    counts = Counter(row[0] for row in outcomes)
    assert counts[200] > 0 and set(counts) <= {200, 429}, counts
    assert len({row[3] for row in outcomes}) == len(outcomes)
    for code, result, _, _ in outcomes:
        if code == 200:
            assert [row["image_id"] for row in result["matches"]] == expected[:10]
    timings = sorted(row[2] * 1000 for row in outcomes if row[0] == 200)
    with urllib.request.urlopen(args.url.rstrip("/") + "/api/v1/ready", timeout=5) as response:
        assert response.status == 200
    print(json.dumps({
        "status": "ok", "requests": args.requests, "concurrency": args.concurrency,
        "http_status_counts": dict(counts), "top10_matches_saved_baseline": True,
        "success_latency_ms": {
            "median": round(statistics.median(timings), 2),
            "p95": round(timings[math.ceil(0.95 * len(timings)) - 1], 2),
        },
        "note": "Small smoke sample, not a capacity or quality benchmark.",
    }))


if __name__ == "__main__":
    main()
