"""Read-only online inference check against an existing baseline run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from street_falcon_reid.api import create_app
from street_falcon_reid.dataset import VehicleDataset
from street_falcon_reid.metrics import cosine_similarity, stable_rank
from street_falcon_reid.predictor import BBox
from street_falcon_reid.records import read_records
from street_falcon_reid.search import SearchService
from street_falcon_reid.train import extract_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    query = read_records(args.data_dir / "test_query.csv", require_labels=False)[0]
    service = SearchService.load(args.data_dir, args.run_dir, "cpu")
    dataset = VehicleDataset(
        [query], args.data_dir, service.predictor.transform,
        crop_margin=float(service.predictor.data_config["crop_margin"]),
    )
    reference, _ = extract_embeddings(
        service.predictor.model, dataset, batch_size=1, workers=0,
        device=service.predictor.device,
    )
    with Image.open(args.data_dir / "images" / f"{query.image_id}.jpg") as image:
        online = service.predictor.embed_image(image, BBox(query.x, query.y, query.w, query.h))
    np.testing.assert_allclose(online, reference[0], atol=1e-6, rtol=1e-6)
    combined = np.load(args.run_dir / "submission/embeddings.npy", allow_pickle=False)
    scores = cosine_similarity(combined[:1], service.gallery.embeddings)
    order = stable_rank(scores)[0, :10]
    with TestClient(create_app(lambda: service)) as client:
        with (args.data_dir / "images" / f"{query.image_id}.jpg").open("rb") as image:
            response = client.post(
                "/api/v1/search",
                files={"image": ("query.jpg", image, "image/jpeg")},
                data={"x": query.x, "y": query.y, "w": query.w, "h": query.h, "top_k": 10},
            )
        response.raise_for_status()
        result = response.json()
        assert [row["image_id"] for row in result["matches"]] == [
            service.gallery.ids[index] for index in order
        ], "Online top-k differs from saved baseline"
        np.testing.assert_allclose(
            [row["score"] for row in result["matches"]], scores[0, order], atol=1e-3, rtol=0
        )
        assert result["accepted"] == bool(scores[0, order[0]] >= service.predictor.threshold)
        assert client.get("/api/v1/health").status_code == 200
    print(json.dumps({
        "status": "ok", "device": "cpu", "gallery_count": len(service.gallery.ids),
        "top_k_matches_baseline": True, "saved_gpu_scores_tolerance": 1e-3,
        "same_cpu_embedding_tolerance": 1e-6,
        "max_saved_score_difference": float(np.max(np.abs(
            np.array([row["score"] for row in result["matches"]]) - scores[0, order]
        ))),
    }))


if __name__ == "__main__":
    main()
