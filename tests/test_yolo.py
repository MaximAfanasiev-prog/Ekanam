import io
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from street_falcon_reid.api import create_app
from street_falcon_reid.predictor import BBox
from street_falcon_reid.search import GalleryIndex
from street_falcon_reid.yolo import YoloSearchService


def make_service(identical=False):
    values = np.random.default_rng(9).normal(size=(12, 1280)).astype(np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    if identical:
        values[:] = values[0]
    gallery = GalleryIndex([f"g{i}" for i in range(12)], values, "test")
    predictor = SimpleNamespace(
        model_version="fixture",
        threshold=0.8,
        device="cpu",
        embedding_dim=1280,
        data_config={"crop_margin": 0},
        embed_image=lambda image, box: values[0],
    )
    return YoloSearchService(predictor, gallery, {"scope": "test-only"})


@pytest.mark.parametrize("identical", [False, True])
def test_yolo_rerank_returns_finite_unique_ranked_results(identical):
    service = make_service(identical)
    with Image.new("RGB", (10, 10)) as image:
        result = service.search(image, BBox(0, 0, 10, 10))
    assert result["accepted"]
    assert len({m["image_id"] for m in result["matches"]}) == 10
    assert [m["rank"] for m in result["matches"]] == list(range(1, 11))
    assert all(np.isfinite(m["rerank_score"]) for m in result["matches"])
    assert result["decision_score"] == pytest.approx(1, abs=1e-6)


@pytest.mark.parametrize("top_k", [1, 10])
@pytest.mark.parametrize("best_returned", [0.7999, 0.8, 0.9])
def test_yolo_threshold_uses_only_returned_candidates(monkeypatch, top_k, best_returned):
    service = make_service()
    matches = [
        {"rank": i + 1, "image_id": f"g{i}", "score": 0.7, "rerank_score": 0.99 - i / 100}
        for i in range(top_k)
    ]
    # The best cosine can be last after re-ranking; the global best is outside this top.
    matches[-1]["score"] = best_returned
    monkeypatch.setattr(service, "rank_embedding", lambda vector, k: (matches, 0.99))
    stream = io.BytesIO()
    Image.new("RGB", (10, 10)).save(stream, format="PNG")
    with TestClient(create_app(lambda: service)) as client:
        response = client.post(
            "/api/v1/search",
            data={"x": 0, "y": 0, "w": 10, "h": 10, "top_k": top_k},
            files={"image": ("test.png", stream.getvalue(), "image/png")},
        )
    assert response.status_code == 200
    result = response.json()
    accepted = best_returned >= service.predictor.threshold
    assert result["accepted"] is accepted
    assert result["matches"] == (matches if accepted else [])
    assert result["decision_score"] == best_returned


def test_yolo_metrics_and_search_contract():
    service = make_service()
    with TestClient(create_app(lambda: service)) as client:
        response = client.get("/api/v1/metrics")
        assert response.json() == {"model_version": "fixture", "report": {"scope": "test-only"}}
        stream = io.BytesIO()
        Image.new("RGB", (10, 10)).save(stream, format="PNG")
        response = client.post(
            "/api/v1/search",
            data={"x": 0, "y": 0, "w": 10, "h": 10},
            files={"image": ("test.png", stream.getvalue(), "image/png")},
        )
        assert response.status_code == 200
        assert response.json()["decision_score"] is not None
        assert response.json()["matches"][0]["rerank_score"] is not None
        assert client.get("/api/v1/info").json()["ranking_method"].startswith("k-reciprocal")


def test_large_gallery_keeps_only_candidate_distances():
    values = np.random.default_rng(33).normal(size=(2049, 16)).astype(np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    gallery = GalleryIndex([f"g{i}" for i in range(len(values))], values, "large")
    service = YoloSearchService(SimpleNamespace(), gallery, {})
    assert service._distances is None
    matches, best = service.rank_embedding(values[0])
    assert len(matches) == 10
    assert best == pytest.approx(1, abs=1e-6)
    assert all(np.isfinite(match["rerank_score"]) for match in matches)


def test_candidate_distances_preserve_cached_ranking():
    service = make_service()
    vector = service.gallery.embeddings[0]
    cached, cached_score = service.rank_embedding(vector)
    service._distances = None
    bounded, bounded_score = service.rank_embedding(vector)
    assert [m["image_id"] for m in bounded] == [m["image_id"] for m in cached]
    assert bounded_score == pytest.approx(cached_score)
    assert [m["rerank_score"] for m in bounded] == pytest.approx(
        [m["rerank_score"] for m in cached], abs=1e-6
    )
