import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, PngImagePlugin

from street_falcon_reid.api import create_app
from street_falcon_reid.frames import ALGORITHM, FrameIndex, frame_digest
from street_falcon_reid.predictor import BBox
from street_falcon_reid.search import GalleryIndex
from street_falcon_reid.yolo import YoloSearchService


def fixture_service(count=12, excluded=2):
    values = np.random.default_rng(42).normal(size=(count, 16)).astype(np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    ids = [f"g{i}" for i in range(count)]
    gallery = GalleryIndex(ids, values, "fixture")
    with Image.new("RGB", (16, 12), "red") as image:
        same = frame_digest(image)
    with Image.new("RGB", (16, 12), "blue") as image:
        other = frame_digest(image)
    index = FrameIndex(ids, [same] * excluded + [other] * (count - excluded))
    predictor = SimpleNamespace(
        embed_image=lambda image, box: values[0], threshold=-1.0,
        model_version="fixture", device="cpu", embedding_dim=16,
        data_config={"crop_margin": 0},
    )
    return YoloSearchService(predictor, gallery, {}, index)


def test_full_frame_hash_ignores_encoding_metadata_but_not_pixels_or_dimensions():
    with Image.new("RGB", (16, 12), "red") as original:
        stream = io.BytesIO()
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("Description", "different filename and metadata")
        original.save(stream, format="PNG", pnginfo=metadata)
        with Image.open(io.BytesIO(stream.getvalue())) as decoded:
            assert frame_digest(original) == frame_digest(decoded)
        with original.copy() as changed:
            changed.putpixel((15, 11), (0, 0, 0))
            assert frame_digest(original) != frame_digest(changed)
        with Image.new("RGB", (12, 16), "red") as reshaped:
            assert frame_digest(original) != frame_digest(reshaped)


@pytest.mark.parametrize("excluded", [0, 2, 110, 118, 119, 120])
def test_exclusion_precedes_candidate_selection_and_reranking(excluded):
    service = fixture_service(120, excluded)
    with Image.new("RGB", (16, 12), "red") as image:
        result = service.search(image, BBox(0, 0, 8, 8))
    matches = result["matches"]
    assert result["excluded_same_frame"] == excluded
    assert len(matches) == min(10, 120 - excluded)
    assert not set(service.gallery.ids[:excluded]) & {m["image_id"] for m in matches}
    assert [m["rank"] for m in matches] == list(range(1, len(matches) + 1))
    if excluded == 120:
        assert result["accepted"] is False
        assert result["decision_score"] is None
        return
    # Removing rows must behave exactly like a gallery containing only eligible rows.
    smaller = GalleryIndex(
        list(service.gallery.ids[excluded:]), service.gallery.embeddings[excluded:].copy(), "subset"
    )
    comparison = YoloSearchService(service.predictor, smaller, {})
    # Preserve the exact floats: constructing GalleryIndex normalizes a second time.
    comparison._gallery_tensor = service._gallery_tensor[excluded:]
    comparison._distances = service._distances[excluded:, excluded:]
    expected, _ = comparison.rank_embedding(service.predictor.embed_image(None, None))
    assert [m["image_id"] for m in matches] == [m["image_id"] for m in expected]
    assert [m["rerank_score"] for m in matches] == pytest.approx(
        [m["rerank_score"] for m in expected], abs=1e-6
    )


@pytest.mark.parametrize("all_same", [False, True])
def test_http_excludes_all_targets_from_renamed_frame_for_any_bbox(all_same):
    service = fixture_service(12, 12 if all_same else 2)
    stream = io.BytesIO()
    Image.new("RGB", (16, 12), "red").save(stream, format="PNG")
    with TestClient(create_app(lambda: service)) as client:
        assert client.get("/api/v1/info").json()["same_frame_filter"] == "exact_rgb"
        for name, x in [("renamed.png", 0), ("unrelated-name.png", 8)]:
            response = client.post(
                "/api/v1/search",
                files={"image": (name, stream.getvalue(), "image/png")},
                data={"x": x, "y": 0, "w": 8, "h": 8},
            )
            assert response.status_code == 200
            result = response.json()
            assert result["excluded_same_frame"] == (12 if all_same else 2)
            assert not {"g0", "g1"} & {m["image_id"] for m in result["matches"]}
            assert len(result["matches"]) == (0 if all_same else 10)
            if all_same:
                assert result["decision_score"] is None
                assert result["accepted"] is False


def test_remaining_candidates_still_have_to_pass_threshold():
    service = fixture_service()
    service.predictor.threshold = 0.9999
    with Image.new("RGB", (16, 12), "red") as image:
        result = service.search(image, BBox(0, 0, 8, 8))
    assert result["excluded_same_frame"] == 2
    assert result["matches"] == []
    assert result["accepted"] is False
    assert result["decision_score"] < service.predictor.threshold


@pytest.mark.parametrize("fault", ["none", "order", "missing", "hash", "algorithm"])
def test_frame_index_requires_exact_gallery_mapping(tmp_path, fault):
    data = {
        "schema_version": 1, "algorithm": ALGORITHM,
        "ids": ["a", "b"], "hashes": ["a" * 64, "b" * 64],
    }
    if fault == "order":
        data["ids"].reverse()
    elif fault == "missing":
        data["hashes"].pop()
    elif fault == "hash":
        data["hashes"][0] = "not-a-hash"
    elif fault == "algorithm":
        data["algorithm"] = "unknown"
    path = tmp_path / "frames.json"
    path.write_text(json.dumps(data))
    if fault == "none":
        FrameIndex.load(path, ["a", "b"])
    else:
        with pytest.raises(ValueError):
            FrameIndex.load(path, ["a", "b"])
