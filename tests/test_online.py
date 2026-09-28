from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from street_falcon_reid.api import create_app
from street_falcon_reid.config import load_config
from street_falcon_reid.dataset import VehicleDataset, build_transform
from street_falcon_reid.infer import run_inference, verify_submission
from street_falcon_reid.metrics import cosine_similarity, stable_rank
from street_falcon_reid.model import ReIDModel
from street_falcon_reid.predictor import BBox, InvalidBBox
from street_falcon_reid.records import VehicleRecord, write_records
from street_falcon_reid.search import GalleryIndex, SearchService
from street_falcon_reid.train import extract_embeddings


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    root = tmp_path_factory.mktemp("online")
    data = root / "data"
    images = data / "images"
    images.mkdir(parents=True)
    run = root / "run"
    run.mkdir()
    config = load_config("configs/baseline.toml")
    config["data"].update(image_height=32, image_width=32)
    config["inference"].update(batch_size=1, workers=0)
    for index, name in enumerate(("q1", "q2", "g1", "g2")):
        pixels = np.random.default_rng(index).integers(0, 256, (48, 64, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(images / f"{name}.jpg")
    for filename, ids in (("test_query.csv", ["q1", "q2"]), ("test_gallery.csv", ["g1", "g2"])):
        write_records(data / filename, [VehicleRecord(name, 2, 3, 40, 30) for name in ids])
    torch.manual_seed(17)
    model_config = {"backbone": "resnet18", "embedding_dim": 8, "num_classes": 2}
    model = ReIDModel(**model_config, pretrained=False)
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": model_config,
            "data_config": config["data"],
            "open_set_threshold": 0.5,
        },
        run / "checkpoint-best.pt",
    )
    text = Path("configs/baseline.toml").read_text()
    text = text.replace("image_height = 256", "image_height = 32")
    text = text.replace("image_width = 256", "image_width = 32")
    (run / "config.toml").write_text(text)
    run_inference(data, run / "checkpoint-best.pt", run / "submission", config)
    verify_submission(data, run / "submission")
    return data, run


@pytest.fixture(scope="module")
def service(assets):
    return SearchService.load(*assets)


def image_bytes(path: Path) -> bytes:
    return path.read_bytes()


def test_online_embedding_matches_dataset_and_batch_extraction(assets, service):
    data, _ = assets
    predictor = service.predictor
    record = VehicleRecord("q1", 2, 3, 40, 30)
    dataset = VehicleDataset(
        [record], data, build_transform(32, 32, training=False), crop_margin=0.05
    )
    vectors, ids = extract_embeddings(
        predictor.model, dataset, batch_size=1, workers=0, device=torch.device("cpu")
    )
    with Image.open(data / "images/q1.jpg") as image:
        online = predictor.embed_image(image, BBox(2, 3, 40, 30))
    assert ids == ["q1"]
    np.testing.assert_allclose(online, vectors[0], atol=1e-6)
    assert not predictor.model.training


@pytest.mark.parametrize("bbox", [
    BBox(-1, 0, 1, 1), BBox(0, -1, 1, 1), BBox(0, 0, 0, 1),
    BBox(0, 0, 1, -1), BBox(63, 0, 2, 1), BBox(0, 47, 1, 2),
    BBox(0.5, 0, 1, 1), BBox(True, 0, 1, 1),
])
def test_bbox_requires_client_rectangle_inside_image(bbox):
    with pytest.raises(InvalidBBox):
        bbox.validate(Image.new("RGB", (64, 48)))


def test_bbox_accepts_exact_image_boundary():
    BBox(0, 0, 64, 48).validate(Image.new("RGB", (64, 48)))


def test_top_k_matches_baseline_and_ties_are_stable():
    values = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32)
    gallery = GalleryIndex(["a", "b", "c"], values, "test")
    query = np.array([1, 0], dtype=np.float32)
    expected = stable_rank(cosine_similarity(query[None], values))[0]
    results = gallery.search(query, 10)
    assert [row["image_id"] for row in results] == [gallery.ids[i] for i in expected]
    assert [row["rank"] for row in results] == [1, 2, 3]
    assert len(gallery.search(query, 1)) == 1
    assert not gallery.embeddings.flags.writeable


@pytest.mark.parametrize("top_k", [0, 11, -1, 1.5, True])
def test_search_rejects_invalid_top_k(top_k):
    gallery = GalleryIndex(["a"], np.array([[1, 0]], dtype=np.float32), "test")
    with pytest.raises(ValueError):
        gallery.search(np.array([1, 0]), top_k)


def test_gallery_loads_only_gallery_rows_in_csv_order(assets, service):
    _, run = assets
    combined = np.load(run / "submission/embeddings.npy")
    assert service.gallery.ids == ("g1", "g2")
    np.testing.assert_allclose(service.gallery.embeddings, combined[2:], atol=1e-6)


@pytest.mark.parametrize("fault", ["model", "dimension", "threshold", "csv", "checksum", "config"])
def test_startup_rejects_incompatible_artifacts(assets, service, tmp_path, fault):
    data, source = assets
    run = tmp_path / "run"
    shutil.copytree(source / "submission", run / "submission")
    shutil.copy(source / "config.toml", run / "config.toml")
    metadata_path = run / "submission/run-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if fault == "model":
        metadata["checkpoint_sha256"] = "0" * 64
    elif fault == "dimension":
        metadata["embedding_dim"] = 999
    elif fault == "threshold":
        metadata["open_set_threshold"] = 0.9
    elif fault == "csv":
        metadata["gallery_csv_sha256"] = "0" * 64
    elif fault == "checksum":
        with (run / "submission/embeddings.npy").open("ab") as stream:
            stream.write(b"corruption")
    else:
        path = run / "config.toml"
        path.write_text(path.read_text().replace("crop_margin = 0.05", "crop_margin = 0.1"))
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        GalleryIndex.load(data, run, service.predictor)


def test_legacy_gallery_is_supported_with_explicit_warning(assets, service, tmp_path, caplog):
    data, source = assets
    run = tmp_path / "legacy"
    shutil.copytree(source / "submission", run / "submission")
    shutil.copy(source / "config.toml", run / "config.toml")
    path = run / "submission/run-metadata.json"
    metadata = json.loads(path.read_text())
    del metadata["gallery_csv_sha256"]
    path.write_text(json.dumps(metadata))
    gallery = GalleryIndex.load(data, run, service.predictor)
    assert gallery.ids == service.gallery.ids
    assert "Legacy run" in caplog.text


def test_api_reuses_resources_and_preserves_client_bbox(assets, service, monkeypatch):
    data, _ = assets
    loads = []
    seen = []
    original = service.predictor.embed_image

    def factory():
        loads.append(1)
        return service

    def embed(image, bbox):
        seen.append(bbox)
        return original(image, bbox)

    monkeypatch.setattr(service.predictor, "embed_image", embed)
    with TestClient(create_app(factory)) as client:
        def unexpected(*args, **kwargs):
            raise AssertionError("Model/gallery reloaded during request")
        monkeypatch.setattr(torch, "load", unexpected)
        monkeypatch.setattr(np, "load", unexpected)
        for _ in range(2):
            response = client.post(
                "/api/v1/search",
                files={"image": ("car.jpg", image_bytes(data / "images/q1.jpg"), "image/jpeg")},
                data={"x": 2, "y": 3, "w": 40, "h": 30, "top_k": 1},
            )
            assert response.status_code == 200, response.text
            result = response.json()
            assert len(result["matches"]) == 1
            assert result["model_version"] == service.predictor.model_version
            assert result["accepted"] == (result["matches"][0]["score"] >= result["threshold"])
        assert client.get("/api/v1/health").json() == {"status": "ok"}
    assert loads == [1]
    assert seen == [BBox(2, 3, 40, 30)] * 2


@pytest.mark.parametrize("field,value", [
    ("x", -1), ("y", -1), ("w", 0), ("h", -1),
    ("x", 64), ("w", 65), ("top_k", 0), ("top_k", 11), ("x", "abc"),
])
def test_api_rejects_invalid_coordinates_and_top_k(assets, service, field, value):
    data, _ = assets
    form = {"x": 0, "y": 0, "w": 1, "h": 1, "top_k": 10}
    form[field] = value
    with TestClient(create_app(lambda: service)) as client:
        response = client.post(
            "/api/v1/search", data=form,
            files={"image": ("car.jpg", image_bytes(data / "images/q1.jpg"))},
        )
        assert response.status_code == 422


def test_api_requires_bbox_and_valid_image(service):
    with TestClient(create_app(lambda: service)) as client:
        assert client.post("/api/v1/search", files={"image": ("x.jpg", b"bad")}).status_code == 422
        response = client.post(
            "/api/v1/search", files={"image": ("x.jpg", b"bad")},
            data={"x": 0, "y": 0, "w": 1, "h": 1},
        )
        assert response.status_code == 415


def test_api_upload_and_pixel_limits(service, monkeypatch):
    import street_falcon_reid.api as api
    monkeypatch.setattr(api, "MAX_REQUEST_BYTES", 1024)
    with TestClient(create_app(lambda: service)) as client:
        assert client.post("/api/v1/search", content=b"x" * 1025).status_code == 413
        assert client.post("/api/v1/search", content=iter([b"x" * 600] * 2)).status_code == 413
    monkeypatch.setattr(api, "MAX_REQUEST_BYTES", 10000)
    monkeypatch.setattr(api, "MAX_IMAGE_PIXELS", 10)
    payload = io.BytesIO()
    Image.new("RGB", (4, 4)).save(payload, format="PNG")
    with TestClient(create_app(lambda: service)) as client:
        response = client.post(
            "/api/v1/search", files={"image": ("x.png", payload.getvalue())},
            data={"x": 0, "y": 0, "w": 1, "h": 1},
        )
        assert response.status_code == 413


def test_service_refusal_keeps_top_k_and_threshold_equality_is_accepted(service, monkeypatch):
    monkeypatch.setattr(service.predictor, "embed_image", lambda *_: service.gallery.embeddings[0])
    image = Image.new("RGB", (1, 1))
    best = service.gallery.search(service.gallery.embeddings[0], 1)[0]["score"]
    monkeypatch.setattr(service.predictor, "threshold", best)
    assert service.search(image, BBox(0, 0, 1, 1), 1)["accepted"]
    monkeypatch.setattr(service.predictor, "threshold", best + 0.01)
    result = service.search(image, BBox(0, 0, 1, 1), 1)
    assert not result["accepted"]
    assert len(result["matches"]) == 1


def test_api_returns_busy_without_parallel_model_call(assets, service):
    data, _ = assets
    with TestClient(create_app(lambda: service)) as client:
        with service._lock:
            response = client.post(
                "/api/v1/search",
                files={"image": ("x.jpg", image_bytes(data / "images/q1.jpg"))},
                data={"x": 0, "y": 0, "w": 1, "h": 1},
            )
        assert response.status_code == 429
        assert response.headers["retry-after"] == "1"


def test_startup_failure_does_not_serve_requests():
    def broken():
        raise ValueError("incompatible model")
    with pytest.raises(ValueError, match="incompatible model"), TestClient(create_app(broken)):
        pass


def test_api_inference_failure_is_not_reported_as_invalid_upload(assets, service, monkeypatch):
    data, _ = assets

    def broken(*args):
        raise OSError("private runtime detail")

    monkeypatch.setattr(service, "search", broken)
    with TestClient(create_app(lambda: service)) as client:
        response = client.post(
            "/api/v1/search",
            files={"image": ("car.jpg", image_bytes(data / "images/q1.jpg"))},
            data={"x": 0, "y": 0, "w": 1, "h": 1},
        )
        assert response.status_code == 500
        assert "private runtime detail" not in response.text


def test_bundle_roundtrip_is_independent_of_csv(assets, service, tmp_path):
    from street_falcon_reid.gallery_bundle import export_bundle, load_bundle
    data, run = assets
    output = export_bundle(data, run, tmp_path / "bundle")
    loaded = load_bundle(output, service.predictor)
    assert loaded.ids == service.gallery.ids
    np.testing.assert_allclose(loaded.embeddings, service.gallery.embeddings, atol=1e-6)
    with pytest.raises(FileExistsError):
        export_bundle(data, run, output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["source_csv_checksum_verified"] is True


@pytest.mark.parametrize("filename", ["ids.json", "gallery.npy"])
def test_bundle_detects_modified_data(assets, service, tmp_path, filename):
    from street_falcon_reid.gallery_bundle import export_bundle, load_bundle
    output = export_bundle(*assets, tmp_path / "bundle")
    with (output / filename).open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(output, service.predictor)


@pytest.mark.parametrize("key,value", [
    ("schema_version", 2), ("checkpoint_sha256", "wrong"), ("embedding_dim", 999),
    ("threshold", 0.9), ("gallery_count", 999), ("preprocessing", {}),
])
def test_bundle_rejects_incompatible_manifest(assets, service, tmp_path, key, value):
    from street_falcon_reid.gallery_bundle import export_bundle, load_bundle
    output = export_bundle(*assets, tmp_path / "bundle")
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[key] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_bundle(output, service.predictor)


def test_legacy_export_requires_explicit_acknowledgement(assets, tmp_path):
    from street_falcon_reid.gallery_bundle import export_bundle
    data, run = assets
    copy = tmp_path / "legacy"
    shutil.copytree(run, copy)
    path = copy / "submission/run-metadata.json"
    metadata = json.loads(path.read_text())
    del metadata["gallery_csv_sha256"]
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="allow-legacy"):
        export_bundle(data, copy, tmp_path / "blocked")
    output = export_bundle(data, copy, tmp_path / "allowed", allow_legacy=True)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["source_csv_checksum_verified"] is False


def test_api_request_ids_errors_and_runtime_info(service):
    with TestClient(create_app(lambda: service)) as client:
        first = client.get("/api/v1/live")
        second = client.get("/api/v1/ready")
        assert first.json() == {"status": "alive"}
        assert second.json()["gallery_count"] == 2
        assert first.headers["x-request-id"] != second.headers["x-request-id"]
        assert float(first.headers["x-process-time-ms"]) >= 0
        assert client.get("/api/v1/info").json()["bbox_source"] == "client"
        response = client.post("/api/v1/search", data={"x": "secret-invalid-value"})
        assert response.status_code == 422
        assert response.json()["request_id"] == response.headers["x-request-id"]
        assert response.json()["error"]["code"] == "validation_error"
        assert "secret-invalid-value" not in response.text
        response = client.get("/unknown")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"


def test_api_png_and_request_id_success(service):
    image = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(image, format="PNG")
    with TestClient(create_app(lambda: service)) as client:
        response = client.post(
            "/api/v1/search", files={"image": ("car.png", image.getvalue(), "image/png")},
            data={"x": 0, "y": 0, "w": 16, "h": 16},
        )
        assert response.status_code == 200
        assert response.json()["request_id"] == response.headers["x-request-id"]


def test_api_rejects_animation(service):
    image = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(
        image, format="PNG", save_all=True,
        append_images=[Image.new("RGB", (16, 16), "blue")], duration=100,
    )
    with TestClient(create_app(lambda: service)) as client:
        response = client.post(
            "/api/v1/search", files={"image": ("animation.png", image.getvalue())},
            data={"x": 0, "y": 0, "w": 16, "h": 16},
        )
        assert response.status_code == 415


def test_api_file_limit_is_independent_of_multipart_limit(service, monkeypatch):
    import street_falcon_reid.api as api
    monkeypatch.setattr(api, "MAX_IMAGE_BYTES", 10)
    with TestClient(create_app(lambda: service)) as client:
        response = client.post(
            "/api/v1/search", files={"image": ("large.jpg", b"x" * 11)},
            data={"x": 0, "y": 0, "w": 1, "h": 1},
        )
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "payload_too_large"


def test_service_bundle_startup_and_warmup(assets, tmp_path):
    from street_falcon_reid.gallery_bundle import export_bundle
    _, run = assets
    output = export_bundle(*assets, tmp_path / "bundle")
    service = SearchService.load_bundle(run / "checkpoint-best.pt", output)
    assert len(service.gallery.ids) == 2
    with Image.new("RGB", (16, 16)) as image:
        assert len(service.search(image, BBox(0, 0, 16, 16))["matches"]) == 2


def test_search_ui_assets_and_api_coexist(service):
    with TestClient(create_app(lambda: service)) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert 'id="x"' in page.text and 'id="h"' in page.text
        assert "frame-ancestors" in page.headers["content-security-policy"]
        assert client.get("/ui-assets/app.js").status_code == 200
        assert client.get("/ui-assets/style.css").status_code == 200
        assert client.get("/ui-assets/missing.js").status_code == 404
        assert client.get("/api/v1/ready").status_code == 200


def test_gallery_thumbnails_restrict_access(service, tmp_path, monkeypatch):
    import hashlib

    monkeypatch.setenv("LCT_THUMBNAIL_DIR", str(tmp_path))
    filename = hashlib.sha256(b"g1").hexdigest() + ".jpg"
    with Image.new("RGB", (24, 16), "green") as image:
        image.save(tmp_path / filename)
    with TestClient(create_app(lambda: service)) as client:
        response = client.get("/api/v1/gallery/g1/thumbnail")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        with Image.open(io.BytesIO(response.content)) as image:
            assert image.size == (24, 16)
        assert client.get("/api/v1/gallery/q1/thumbnail").status_code == 404
        assert client.get("/api/v1/gallery/g2/thumbnail").status_code == 404
        (tmp_path / filename).unlink()
        (tmp_path / filename).symlink_to(tmp_path / "private.jpg")
        assert client.get("/api/v1/gallery/g1/thumbnail").status_code == 404


def test_gallery_thumbnails_optional(service, monkeypatch):
    monkeypatch.delenv("LCT_THUMBNAIL_DIR", raising=False)
    with TestClient(create_app(lambda: service)) as client:
        assert client.get("/api/v1/gallery/g1/thumbnail").status_code == 404
