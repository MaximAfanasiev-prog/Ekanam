import numpy as np

from street_falcon_reid.metrics import (
    calibrate_open_set,
    cosine_similarity,
    ranking_metrics,
)
from street_falcon_reid.records import VehicleRecord


def _record(image_id: str, vehicle_id: str, camera_id: str) -> VehicleRecord:
    return VehicleRecord(image_id, 0, 0, 10, 10, vehicle_id, camera_id)


def test_ranking_and_refusal_contract() -> None:
    query = [_record("q1", "a", "1"), _record("q2", "missing", "1")]
    gallery = [_record("g1", "a", "2"), _record("g2", "b", "2")]
    query_embeddings = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float32)
    gallery_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    similarities = cosine_similarity(query_embeddings, gallery_embeddings)

    ranking = ranking_metrics(query, gallery, similarities, top_k=2)
    threshold, refusal = calibrate_open_set(query, gallery, similarities)

    assert ranking["mAP@2"] == 1.0
    assert ranking["Rank-1"] == 1.0
    assert ranking["n_open_set"] == 1
    assert refusal["F1"] == 1.0
    assert -1.0 <= threshold <= 1.0
