from __future__ import annotations

import math

import numpy as np

from .records import VehicleRecord


def cosine_similarity(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    query = _normalize(query)
    gallery = _normalize(gallery)
    return query @ gallery.T


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)


def stable_rank(similarities: np.ndarray) -> np.ndarray:
    return np.argsort(-similarities, axis=1, kind="stable")


def ranking_metrics(
    query: list[VehicleRecord],
    gallery: list[VehicleRecord],
    similarities: np.ndarray,
    *,
    top_k: int,
) -> dict[str, float | int]:
    order = stable_rank(similarities)
    average_precisions: list[float] = []
    rank_1: list[bool] = []
    rank_5: list[bool] = []
    open_set = 0

    for index, query_record in enumerate(query):
        positive = np.array(
            [
                item.vehicle_id == query_record.vehicle_id
                and item.camera_id != query_record.camera_id
                for item in gallery
            ],
            dtype=bool,
        )
        junk = np.array(
            [
                item.vehicle_id == query_record.vehicle_id
                and item.camera_id == query_record.camera_id
                for item in gallery
            ],
            dtype=bool,
        )
        n_positive = int(positive.sum())
        if n_positive == 0:
            open_set += 1
            continue

        ranked = [gallery_index for gallery_index in order[index] if not junk[gallery_index]][
            :top_k
        ]
        relevance = positive[ranked]
        cumulative = np.cumsum(relevance)
        precision = cumulative / (np.arange(len(relevance)) + 1)
        average_precisions.append(float((precision * relevance).sum() / min(n_positive, top_k)))
        rank_1.append(bool(relevance[:1].any()))
        rank_5.append(bool(relevance[:5].any()))

    return {
        "n_scored": len(average_precisions),
        "n_open_set": open_set,
        f"mAP@{top_k}": _mean(average_precisions),
        "Rank-1": _mean(rank_1),
        "Rank-5": _mean(rank_5),
    }


def calibrate_open_set(
    query: list[VehicleRecord],
    gallery: list[VehicleRecord],
    similarities: np.ndarray,
) -> tuple[float, dict[str, float | int]]:
    if not gallery:
        raise ValueError("Нельзя калибровать порог на пустой gallery")
    order = stable_rank(similarities)
    top_indices = order[:, 0]
    scores = similarities[np.arange(len(query)), top_indices]
    top_correct = np.array(
        [
            gallery[g].vehicle_id == item.vehicle_id
            for item, g in zip(query, top_indices, strict=False)
        ],
        dtype=bool,
    )
    has_match = np.array(
        [
            any(
                gallery_item.vehicle_id == item.vehicle_id
                and gallery_item.camera_id != item.camera_id
                for gallery_item in gallery
            )
            for item in query
        ],
        dtype=bool,
    )

    unique = np.unique(scores)
    epsilon = np.finfo(np.float32).eps
    thresholds = np.concatenate(
        ([float(unique.max() + epsilon)], unique[::-1], [float(unique.min() - epsilon)])
    )
    best_threshold = float(thresholds[0])
    best_report: dict[str, float | int] | None = None
    best_key = (-1.0, -1.0, -1.0, -math.inf)
    for threshold in thresholds:
        accepted = scores >= threshold
        tp = int((accepted & has_match & top_correct).sum())
        fp = int((accepted & ~(has_match & top_correct)).sum())
        fn = int((~accepted & has_match).sum())
        tn = int((~accepted & ~has_match).sum())
        fp_open = int((accepted & ~has_match).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        tnr = tn / (tn + fp_open) if tn + fp_open else 0.0
        key = (f1, tnr, precision, float(threshold))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_report = {
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "TNR": tnr,
                "threshold": best_threshold,
            }
    assert best_report is not None
    return best_threshold, best_report


def _mean(values: list[float] | list[bool]) -> float:
    return float(np.mean(values)) if values else 0.0
