from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dataset import VehicleDataset, build_transform
from .metrics import cosine_similarity, stable_rank
from .predictor import load_model
from .prepare import sha256_file
from .records import read_records
from .train import extract_embeddings


def run_inference(
    data_dir: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
) -> Path:
    data_root = Path(data_dir).resolve()
    checkpoint_file = Path(checkpoint_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, device = load_model(checkpoint_file, str(device))

    data_config = config["data"]
    inference_config = config["inference"]
    transform = build_transform(
        int(data_config["image_height"]),
        int(data_config["image_width"]),
        training=False,
    )
    query_records = read_records(data_root / "test_query.csv", require_labels=False)
    gallery_records = read_records(data_root / "test_gallery.csv", require_labels=False)
    query_dataset = VehicleDataset(
        query_records,
        data_root,
        transform,
        crop_margin=float(data_config["crop_margin"]),
    )
    gallery_dataset = VehicleDataset(
        gallery_records,
        data_root,
        transform,
        crop_margin=float(data_config["crop_margin"]),
    )
    query_embeddings, query_ids = extract_embeddings(
        model,
        query_dataset,
        batch_size=int(inference_config["batch_size"]),
        workers=int(inference_config["workers"]),
        device=device,
    )
    gallery_embeddings, gallery_ids = extract_embeddings(
        model,
        gallery_dataset,
        batch_size=int(inference_config["batch_size"]),
        workers=int(inference_config["workers"]),
        device=device,
    )
    similarities = cosine_similarity(query_embeddings, gallery_embeddings)
    order = stable_rank(similarities)
    top_k = min(int(inference_config["top_k"]), len(gallery_ids))

    submission_path = output / "submission.csv"
    with submission_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        for query_index, query_id in enumerate(query_ids):
            ranked_ids = [gallery_ids[index] for index in order[query_index, :top_k]]
            writer.writerow([query_id, *ranked_ids])

    threshold = float(checkpoint["open_set_threshold"])
    candidates_path = output / "candidates.csv"
    with candidates_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["query_id", "gallery_id", "confidence"])
        for query_index, query_id in enumerate(query_ids):
            best_index = int(order[query_index, 0])
            confidence = float(similarities[query_index, best_index])
            if confidence >= threshold:
                writer.writerow([query_id, gallery_ids[best_index], f"{confidence:.8f}"])

    embeddings_path = output / "embeddings.npy"
    np.save(
        embeddings_path,
        np.concatenate([query_embeddings, gallery_embeddings], axis=0).astype(np.float32),
        allow_pickle=False,
    )
    metadata = {
        "schema_version": 1,
        "checkpoint": checkpoint_file.name,
        "checkpoint_sha256": sha256_file(checkpoint_file),
        "source_commit": checkpoint.get("source_commit", "unknown"),
        "open_set_threshold": threshold,
        "query_count": len(query_ids),
        "gallery_count": len(gallery_ids),
        "gallery_csv_sha256": sha256_file(data_root / "test_gallery.csv"),
        "embedding_dim": int(query_embeddings.shape[1]),
        "device": str(device),
        "torch": torch.__version__,
    }
    metadata_path = output / "run-metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output


def verify_submission(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    top_k: int = 10,
) -> dict[str, object]:
    data_root = Path(data_dir)
    output = Path(output_dir)
    query = read_records(data_root / "test_query.csv", require_labels=False)
    gallery = read_records(data_root / "test_gallery.csv", require_labels=False)
    query_ids = [item.image_id for item in query]
    gallery_ids = {item.image_id for item in gallery}
    expected_width = 1 + min(top_k, len(gallery))

    submission_rows: dict[str, list[str]] = {}
    with (output / "submission.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        for number, row in enumerate(csv.reader(stream), start=1):
            if len(row) != expected_width:
                raise ValueError(
                    f"submission.csv строка {number}: ожидалось {expected_width} полей"
                )
            query_id, candidates = row[0], row[1:]
            if query_id in submission_rows:
                raise ValueError(f"Повтор query_id в submission.csv: {query_id}")
            if len(candidates) != len(set(candidates)):
                raise ValueError(f"Повтор gallery_id для query_id={query_id}")
            unknown = set(candidates).difference(gallery_ids)
            if unknown:
                raise ValueError(f"Неизвестные gallery_id для {query_id}: {sorted(unknown)}")
            submission_rows[query_id] = candidates
    if set(submission_rows) != set(query_ids):
        raise ValueError("Набор query_id в submission.csv не совпадает с test_query.csv")

    with (output / "candidates.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"query_id", "gallery_id", "confidence"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("В candidates.csv отсутствуют обязательные колонки")
        candidate_count = 0
        for row in reader:
            if row["query_id"] not in submission_rows:
                raise ValueError(f"Неизвестный query_id в candidates.csv: {row['query_id']}")
            if row["gallery_id"] not in gallery_ids:
                raise ValueError(f"Неизвестный gallery_id в candidates.csv: {row['gallery_id']}")
            if not np.isfinite(float(row["confidence"])):
                raise ValueError("confidence должен быть конечным числом")
            candidate_count += 1

    embeddings = np.load(output / "embeddings.npy", allow_pickle=False)
    expected_shape_0 = len(query) + len(gallery)
    if embeddings.ndim != 2 or embeddings.shape[0] != expected_shape_0:
        raise ValueError(
            f"embeddings.npy имеет форму {embeddings.shape}, ожидалось ({expected_shape_0}, D)"
        )
    if embeddings.dtype != np.float32 or not np.isfinite(embeddings).all():
        raise ValueError("embeddings.npy должен содержать конечные float32")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("Строки embeddings.npy должны быть L2-нормализованы")

    report = {
        "query_count": len(query),
        "gallery_count": len(gallery),
        "candidate_count": candidate_count,
        "embedding_shape": list(embeddings.shape),
        "sha256": {
            name: sha256_file(output / name)
            for name in ("submission.csv", "candidates.csv", "embeddings.npy")
        },
    }
    (output / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def package_submission(output_dir: str | Path) -> Path:
    output = Path(output_dir)
    destination = output / "submission.zip"
    temporary = output / ".submission.zip.tmp"
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("submission.csv", "candidates.csv", "embeddings.npy"):
            archive.write(output / name, arcname=name)
    temporary.replace(destination)
    return destination
