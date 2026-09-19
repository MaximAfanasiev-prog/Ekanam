from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from street_falcon_reid.web import DashboardData, DashboardStore


def _write_records(path: Path, image_ids: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id", "x", "y", "w", "h"])
        for image_id in image_ids:
            writer.writerow([image_id, 2, 2, 16, 12])


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "data"
    image_dir = data_dir / "images"
    run_dir = tmp_path / "runs" / "baseline-test"
    submission_dir = run_dir / "submission"
    image_dir.mkdir(parents=True)
    submission_dir.mkdir(parents=True)
    _write_records(data_dir / "test_query.csv", ["q1", "q2"])
    _write_records(data_dir / "test_gallery.csv", ["g1", "g2"])
    for index, image_id in enumerate(["q1", "q2", "g1", "g2"]):
        Image.new("RGB", (24, 20), (50 + index * 25, 80, 120)).save(
            image_dir / f"{image_id}.jpg"
        )

    history = []
    for epoch in range(1, 31):
        history.append(
            {
                "epoch": epoch,
                "loss": 5.0 / epoch,
                "ranking": {"mAP@10": epoch / 100, "Rank-1": 0.4, "Rank-5": 0.6},
                "refusal": {"F1": 0.8, "Precision": 0.8, "Recall": 0.8},
            }
        )
    (run_dir / "history.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in history), encoding="utf-8"
    )
    (run_dir / "validation-report.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "split.json").write_text(
        json.dumps(
            {
                "counts": {
                    "train_records": 10,
                    "query_records": 2,
                    "gallery_records": 2,
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "config.toml").write_text(
        '[model]\nbackbone = "resnet50"\n[inference]\ntop_k = 10\n', encoding="utf-8"
    )
    (submission_dir / "run-metadata.json").write_text(
        json.dumps(
            {
                "embedding_dim": 2,
                "checkpoint_sha256": "a" * 64,
                "source_commit": "b" * 40,
                "open_set_threshold": 0.75,
                "device": "cpu",
                "torch": "test",
            }
        ),
        encoding="utf-8",
    )
    (submission_dir / "verification.json").write_text(
        json.dumps(
            {
                "query_count": 2,
                "gallery_count": 2,
                "candidate_count": 1,
                "embedding_shape": [4, 2],
                "sha256": {},
            }
        ),
        encoding="utf-8",
    )
    (submission_dir / "submission.csv").write_text(
        "q1,g1,g2\nq2,g2,g1\n", encoding="utf-8"
    )
    (submission_dir / "candidates.csv").write_text(
        "query_id,gallery_id,confidence\nq1,g1,1.0\n", encoding="utf-8"
    )
    np.save(
        submission_dir / "embeddings.npy",
        np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32),
        allow_pickle=False,
    )
    return data_dir, run_dir


def test_dashboard_reads_metrics_and_retrieval_results(tmp_path: Path) -> None:
    data_dir, run_dir = _fixture(tmp_path)
    dashboard = DashboardData(data_dir, run_dir)

    summary = dashboard.summary()
    detail = dashboard.query_detail("q1")

    assert summary["best"]["epoch"] == 30
    assert "не результат скрытого leaderboard" in summary["boundary"]
    assert summary["submission"]["acceptedQueries"] == 1
    assert detail["decision"] == "accepted"
    assert detail["gallery"][0] == {"rank": 1, "imageId": "g1", "score": 1.0}
    assert detail["groundTruthAvailable"] is False
    assert dashboard.image("q1").startswith(b"\xff\xd8")


def test_dashboard_store_reports_missing_run_as_degraded(tmp_path: Path) -> None:
    store = DashboardStore(tmp_path / "missing-data", tmp_path / "missing-run")

    assert store.data is None
    assert store.error
