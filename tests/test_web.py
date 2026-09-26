from __future__ import annotations

import csv
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
from PIL import Image

from street_falcon_reid.web import DashboardData, DashboardHandler, DashboardStore


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


class TestDashboardErrors(unittest.TestCase):
    def test_missing_file_and_recovery(self) -> None:
        with TemporaryDirectory() as directory:
            data_dir, run_dir = _fixture(Path(directory))
            path = run_dir / "history.jsonl"
            original = path.read_text(encoding="utf-8")
            path.unlink()
            store = DashboardStore(data_dir, run_dir)
            self.assertIsNone(store.data)
            self.assertIn("history.jsonl", store.error)
            self.assertNotIn(directory, store.error)
            with self.assertRaises(RuntimeError):
                store.require()
            path.write_text(original, encoding="utf-8")
            store.reload()
            self.assertIsNone(store.error)
            self.assertEqual(store.require().summary()["best"]["epoch"], 30)

    def test_invalid_results_return_json_service_unavailable(self) -> None:
        cases = [
            ("validation-report.json", "{broken"),
            ("config.toml", "[broken"),
            ("submission/candidates.csv", "wrong,columns\nx,y\n"),
            ("submission/candidates.csv", "query_id,gallery_id,confidence\nq1,g1\n"),
        ]
        for filename, contents in cases:
            with self.subTest(filename=filename, contents=contents):
                with TemporaryDirectory() as directory:
                    data_dir, run_dir = _fixture(Path(directory))
                    (run_dir / filename).write_text(contents, encoding="utf-8")
                    store = DashboardStore(data_dir, run_dir)
                    self.assertIsNone(store.data)
                    self.assertIn("invalid format", store.error)
                    handler = type("TestHandler", (DashboardHandler,), {"store": store})
                    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
                        thread = threading.Thread(target=server.serve_forever, daemon=True)
                        thread.start()
                        try:
                            address = f"http://127.0.0.1:{server.server_port}"
                            for endpoint in ("/api/health", "/api/summary", "/api/queries"):
                                with self.assertRaises(HTTPError) as caught:
                                    urlopen(address + endpoint, timeout=3)
                                with caught.exception as response:
                                    self.assertEqual(response.code, 503)
                                    payload = json.load(response)
                                self.assertIn("invalid format", payload["error"])
                                self.assertNotIn(directory, payload["error"])
                                if endpoint == "/api/health":
                                    self.assertEqual(payload["status"], "degraded")
                            with urlopen(address + "/", timeout=3) as response:
                                self.assertEqual(response.status, 200)
                        finally:
                            server.shutdown()
                            thread.join(timeout=3)

    def test_existing_dashboard_behavior(self) -> None:
        with TemporaryDirectory() as directory:
            test_dashboard_reads_metrics_and_retrieval_results(Path(directory))
        with TemporaryDirectory() as directory:
            test_dashboard_store_reports_missing_run_as_degraded(Path(directory))


class TestResultConsistency(unittest.TestCase):
    def test_invalid_rankings_and_candidates_are_rejected(self) -> None:
        cases = [
            ("submission.csv", "q1,missing\nq2,g2\n"),
            ("submission.csv", "q1,g1,g1\nq2,g2\n"),
            ("candidates.csv", "query_id,gallery_id,confidence\nmissing,g1,1\n"),
            ("candidates.csv", "query_id,gallery_id,confidence\nq1,missing,1\n"),
            ("candidates.csv", "query_id,gallery_id,confidence\nq1,g1,nan\n"),
        ]
        for filename, content in cases:
            with self.subTest(filename=filename, content=content):
                with TemporaryDirectory() as directory:
                    data_dir, run_dir = _fixture(Path(directory))
                    (run_dir / "submission" / filename).write_text(content, encoding="utf-8")
                    store = DashboardStore(data_dir, run_dir)
                    self.assertIsNone(store.data)
                    self.assertIn("invalid format", store.error)

    def test_invalid_embeddings_and_recovery(self) -> None:
        arrays = [
            np.ones((3, 2)), np.ones((4, 3)), np.ones(4),
            np.full((4, 2), np.nan), np.full((4, 2), np.inf),
            np.zeros((4, 2)), np.full((4, 2), "bad"),
            np.ones((4, 2), dtype=complex),
        ]
        for values in arrays:
            with self.subTest(shape=values.shape, dtype=values.dtype):
                with TemporaryDirectory() as directory:
                    data_dir, run_dir = _fixture(Path(directory))
                    dashboard = DashboardData(data_dir, run_dir)
                    path = run_dir / "submission" / "embeddings.npy"
                    np.save(path, values, allow_pickle=False)
                    with self.assertRaises(ValueError):
                        dashboard.query_detail("q1")
                    np.save(path, np.ones((4, 2)), allow_pickle=False)
                    result = dashboard.query_detail("q1")
                    self.assertAlmostEqual(result["gallery"][0]["score"], 1.0, places=6)

    def test_unknown_query_still_raises_key_error(self) -> None:
        with TemporaryDirectory() as directory:
            data_dir, run_dir = _fixture(Path(directory))
            dashboard = DashboardData(data_dir, run_dir)
            with self.assertRaises(KeyError):
                dashboard.query_detail("unknown")
