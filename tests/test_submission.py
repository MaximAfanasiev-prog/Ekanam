import csv
from pathlib import Path

import numpy as np

from street_falcon_reid.infer import package_submission, verify_submission


def _write_table(path: Path, image_ids: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id", "x", "y", "w", "h"])
        for image_id in image_ids:
            writer.writerow([image_id, 0, 0, 10, 10])


def test_verify_and_package_submission(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    data_dir.mkdir()
    output_dir.mkdir()
    _write_table(data_dir / "test_query.csv", ["q1", "q2"])
    _write_table(data_dir / "test_gallery.csv", ["g1", "g2"])

    (output_dir / "submission.csv").write_text("q1,g1,g2\nq2,g2,g1\n", encoding="utf-8")
    (output_dir / "candidates.csv").write_text(
        "query_id,gallery_id,confidence\nq1,g1,0.9\n", encoding="utf-8"
    )
    embeddings = np.eye(4, dtype=np.float32)
    np.save(output_dir / "embeddings.npy", embeddings, allow_pickle=False)

    report = verify_submission(data_dir, output_dir)
    archive = package_submission(output_dir)

    assert report["query_count"] == 2
    assert archive.is_file()
