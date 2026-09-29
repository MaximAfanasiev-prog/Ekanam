"""Cross-check: our metric code vs the organizers' evaluate.py on files written exactly as the test submission.

    python -m models.yolo_finetune.check_official          # exit code 1 if any number differs

For each saved set of mini-val embeddings (runs/yolo_embedding/minival/<mode>/<model>/embeddings.npy,
runs/yolo_finetune/eval/<split>/<tag>[_rerank]/embeddings.npy) the artifacts are written afresh by the code path of
predict.py (rerank.retrieve + retrieval writers; RERANK_DEFAULT for `_rerank` directories), with two thresholds:
our F1-optimal sweep and the submission rule (refuse the REFUSE_RATE share of queries with the lowest top-1).
evaluate.py is run on them with the split's mini_gt.csv and every number of ours must be equal to it exactly.
The split's mini_gt.csv itself is checked against train.csv first (vehicle_id / camera_id, query/gallery order).
"""

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
from models.yolo_embedding import metrics as M
from models.yolo_embedding.data import REPO_DIR, read_table

from .data import load_split, train_table
from .evaluate import run_official, score_retrieved, write_artifacts
from .rerank import RERANK_DEFAULT, retrieve

REFUSE_RATE = 0.25  # docker-compose.yml: predict.py --refuse-rate
RANKING = ("n_scored", "n_openset_excluded", "mAP@10", "Rank-1", "Rank-5")
FULL = ("mAP_full", "mINP")
CANDIDATES = ("TP", "FP", "FN", "TN", "n_openset_queries", "Precision", "Recall", "F1", "TNR", "PR-AUC")


def run_dirs():
    """-> [(run dir, split dir, re-ranked?)]"""
    out = [(d, REPO_DIR / "data", False) for d in sorted((REPO_DIR / "runs/yolo_embedding/minival").glob("*/*"))]
    for d in sorted((REPO_DIR / "runs/yolo_finetune/eval").glob("*/*")):
        split = d.parent.name
        split_dir = REPO_DIR / ("data" if split == "data" else f"data/splits/{split}")
        out.append((d, split_dir, d.name.endswith("_rerank")))
    return [r for r in out if (r[0] / "embeddings.npy").exists()]


def check_gt(split_dir):
    """mini_gt.csv is the evaluate.py ground truth: train.csv labels, query then gallery in mini_*.csv order."""
    gt = read_table(split_dir / "mini_gt.csv")
    q, g = read_table(split_dir / "mini_query.csv"), read_table(split_dir / "mini_gallery.csv")
    assert list(gt.columns) == ["image_id", "vehicle_id", "camera_id", "split"], gt.columns
    assert gt[gt.split == "query"].image_id.tolist() == q.image_id.tolist()
    assert gt[gt.split == "gallery"].image_id.tolist() == g.image_id.tolist()
    assert len(gt) == len(q) + len(g)
    train = train_table().set_index("image_id")
    lab = train.loc[gt.image_id, ["vehicle_id", "camera_id"]]
    assert (lab.vehicle_id.to_numpy() == gt.vehicle_id.to_numpy()).all()
    assert (lab.camera_id.to_numpy() == gt.camera_id.to_numpy()).all()


def same(a, b):
    return a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))


def compare(res, cand, off):
    pairs = [(k, res[k], off["ranking"][k]) for k in RANKING] + [(k, res[k], off["full_ranking"][k]) for k in FULL]
    pairs += [(k, cand[k], off["candidates"][k]) for k in CANDIDATES]
    return [f"{k}: ours={a!r} official={b!r}" for k, a, b in pairs if not same(a, b)]


def main():
    n_bad, checked = 0, set()
    with tempfile.TemporaryDirectory() as tmp:
        for run_dir, split_dir, rerank in run_dirs():
            if split_dir not in checked:
                check_gt(split_dir)
                checked.add(split_dir)
            sp = load_split(split_dir)
            emb = np.load(run_dir / "embeddings.npy")
            assert emb.shape[0] == len(sp["q_ids"]) + len(sp["g_ids"]), emb.shape
            q, g = emb[: len(sp["q_ids"])], emb[len(sp["q_ids"]) :]
            retrieved = retrieve(q, g, RERANK_DEFAULT if rerank else None)
            res = score_retrieved(q, g, sp, retrieved)
            decision = M.top1_decision_inputs(retrieved[0], retrieved[1], sp["q_vid"], sp["q_cam"], sp["g_vid"],
                                              sp["g_cam"])
            rule = M.candidate_metrics(*decision, float(np.quantile(retrieved[0][:, 0], REFUSE_RATE)))
            name = run_dir.relative_to(REPO_DIR)
            for label, cand in (("F1-opt", res["candidates"]), (f"refuse {REFUSE_RATE}", rule)):
                out = Path(tmp) / label
                write_artifacts(out, sp, q, g, retrieved, cand["threshold"])
                diffs = compare(res, cand, run_official(out, sp))
                n_bad += bool(diffs)
                print(f"{'OK  ' if not diffs else 'DIFF'} {name} [{label}]  mAP@10={res['mAP@10']:.4f} "
                      f"mINP={res['mINP']:.4f} F1={cand['F1']:.4f} TNR={cand['TNR']:.4f} PR-AUC={cand['PR-AUC']:.4f}",
                      flush=True)
                for d in diffs:
                    print("     ", d)
    print(f"{n_bad} check(s) differ from evaluate.py" if n_bad else "all checks identical to evaluate.py")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
