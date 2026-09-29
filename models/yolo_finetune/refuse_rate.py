"""Justification of `predict.py --refuse-rate`: the candidates rule evaluated on the three open-set mini-val splits.

    python -m models.yolo_finetune.refuse_rate            # -> docs/results/yolo_finetune_refuse_rate.json

Inputs are the embeddings written by `evaluate --official` for the final configuration (squash, `feat`, flip TTA,
no re-ranking): the seed-42 checkpoint on seed 42 and the held-out checkpoints trained with the same config on
seeds 7 and 2026 (each evaluated on its own split). For every split and every refuse rate R the threshold is the
R-quantile of that split's top-1 scores, exactly as predict.py computes it on the test; the decision metrics are
the organizers' ones (models.yolo_embedding.metrics.candidate_metrics). For comparison the file also holds the
split's own F1-optimal fixed threshold, the seed-42 threshold transferred to the other splits, and the top-1 score
distribution of the test, which is why a fixed threshold is not used: for the seed-42 checkpoint (the one the
threshold belongs to, submissions/yolo_finetune/yolo26l-cls-reid) and for the final full-train model (the
committed embeddings.npy).
"""

import json

import numpy as np
from models.yolo_embedding import metrics as M
from models.yolo_embedding.data import DATASET_DIR, REPO_DIR, read_table
from models.yolo_embedding.retrieval import search

from .data import load_split

EVAL_DIR = REPO_DIR / "runs/yolo_finetune/eval"
SPLITS = {  # seed -> (split dir, evaluate --official output dir)
    42: ("data", "data/p1_26l_tronly_squash"),
    7: ("data/splits/seed7", "seed7/ho_26l_tronly_squash_s7"),
    2026: ("data/splits/seed2026", "seed2026/ho_26l_tronly_squash_s2026"),
}
RATES = np.round(np.arange(0.10, 0.4001, 0.01), 2)
OUT = REPO_DIR / "docs/results/yolo_finetune_refuse_rate.json"
TEST_EMB = {  # test embeddings: seed-42 checkpoint (80% of train.csv) and the submitted full-train model
    "test": REPO_DIR / "submissions/yolo_finetune/yolo26l-cls-reid/embeddings.npy",
    "test_final_model": REPO_DIR / "embeddings.npy",
}


def decision_inputs(split_dir, emb_path):
    sp = load_split(REPO_DIR / split_dir)
    emb = np.load(emb_path)
    q, g = emb[: len(sp["q_ids"])], emb[len(sp["q_ids"]) :]
    return M.top1_decision_inputs(*search(q, g), sp["q_vid"], sp["q_cam"], sp["g_vid"], sp["g_cam"])


def main():
    inputs = {seed: decision_inputs(d, EVAL_DIR / e / "embeddings.npy") for seed, (d, e) in SPLITS.items()}
    thr42 = M.sweep_threshold(*inputs[42])[0]["threshold"]
    keys = ("threshold", "Precision", "Recall", "F1", "TNR", "PR-AUC")
    res = {"splits": {}, "by_rate": {}, "seed42_threshold": thr42}

    for seed, (score, correct, has_match) in inputs.items():
        res["splits"][seed] = {
            "n_query": len(score),
            "no_match_share": float(1 - has_match.mean()),
            "top1_median": float(np.median(score)),
            "f1_optimal": {k: M.sweep_threshold(score, correct, has_match)[0][k] for k in keys},
            "seed42_threshold": {k: M.candidate_metrics(score, correct, has_match, thr42)[k] for k in keys},
        }
    for r in RATES:
        per = {seed: M.candidate_metrics(s, c, h, float(np.quantile(s, r))) for seed, (s, c, h) in inputs.items()}
        res["by_rate"][f"{r:.2f}"] = {
            **{f"mean_{k}": float(np.mean([m[k] for m in per.values()])) for k in keys[1:]},
            "min_F1": float(min(m["F1"] for m in per.values())),
            "per_split": {seed: {k: m[k] for k in keys} for seed, m in per.items()},
        }

    n_q = len(read_table(DATASET_DIR / "test_query.csv"))
    for key, path in TEST_EMB.items():
        test = np.load(path)
        test_top1 = search(test[:n_q], test[n_q:])[0][:, 0]  # as predict.py computes the refuse-rate threshold
        res[key] = {
            "embeddings": str(path.relative_to(REPO_DIR)),
            "n_query": n_q,
            "top1_median": float(np.median(test_top1)),
            "refused_at_seed42_threshold": float((test_top1 < thr42).mean()),
            "threshold_at_rate_0.25": float(np.quantile(test_top1, 0.25)),
        }

    OUT.write_text(json.dumps(res, indent=1))
    print(f"seed-42 F1-optimal threshold {thr42:.4f}")
    for seed, s in res["splits"].items():
        print(f"seed {seed:4d}: no-match {s['no_match_share']:.3f}  top-1 median {s['top1_median']:.3f}  "
              f"own F1-opt {s['f1_optimal']['F1']:.3f} (TNR {s['f1_optimal']['TNR']:.2f})  "
              f"with seed-42 thr: F1 {s['seed42_threshold']['F1']:.3f} TNR {s['seed42_threshold']['TNR']:.2f}")
    for key in TEST_EMB:
        print(f"{key}: top-1 median {res[key]['top1_median']:.3f}, refused at seed-42 threshold "
              f"{res[key]['refused_at_seed42_threshold']:.3f}, "
              f"threshold at rate 0.25 {res[key]['threshold_at_rate_0.25']:.4f}")
    print("rate   F1    minF1  TNR   Prec  Rec")
    for r, m in res["by_rate"].items():
        print(f"{r}  {m['mean_F1']:.3f} {m['min_F1']:.3f}  {m['mean_TNR']:.2f}  {m['mean_Precision']:.3f} "
              f"{m['mean_Recall']:.3f}")


if __name__ == "__main__":
    main()
